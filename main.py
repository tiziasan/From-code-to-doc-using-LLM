#!/usr/bin/env python3
"""
Async arc42 generator + mypy-ready types (strict, fully patched)
-----------------------------------------------------------------
- asyncio with bounded concurrency
- Strong typing via TypedDict and precise OpenAI message param unions
- Exponential backoff retry for OpenAI errors
- Clear schema validation for memory JSON

USAGE
-----
  export OPENAI_API_KEY=sk-...
  pip install openai aiofiles tiktoken mypy types-requests
  python script5.py <repo_or_git_url> 

Run mypy:
  mypy script5.py

 mypy.ini (saved in project root)
-----------------------------------------
"""
# Enables postponed evaluation of type annotations, which is standard in Python 3.10+
# and allows for forward references in type hints, like in `SectionSpec`.
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
    Union,
    List,
    Literal,
    cast,
    TypedDict,
)

# Third-party library imports for asynchronous file I/O, token counting, and OpenAI API interaction.
import aiofiles
import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionDeveloperMessageParam,
)

# ──────────────── CONSTANTS & PATHS ───────────────────────────────
# This block defines global constants and file paths used throughout the script.
# Using a central place for configuration makes the script easier to manage.

BASE_DIR: Path = Path(__file__).resolve().parent
CONFIG_PATH: Path = BASE_DIR / "repomix.config.json"
MEMORY_PATH: Path = BASE_DIR / "new_memory_arc42.json"
FULL_CODE_PATH: Path = BASE_DIR / "full_code.txt"
LATEX_PATH: Path = BASE_DIR / "arc42_documentation.tex"

# Configuration for the OpenAI model and its token limit.
MODEL_NAME: str = "gpt-4o-mini-2024-07-18"
TOKEN_LIMIT: int = 128_000

# --- Global Singletons ---

# Initialize the tokenizer for counting prompt tokens.
# It tries to get the specific encoding for the model, but falls back to a common one.
try:
    enc = tiktoken.encoding_for_model(MODEL_NAME)
except Exception:
    enc = tiktoken.get_encoding("cl100k_base")

# Initialize the asynchronous OpenAI client. It reads the API key from environment variables.
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ──────────────── OPENAI MESSAGE TYPE ALIAS & HELPERS ─────────────
# This section simplifies the creation of OpenAI message dictionaries by providing
# type aliases and helper functions, improving both readability and type safety.

# `MessageParam` is a union of all possible message types for the OpenAI Chat API.
MessageParam = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionDeveloperMessageParam,
]

# Helper functions to create message dictionaries with the correct role.
def sys_msg(content: str) -> ChatCompletionSystemMessageParam:
    """Creates a system message dictionary."""
    return {"role": "system", "content": content}

def usr_msg(content: str) -> ChatCompletionUserMessageParam:
    """Creates a user message dictionary."""
    return {"role": "user", "content": content}

def asst_msg(content: str) -> ChatCompletionAssistantMessageParam:
    """Creates an assistant message dictionary."""
    return {"role": "assistant", "content": content}


# ──────────────── TYPED STRUCTURES ────────────────────────────────
# These `TypedDict` classes define the expected schema for the `memory.json` file.
# This allows `mypy` to statically check the code that accesses this data,
# preventing common errors like typos in keys or incorrect data types.
# Using `total=False` means that keys are optional.

class TextFormat(TypedDict):
    description: str

class TableFormat(TypedDict):
    columns: Sequence[str]
    caption: str

class DiagramFormat(TypedDict, total=False):
    type: str
    description: str

class StepsFormat(TypedDict):
    description: str

class ListFormat(TypedDict):
    items: Sequence[str]

class TreeFormat(TypedDict):
    description: str

# Defines the possible content formats for a documentation section.
class SectionFormat(TypedDict, total=False):
    text: TextFormat
    table: TableFormat
    diagram: DiagramFormat
    steps: StepsFormat
    list: ListFormat
    tree: TreeFormat
    examples: Dict[str, Any]

# Defines the specification for a documentation section or subsection.
# Note the recursive type hint on `subsections`, enabled by `from __future__ import annotations`.
class SectionSpec(TypedDict, total=False):
    title: str
    goal: str
    format: SectionFormat
    style: str
    optional: bool
    example: Dict[str, Any]
    subsections: Dict[str, "SectionSpec"]

# Defines global instructions for the AI model.
class GlobalGuidelines(TypedDict):
    objective: str
    formatting: Sequence[str]
    commitment: Sequence[str]
    code_analysis: str

# Defines the persona and output requirements for the AI.
class UserProfile(TypedDict):
    role: str
    preferred_language: str
    output_format: str
    writing_style: str
    target_audience: str
    include: Sequence[str]
    diagram_format: str

# The root TypedDict that represents the entire structure of the memory JSON file.
class Memory(TypedDict):
    global_guidelines: GlobalGuidelines
    latex_safety: Sequence[str]
    user_profile: UserProfile
    doc_template: Dict[str, SectionSpec]


# ──────────────── UTILITIES ───────────────────────────────────────

def ensure_memory_schema(raw: Mapping[str, Any]) -> Memory:
    """
    Validates that the loaded JSON has the required top-level keys.
    This is a simple runtime check to ensure the file structure is correct before
    casting it to the `Memory` TypedDict for static analysis by mypy.
    """
    required_top = {"global_guidelines", "latex_safety", "user_profile", "doc_template"}
    missing = required_top.difference(raw.keys())
    if missing:
        raise KeyError(f"Missing keys in memory JSON: {missing}")
    # `cast` tells mypy to trust that `raw` now conforms to the `Memory` type.
    return cast(Memory, raw)


def walk(tree: Mapping[str, SectionSpec]) -> Iterator[Tuple[str, SectionSpec]]:
    """
    Recursively walks the nested `doc_template` dictionary.
    It yields a tuple of (section_id, spec) for every section that has a 'goal'.
    This is used to create a flat list of all sections that need to be generated.
    """
    for sid, spec in tree.items():
        if "goal" in spec:
            yield sid, spec
        if "subsections" in spec:
            yield from walk(spec["subsections"])  # Recursive call for nested sections


def prompt_token_count(*parts: str) -> int:
    """Calculates the total number of tokens for all parts of a prompt."""
    return sum(len(enc.encode(p)) for p in parts)


async def call_openai_with_retry(
    messages: Sequence[MessageParam],
    model: str,
    *,
    max_retries: int = 3,
    initial_backoff: float = 2.0,
) -> str:
    """
    Calls the OpenAI API with an exponential backoff retry mechanism.
    This makes the script resilient to transient network or API errors.
    If all retries fail, it raises the last encountered exception.
    """
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            resp: ChatCompletion = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
            content_any: Any = resp.choices[0].message.content
            if content_any is None:
                raise ValueError("OpenAI returned empty content")
            content: str = cast(str, content_any)
            return content
        except Exception as exc:
            if attempt == max_retries:
                # If this was the last attempt, re-raise the exception.
                raise
            msg = (
                "⚠️  OpenAI error (attempt {}/{}): {} Retrying in {:.1f}s…"
            ).format(attempt, max_retries, exc, backoff)
            print(msg)
            await asyncio.sleep(backoff)
            backoff *= 2  # Double the backoff time for the next retry.
    # This line should be unreachable due to the `raise` in the loop.
    raise RuntimeError("Retry loop failed unexpectedly")


async def flatten_repo(repo: str) -> str:
    """
    Runs the `repomix` CLI tool to flatten a repository into a single string.
    It uses `asyncio.create_subprocess_exec` to run the command asynchronously,
    preventing it from blocking the event loop.
    """
    if not CONFIG_PATH.exists():
        raise SystemExit(f"❌ Config file missing: {CONFIG_PATH}")

    # Determine if the repo is a remote URL or a local path to build the correct command.
    is_remote = repo.startswith(("http://", "https://", "git@"))
    cmd = ["repomix", "--remote", repo, "-c", str(CONFIG_PATH)] if is_remote else ["repomix", repo, "-c", str(CONFIG_PATH)]

    print("   ↪", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    if proc.returncode != 0:
        # If repomix fails, exit with its error message.
        raise SystemExit(
            f"❌ Repomix error (code {proc.returncode}):\n{stderr_b.decode()}\n{stdout_b.decode()}"
        )

    # Read the flattened code from the output file.
    try:
        async with aiofiles.open(FULL_CODE_PATH, "r", encoding="utf-8") as f:
            data = await f.read()
            return cast(str, data)
    except FileNotFoundError as exc:
        raise SystemExit(f"❌ Repomix did not create {FULL_CODE_PATH.name}: {exc}")


async def generate_section(
    sid: str,
    spec: SectionSpec,
    code: str,
    profile: UserProfile,
    guidelines: GlobalGuidelines,
    latex_rules: Iterable[str],
    semaphore: asyncio.Semaphore,
) -> str:
    """
    Generates the content for a single documentation section using the OpenAI API.
    This function is the core of the AI generation process.
    """
    print(f"🔄 {sid:>4} — {spec['title']}")
    start = time.perf_counter()

    # --- 1. Construct the System Prompt ---
    # This prompt sets the context and rules for the AI model.
    system_global = f"""
ROLE  : You are {profile['role']} writing for {profile['target_audience']}.
LANG  : {profile['preferred_language']}
OUT   : {profile['output_format']}   — tone: {profile['writing_style']}
INCL  : {', '.join(profile['include'])}; diagrams → {profile['diagram_format']}

OBJECTIVE
  {guidelines['objective']}

FORMATTING
  {'; '.join(guidelines['formatting'])}

COMMITMENTS
  {'; '.join(guidelines['commitment'])}

POLICY
  {guidelines['code_analysis']}
""".strip()

    system_latex = "LATEX SAFETY\n" + "\n".join("• " + r for r in latex_rules)

    # --- 2. Construct the Assistant Prompt ---
    # This pre-fills the assistant's response, guiding it on the specific task.
    assistant_payload = f"""
SECTION {sid} — {spec['title']}
Goal: {spec['goal']}

Required artefacts (JSON):
{json.dumps(spec.get('format', {}), indent=2)}

Style hints:
{spec.get('style', '')}

CHECKLIST
[ ] produce tables / figures / steps listed above
[ ] Write an introduction paragraph describing the purpose of the section.
[ ] Generate content in LaTeX format as described in the format field.
[ ] Derive all details from the source code; do not invent fictitious elements.
[ ] Where diagrams are expected, describe or insert PlantUML or LaTeX figures.
[ ] Ensure the section can be validated by someone familiar with the codebase.
""".strip()

    # --- 3. Construct the User Prompt ---
    # This contains the main input data: the flattened source code.
    user_payload = f"### Flattened repository ###\n```plaintext\n{code}\n```"

    # --- 4. Token Check and API Call ---
    # Check if the total prompt size exceeds the model's token limit.
    total_tok = prompt_token_count(system_global, system_latex, assistant_payload, user_payload)
    if total_tok > TOKEN_LIMIT:
        print(f"⚠️  Skipped {sid} (prompt {total_tok} tokens > {TOKEN_LIMIT})")
        return ""

    messages: List[MessageParam] = [
        sys_msg(system_global),
        sys_msg(system_latex),
        asst_msg(assistant_payload),
        usr_msg(user_payload),
    ]

    # Use a semaphore to limit the number of concurrent API calls.
    # The `async with` block ensures a request is only made if the semaphore allows it.
    async with semaphore:
        try:
            content = await call_openai_with_retry(messages, MODEL_NAME)
        except Exception as exc:
            print(f"❌ OpenAI error in {sid}: {exc}")
            return ""

    dur = time.perf_counter() - start
    print(f"✔️  {sid} finished in {dur:.2f}s")
    # Return the generated content, prefixed with a LaTeX comment for traceability.
    return f"% {sid} — {spec['title']}\n{content}"


async def async_main() -> None:
    """The main entry point and orchestrator for the script."""
    # --- 0. Setup and Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate arc42 docs via Repomix CLI + OpenAI (async & typed)")
    parser.add_argument("repository", help="Local path or Git URL")
    parser.add_argument("--max-parallel", type=int, default=20, help="Maximum concurrent OpenAI calls (default: 3)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("❌ Environment variable OPENAI_API_KEY not set.")

    # --- 1. Flatten Repository ---
    print("🌀 Flattening repository …")
    code = await flatten_repo(args.repository)
    print(f"📄 {FULL_CODE_PATH.name} ({len(code):,} characters) ready")

    # --- 2. Load & Validate Memory JSON ---
    try:
        raw_memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"❌ Memory JSON not found: {MEMORY_PATH}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"❌ Invalid JSON in {MEMORY_PATH}: {exc}")

    # Validate and cast the loaded JSON to the `Memory` TypedDict.
    memory: Memory = ensure_memory_schema(raw_memory)

    guidelines = memory["global_guidelines"]
    profile = memory["user_profile"]
    template = memory["doc_template"]
    latex_rules = memory["latex_safety"]

    # --- 3. Prepare LaTeX Document ---
    # This list will hold all parts of the final LaTeX file.
    latex_parts: List[str] = [
        "\\documentclass{article}",
        "\\usepackage{graphicx}",
        "\\usepackage{longtable}",
        "\\usepackage{hyperref}",
        "\\usepackage{plantuml}",
        "\\begin{document}",
    ]

    # --- 4. Generate All Sections Concurrently ---
    semaphore = asyncio.Semaphore(args.max_parallel)
    tasks: List[asyncio.Task[str]] = []

    # Use the `walk` utility to find all sections that need generation.
    for sid, spec in walk(template):
        # Create an asyncio.Task for each section.
        tasks.append(asyncio.create_task(
            generate_section(sid, spec, code, profile, guidelines, latex_rules, semaphore)
        ))

    t0 = time.perf_counter()
    print(f"[MAIN] Launching {len(tasks)} tasks...")
    # `asyncio.gather` runs all tasks concurrently and waits for them to complete.
    sections = await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - t0
    print(f"[MAIN] All tasks done in {total_duration:.2f}s")

    # Add the generated content to the LaTeX parts, filtering out any empty strings from skipped sections.
    latex_parts.extend(filter(None, sections))
    latex_parts.append("\\end{document}")

    # --- 5. Write Final LaTeX Output ---
    async with aiofiles.open(LATEX_PATH, "w", encoding="utf-8") as f:
        await f.write("\n\n".join(latex_parts))

    print(f"✅ LaTeX written to {LATEX_PATH}")


# Standard Python entry point.
if __name__ == "__main__":
    try:
        # Run the main asynchronous function.
        asyncio.run(async_main())
    except KeyboardInterrupt:
        # Allows the user to gracefully exit the script with Ctrl+C.
        print("\nInterrupted by user.")