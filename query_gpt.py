#!/usr/bin/env python3
"""
Dispatch a query to Codex CLI (no API tokens needed).
Usage:
    python query_gpt.py "your question"
    python query_gpt.py "your question" --model o4-mini
    echo "long prompt" | python query_gpt.py -
"""
import argparse
import subprocess
import sys


def query_codex(prompt: str, model: str | None = None) -> str:
    cmd = ["codex", "exec"]
    if model:
        cmd += ["-c", f'model="{model}"']
    cmd += ["-c", "approval=never", "-"]

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Codex error:\n{result.stderr}")

    import re
    # strip ANSI escape codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mA-Za-z]|\x1b\[[?][0-9;]*[hl]')
    clean = ansi_escape.sub("", result.stdout)

    # codex output: header block, then "user\n<prompt>", then "codex\n<answer>", then "tokens used\n<answer again>"
    # extract between last "codex\n" marker and "tokens used"
    match = re.search(r'codex\n(.*?)(?:tokens used|\Z)', clean, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: everything after the header dashes
    parts = clean.split("--------\n", maxsplit=2)
    return parts[-1].strip() if parts else clean.strip()


def main():
    parser = argparse.ArgumentParser(description="Query Codex from Claude Code")
    parser.add_argument("prompt", help="Prompt text, or '-' to read from stdin")
    parser.add_argument("--model", default=None, help="e.g. o4-mini, gpt-4o")
    args = parser.parse_args()

    prompt = sys.stdin.read().strip() if args.prompt == "-" else args.prompt
    result = query_codex(prompt, model=args.model)
    print(result)


if __name__ == "__main__":
    main()
