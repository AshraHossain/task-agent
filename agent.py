#!/usr/bin/env python3
"""Task planning + execution agent.

Loop: PLAN the goal into steps -> EXECUTE the next step with a shell tool
(the LLM picks commands and reads their output) -> REPLAN from the results ->
repeat until the replanner returns a final answer.

Stdlib only. Talks to any OpenAI-compatible endpoint via OPENAI_BASE_URL /
OPENAI_API_KEY (already exported for Nara Router). Model: --model or AGENT_MODEL.

    python3 agent.py "find the 3 largest files here"          # asks before every command
    python3 agent.py "write fib.py, run it" --workdir w --yes  # auto-approve: NOT sandboxed
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable

Messages = list[dict[str, str]]
LLM = Callable[[Messages], str]

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5.5")
MAX_STEPS = 10  # executed steps per run, replans included
MAX_TOOL_CALLS = 8  # shell commands per step
OUTPUT_LIMIT = 4000  # chars of command output fed back to the LLM
CMD_TIMEOUT = 120  # seconds per shell command

PLANNER = """You plan tasks. Break the user's goal into 1-6 concrete, verifiable steps.
Reply with one JSON object only: {"steps": ["...", "..."]}"""

EXECUTOR = """You carry out ONE step of a larger plan using a non-interactive bash shell.
Each turn, reply with exactly one JSON object:
  {"cmd": "<bash command>"}               to run a command (you will see its output), or
  {"result": "<what was done or found>"}  once the step is complete.
Keep commands short and safe. Work in the current directory."""

REPLANNER = """You track progress toward a goal. Given the goal, the completed steps with
their results, and the remaining plan, reply with one JSON object only:
  {"answer": "<final answer for the user>"}  if the goal is achieved (or impossible), or
  {"steps": ["...", "..."]}                   the updated remaining steps."""


def chat(messages: Messages, model: str = DEFAULT_MODEL) -> str:
    """One chat-completions call to an OpenAI-compatible endpoint."""
    base = os.environ.get("OPENAI_BASE_URL", "https://router.bynara.id/v1").rstrip("/")
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps({"model": model, "messages": messages}).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
            "User-Agent": "task-agent/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["choices"][0]["message"]["content"] or ""


def anthropic_chat(messages: Messages, model: str = "claude-haiku-5-5") -> str:
    """One Messages-API call to api.anthropic.com (native, not OpenAI-compatible)."""
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    body = {"model": model, "max_tokens": 1024, "messages": turns}
    if system:
        body["system"] = system
    request = urllib.request.Request(
        f"{base}/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "task-agent/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["content"][0]["text"] or ""


def parse_json(text: str) -> dict:
    """First JSON object in an LLM reply; tolerates prose, ```fences``` and <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"no JSON object in LLM reply: {text[:300]!r}")


def ask(llm: LLM, messages: Messages) -> dict:
    """LLM call that must yield a JSON object; one corrective retry."""
    reply = llm(messages)
    try:
        return parse_json(reply)
    except ValueError:
        nudge = [
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": "Invalid format. Reply with one JSON object only.",
            },
        ]
        return parse_json(llm(messages + nudge))


def steps_of(obj: dict) -> list[str]:
    return [str(step) for step in obj.get("steps") or []]


_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
_SAFE_ENV_KEYS = {"PATH", "HOME", "LANG", "TMPDIR", "USER", "SHELL", "TERM"}


def _safe_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}


def _sanitize(text: str) -> str:
    """Escape non-printable/control characters so the terminal can't be spoofed."""
    return "".join(c if c.isprintable() and c != "\\" else repr(c)[1:-1] for c in text)


def run_shell(cmd: str, workdir: str) -> str:
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT,
            check=False,  # non-zero exit is fed back to the LLM, not raised
            env=_safe_env(),
        )
        out = f"exit={proc.returncode}\n{proc.stdout}{proc.stderr}"
    except subprocess.TimeoutExpired:
        out = f"timed out after {CMD_TIMEOUT}s"
    return out[-OUTPUT_LIMIT:]


def confirm(cmd: str) -> bool:
    return input(f"    run `{_sanitize(cmd)}`? [y/N] ").strip().lower() in ("y", "yes")


def execute(
    step: str,
    goal: str,
    done: list[dict],
    llm: LLM,
    approve: Callable[[str], bool],
    workdir: str,
    log: Callable[[str], None],
) -> str:
    """Tool loop for one step: the LLM runs shell commands until it reports a result."""
    task = f"Goal: {goal}\nDone so far: {json.dumps(done)}\nYour step: {step}"
    msgs = [{"role": "system", "content": EXECUTOR}, {"role": "user", "content": task}]
    for _ in range(MAX_TOOL_CALLS):
        action = ask(llm, msgs)
        if "result" in action:
            return str(action["result"])
        cmd = action.get("cmd")
        if not cmd:
            out = 'Reply with {"cmd": ...} or {"result": ...}.'
        elif approve(cmd):
            log(f"    $ {_sanitize(cmd)}")
            out = run_shell(cmd, workdir)
            log("      " + _sanitize(out.strip()[:800]).replace("\n", "\n      "))
        else:
            out = "The user denied this command. Try another approach or report the step result."
        msgs = msgs + [
            {"role": "assistant", "content": json.dumps(action)},
            {"role": "user", "content": f"Output:\n{out}"},
        ]
    return f"gave up after {MAX_TOOL_CALLS} commands"


def run(
    goal: str,
    llm: LLM = chat,
    approve: Callable[[str], bool] = confirm,
    workdir: str = ".",
    log: Callable[[str], None] = print,
) -> str:
    """Plan -> execute -> replan until the replanner returns a final answer."""
    first = ask(
        llm, [{"role": "system", "content": PLANNER}, {"role": "user", "content": goal}]
    )
    plan = steps_of(first) or [goal]
    done: list[dict] = []
    for _ in range(MAX_STEPS):
        log("\nPLAN: " + " -> ".join(plan))
        log(f">> {plan[0]}")
        result = execute(plan[0], goal, done, llm, approve, workdir, log)
        log(f"   = {result}")
        done = done + [{"step": plan[0], "result": result}]
        progress = {"goal": goal, "completed": done, "remaining_plan": plan[1:]}
        update = ask(
            llm,
            [
                {"role": "system", "content": REPLANNER},
                {"role": "user", "content": json.dumps(progress)},
            ],
        )
        if "answer" in update:
            return str(update["answer"])
        plan = steps_of(update)
        if not plan:
            break
    return "Stopped without a final answer. Progress:\n" + json.dumps(done, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan-and-execute agent over an OpenAI-compatible LLM."
    )
    parser.add_argument("goal")
    parser.add_argument(
        "--workdir", default=".", help="directory commands run in (created if missing)"
    )
    parser.add_argument(
        "--model", default=None, help="model id (default: provider-specific)"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="openai = any OpenAI-compatible endpoint (OPENAI_BASE_URL/KEY, e.g. Nara Router); "
        "anthropic = api.anthropic.com direct, real spend (default: %(default)s)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="run commands without asking (not sandboxed: they run as you)",
    )
    args = parser.parse_args()
    if args.provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set.")
        llm = functools.partial(anthropic_chat, model=args.model or "claude-haiku-5-5")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit(
                "OPENAI_API_KEY is not set (OPENAI_BASE_URL picks the OpenAI-compatible endpoint)."
            )
        llm = functools.partial(chat, model=args.model or DEFAULT_MODEL)
    os.makedirs(args.workdir, exist_ok=True)
    approve = (lambda cmd: True) if args.yes else confirm
    try:
        answer = run(args.goal, llm, approve, args.workdir)
    except urllib.error.HTTPError as err:
        sys.exit(
            f"LLM request failed: HTTP {err.code} {err.read().decode(errors='replace')[:500]}"
        )
    except urllib.error.URLError as err:
        sys.exit(f"LLM endpoint unreachable: {err.reason}")
    except ValueError as err:
        sys.exit(f"LLM returned unusable output: {err}")
    print(f"\nANSWER:\n{answer}")


if __name__ == "__main__":
    main()
