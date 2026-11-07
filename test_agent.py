"""Offline check for agent.py: a scripted fake LLM drives plan -> shell -> replan -> answer.

Run: python3 test_agent.py   (pytest also works)
"""

import tempfile
from pathlib import Path

import agent


def scripted_llm(replies):
    """Fake LLM: returns canned replies in order and records every prompt it saw."""
    queue = iter(replies)
    seen = []

    def llm(messages):
        seen.append(messages)
        return next(queue)

    return llm, seen


def test_plan_execute_replan():
    llm, seen = scripted_llm(
        [
            '```json\n{"steps": ["make file", "check it"]}\n```',  # 0 planner (fenced JSON)
            "sure, let me think",  # 1 executor: invalid -> retry
            '{"cmd": "echo hi > out.txt && cat out.txt"}',  # 2 executor: run command
            '{"result": "made out.txt"}',  # 3 executor: step done
            '{"steps": ["check it"]}',  # 4 replanner: continue
            '{"cmd": "touch denied.txt"}',  # 5 executor: user denies this
            '{"result": "skipped"}',  # 6 executor: step done
            '<think>goal met {ok}</think> {"answer": "out.txt says hi"}',  # 7 replanner: finish
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        answer = agent.run(
            "write hi to out.txt",
            llm=llm,
            approve=lambda cmd: "touch" not in cmd,
            workdir=tmp,
        )
        assert answer == "out.txt says hi"
        assert Path(tmp, "out.txt").read_text() == "hi\n"
        assert not Path(tmp, "denied.txt").exists()  # denied command never ran
    assert "hi" in seen[3][-1]["content"]  # command output fed back to LLM
    assert "denied" in seen[6][-1]["content"]  # denial fed back to LLM
    assert "made out.txt" in seen[7][-1]["content"]  # replanner sees completed work
    assert len(seen) == 8


if __name__ == "__main__":
    test_plan_execute_replan()
    print("ok")
