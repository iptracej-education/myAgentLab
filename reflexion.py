"""Reflexion, faithful to Shinn et al. 2023. Single file, no store, no scoping.

  Actor (Ma)           LLM proposes an action/trajectory
  Evaluator (Me)        task-specific: env check / unit tests / LLM judge
  Self-Reflection (Msr) LLM converts (trajectory, feedback) -> verbal lesson

  Short-term memory     the current trajectory (in the prompt)
  Long-term memory      bounded buffer `mem`, capped at max_memories
                        (paper: 1-3), prepended to the actor's context
                        each trial. Sliding window: oldest evicted first.
                        Discarded when the task ends.

Run:  python3 reflexion.py            (offline, MockLLM, deterministic)
Real: Reflexion(llm=AnthropicLLM())   (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

import subprocess
import shutil

import random
# ------------------------------------------------------------------ LLM edge


# Interface (We are defining LLM type)
class LLM(Protocol):  # Duck Typing
    def ask_ai(self, system: str, prompt: str) -> str: ...


# This is NOT an inheritance. Just a new class for Claude Subscription
class ClaudeSubscriptionLLM:
    """Uses your active 'claude login' subscription via the official CLI."""

    def __init__(self, model: str = "claude-3-5-sonnet"):
        self.model = model
        if not shutil.which("claude"):
            raise RuntimeError(
                "Claude CLI not found! Install it via 'npm install -g @anthropic-ai/claude-code' "
                "and run 'claude login' first."
            )

    def ask_ai(self, system: str, prompt: str) -> str:
        full_prompt = f"System: {system}\n\nUser: {prompt}"

        # Runs 'claude -p "..."' as a local subprocess using your authenticated CLI session

        process = subprocess.run(
            ["claude", "-p", full_prompt], capture_output=True, text=True, check=True
        )
        return process.stdout.strip()


# ---------------------------------------------------------------- Reflexion

ACTOR_SYSTEM = (
    "[role:actor] Solve the task. Reflections from your previous failed "
    "attempts are provided; use them."
)
REFLECT_SYSTEM = (
    "[role:reflector] You are the self-reflection module. Given your failed "
    "attempt and the evaluator's feedback, write ONE short concrete lesson "
    "(a single sentence) that would prevent this failure next trial."
)

Evaluator = Callable[[str], tuple[bool, str]]  # attempt -> (success, feedback)


@dataclass
class Result:
    success: bool
    trials: int
    trajectory: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)


class Reflexion:
    def __init__(self, llm: LLM, max_trials: int = 10, max_memories: int = 3):
        self.llm = llm
        self.max_trials = max_trials
        self.max_memories = max_memories  # paper's Omega: 1-3

    def run(self, task: str, evaluate: Evaluator) -> Result:
        LongMemory: list[str] = []
        result = Result(success=False, trials=0)  # Fill the values in Class Result.

        for trial in range(1, self.max_trials + 1):
            print(f"\n==================== TRIAL {trial} ====================")
            result.trials = trial
            reflections = "\n".join(f"- {r}" for r in LongMemory) or "- (none)"

            # --- 1. ACTOR PHASE ---
            print("\n[1. ACTOR] Generating an attempt based on current memory...")
            actor_prompt = (
                f"Task: {task}\n\nReflections:\n{reflections}\n\nYour attempt:"
            )

            ResponseMessage = self.llm.ask_ai(ACTOR_SYSTEM, actor_prompt).strip()
            print(f'  └─> Actor Proposed Attempt: "{ResponseMessage}"')
            result.trajectory.append(ResponseMessage)

            # --- 2. EVALUATOR PHASE ---
            print("\n[2. EVALUATOR] Checking attempt against task requirements...")
            success, feedback = evaluate(ResponseMessage)
            print(f'  └─> Success: {success} | Feedback: "{feedback}"')

            if success:
                print("\n[SUCCESS] Goal achieved! Exiting loop.")
                result.success = True
                break

            # --- 3. REFLECTION PHASE ---
            print("\n[3. REFLECTOR] Attempt failed. Formulating verbal lesson...")
            reflector_prompt = (
                f"Task: {task}\nAttempt: {ResponseMessage}\nFeedback: {feedback}"
            )

            reflection = self.llm.ask_ai(REFLECT_SYSTEM, reflector_prompt).strip()
            print(f'  └─> Reflection Lesson: "{reflection}"')

            # --- 4. MEMORY MANAGEMENT ---
            LongMemory.append(reflection)
            LongMemory[:] = LongMemory[-self.max_memories :]  # sliding window
            result.reflections = list(LongMemory)
            print(
                f"  └─> Updated Memory Buffer ({len(LongMemory)}/{self.max_memories} slots used):"
            )
            for m_idx, m_text in enumerate(LongMemory, 1):
                print(f"       {m_idx}. {m_text}")

        return result


# ---------------------------------------------- Environment & Mock LLM


class CombinationLock:
    """Task Environment replacing the make_lock closure function."""

    def __init__(self, actions: list[str], length: int):
        # Equivalent to picking N random choices with replacement
        self.secret = [random.choice(actions) for _ in range(length)]

    def evaluate(self, attempt: str) -> tuple[bool, str]:
        actions = [a.strip() for a in attempt.split(",")]

        if actions == self.secret:
            return True, "unlocked"

        # Find the 1-based index of the first incorrect action
        k = next(
            (i + 1 for i, (a, s) in enumerate(zip(actions, self.secret)) if a != s),
            len(self.secret),
        )
        return False, f"first mismatch at position {k}"


# ---------------------------------------------------------------- Main Entry

if __name__ == "__main__":
    task = "Open the lock. Actions: press, pull, twist. Sequence length 3. Reply comma-separated."

    # 1. Instantiate environment object using Option B (Standard Class)

    possible_actions = ["press", "pull", "twist"]

    lock = CombinationLock(actions=possible_actions, length=3)
    print(f"Generated Secret: {lock.secret}")

    # 2. Instantiate Mock LLM and Reflexion runner
    reflexion_runner = Reflexion(llm=ClaudeSubscriptionLLM(), max_trials=10)  # type: ignore

    print(">>> Starting Reflexion Agent Loop <<<")

    # 3. Pass goal.evaluate directly as the evaluator callback
    final_result = reflexion_runner.run(task, evaluate=lock.evaluate)

    # 4. Final summary
    print("\n==================== FINAL RESULTS ====================")
    print(f"Task Solved : {final_result.success}")
    print(f"Total Trials: {final_result.trials}")
    print("\nTrajectory History:")
    for idx, att in enumerate(final_result.trajectory, 1):
        print(f"  Trial {idx}: {att}")

    print("\nFinal Memory State:")
    for ref in final_result.reflections:
        print(f"  - {ref}")
