#!/usr/bin/env python3
"""
Example of using custom reward functions with decentralized veRL.

This example shows how to implement verifiable reward functions
for tasks like math problems, code execution, and format adherence.
"""

import asyncio
import re
import logging
from typing import Callable, Dict, Any

from decentralized_verl.workers.reward import (
    RewardWorker,
    VerifiableRewardFunctions,
)
from decentralized_verl.core.config import DecentralizedConfig, AlgorithmConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom reward functions

def math_problem_reward(prompt: str, response: str) -> float:
    """
    Reward function for math problems.

    Checks if the response contains the correct numerical answer.
    """
    # Extract expected answer from prompt (format: "Answer: X")
    expected_match = re.search(r"(?:answer|result).*?(\d+(?:\.\d+)?)", prompt.lower())
    if not expected_match:
        return 0.0

    expected = expected_match.group(1)

    # Look for boxed answer first (LaTeX style)
    boxed = re.search(r"\\boxed{([^}]+)}", response)
    if boxed:
        answer = boxed.group(1).strip()
        return 1.0 if answer == expected else 0.0

    # Look for "answer is X" or "= X"
    answer_patterns = [
        r"answer\s*(?:is|=)\s*(\d+(?:\.\d+)?)",
        r"=\s*(\d+(?:\.\d+)?)\s*$",
        r"(\d+(?:\.\d+)?)\s*$",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1)
            return 1.0 if answer == expected else 0.0

    return 0.0


def code_correctness_reward(prompt: str, response: str) -> float:
    """
    Reward function for code generation.

    Extracts code from response and checks if it executes without errors.
    Optionally runs test cases if provided in the prompt.
    """
    # Extract code block
    code_patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
        r"def \w+\(.*?\):.*?(?=\n\n|\Z)",
    ]

    code = None
    for pattern in code_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1) if "```" in pattern else match.group(0)
            break

    if not code:
        return 0.0

    # Try to execute
    try:
        # Create safe execution environment
        safe_globals = {
            "__builtins__": {
                "range": range,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "print": lambda *args: None,  # Disable print
            }
        }

        exec(compile(code, "<string>", "exec"), safe_globals)

        # Check if test cases in prompt
        test_match = re.search(r"test.*?:\s*(.+)", prompt, re.IGNORECASE | re.DOTALL)
        if test_match and "assert" in response:
            # Has assertions that passed
            return 1.0

        return 0.8  # Code runs but no test verification

    except SyntaxError:
        return 0.1  # Partial credit for attempt
    except Exception:
        return 0.2  # Code has errors


def format_adherence_reward(prompt: str, response: str) -> float:
    """
    Reward function for format adherence.

    Checks if response follows required formatting (headers, bullet points, etc.)
    """
    reward = 0.0
    checks = 0

    # Check for requested sections
    if "introduction" in prompt.lower():
        checks += 1
        if re.search(r"(introduction|overview|background)", response.lower()):
            reward += 1.0

    if "conclusion" in prompt.lower():
        checks += 1
        if re.search(r"(conclusion|summary|finally)", response.lower()):
            reward += 1.0

    # Check for bullet points if requested
    if "bullet" in prompt.lower() or "list" in prompt.lower():
        checks += 1
        if re.search(r"^[\s]*[-*•]\s", response, re.MULTILINE):
            reward += 1.0

    # Check for numbered list if requested
    if "numbered" in prompt.lower() or "steps" in prompt.lower():
        checks += 1
        if re.search(r"^\s*\d+[.)]\s", response, re.MULTILINE):
            reward += 1.0

    # Check length constraints
    word_match = re.search(r"(\d+)\s*words?", prompt.lower())
    if word_match:
        target = int(word_match.group(1))
        actual = len(response.split())
        checks += 1
        # Allow 20% deviation
        if abs(actual - target) / target <= 0.2:
            reward += 1.0
        elif abs(actual - target) / target <= 0.5:
            reward += 0.5

    return reward / max(checks, 1)


def helpfulness_reward(prompt: str, response: str) -> float:
    """
    Heuristic reward for helpfulness.

    Combines multiple signals to estimate response quality.
    """
    reward = 0.0

    # Length appropriateness (not too short, not too long)
    words = len(response.split())
    if 50 <= words <= 500:
        reward += 0.2
    elif 20 <= words < 50 or 500 < words <= 1000:
        reward += 0.1

    # Relevance (word overlap with prompt)
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    overlap = len(prompt_words & response_words)
    reward += min(overlap * 0.05, 0.2)

    # Completeness (ends with proper punctuation)
    if response.strip() and response.strip()[-1] in ".!?":
        reward += 0.1

    # Coherence (no excessive repetition)
    sentences = response.split(".")
    if len(sentences) > 1:
        unique_ratio = len(set(sentences)) / len(sentences)
        reward += unique_ratio * 0.2

    # Structure (has paragraph breaks for long responses)
    if words > 100 and "\n\n" in response:
        reward += 0.1

    # Hedging appropriately (shows uncertainty where appropriate)
    hedge_words = ["might", "could", "perhaps", "possibly", "likely"]
    if any(word in response.lower() for word in hedge_words):
        reward += 0.1

    return min(reward, 1.0)


class CompositeRewardFunction:
    """
    Combines multiple reward functions with configurable weights.
    """

    def __init__(self, reward_functions: Dict[str, tuple]):
        """
        Args:
            reward_functions: Dict of {name: (function, weight)}
        """
        self.reward_functions = reward_functions

    def __call__(self, prompt: str, response: str) -> float:
        total_reward = 0.0
        total_weight = 0.0

        for name, (func, weight) in self.reward_functions.items():
            try:
                reward = func(prompt, response)
                total_reward += reward * weight
                total_weight += weight
            except Exception as e:
                logger.warning(f"Reward function {name} failed: {e}")

        return total_reward / max(total_weight, 1.0)


async def demo_reward_functions():
    """Demonstrate reward functions on sample inputs."""

    print("=" * 60)
    print("Custom Reward Functions Demo")
    print("=" * 60)

    # Math problem
    print("\n1. Math Problem Reward:")
    prompt1 = "What is 15 + 27? The answer is 42."
    response1 = "Let me calculate: 15 + 27 = 42. The answer is 42."
    reward1 = math_problem_reward(prompt1, response1)
    print(f"   Prompt: {prompt1}")
    print(f"   Response: {response1}")
    print(f"   Reward: {reward1:.2f}")

    # Code correctness
    print("\n2. Code Correctness Reward:")
    prompt2 = "Write a function to calculate factorial."
    response2 = """Here's a factorial function:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```
"""
    reward2 = code_correctness_reward(prompt2, response2)
    print(f"   Prompt: {prompt2}")
    print(f"   Response: (factorial function)")
    print(f"   Reward: {reward2:.2f}")

    # Format adherence
    print("\n3. Format Adherence Reward:")
    prompt3 = "Write a 100-word summary with an introduction and conclusion."
    response3 = """Introduction: AI is transforming our world.

Machine learning enables computers to learn from data without explicit programming.
Deep learning uses neural networks with many layers to achieve remarkable results
in image recognition, natural language processing, and game playing.

Conclusion: AI continues to advance rapidly, promising both opportunities and challenges."""
    reward3 = format_adherence_reward(prompt3, response3)
    print(f"   Prompt: {prompt3}")
    print(f"   Reward: {reward3:.2f}")

    # Composite reward
    print("\n4. Composite Reward:")
    composite = CompositeRewardFunction({
        "helpfulness": (helpfulness_reward, 0.5),
        "format": (format_adherence_reward, 0.3),
        "length": (lambda p, r: min(len(r.split()) / 100, 1.0), 0.2),
    })

    prompt4 = "Explain machine learning in simple terms."
    response4 = """Machine learning is a type of artificial intelligence that allows computers
to learn from experience without being explicitly programmed. Instead of following
pre-written rules, ML systems find patterns in data and use these patterns to make
predictions or decisions. For example, email spam filters learn to identify spam by
analyzing millions of emails marked as spam or not spam."""

    reward4 = composite(prompt4, response4)
    print(f"   Prompt: {prompt4}")
    print(f"   Composite Reward: {reward4:.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_reward_functions())
