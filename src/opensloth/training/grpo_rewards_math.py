"""Math reasoning reward functions for GRPO training.

These functions are ported from the Unsloth GRPO tutorial to evaluate
math problem-solving reasoning traces with proper formatting.
"""

import re
from typing import Dict, Any, List

from ..training.grpo_trainer import register_reward


# Regex patterns for answer extraction
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

# Format matching regex
solution_end_regex = r"</SOLUTION>[\s]{0,}"
match_format = re.compile(
    rf"{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)

# Number extraction regex
match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags=re.MULTILINE | re.DOTALL
)


@register_reward("match_format_exactly")
def match_format_exactly(prompts: List[str], completions: List[List[str]], **kwargs) -> List[float]:
    """Reward for exact format matching: reasoning + solution tags."""
    scores = []
    for completion_group in completions:
        for completion in completion_group:
            response = completion[0]["content"] if isinstance(completion, list) and completion else str(completion)
            score = 3.0 if match_format.search(response) is not None else 0.0
            scores.append(score)
    return scores


@register_reward("match_format_approximately")
def match_format_approximately(prompts: List[str], completions: List[List[str]], **kwargs) -> List[float]:
    """Reward for partial format matching: count individual format elements."""
    scores = []
    for completion_group in completions:
        for completion in completion_group:
            response = completion[0]["content"] if isinstance(completion, list) and completion else str(completion)
            score = 0.0
            
            # Count format elements (penalize if too many)
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            
            scores.append(score)
    return scores


@register_reward("check_answer")
def check_answer(prompts: List[str], completions: List[List[str]], answer: List[str], **kwargs) -> List[float]:
    """Reward for correct answers extracted from solution tags."""
    scores = []
    prompt_idx = 0
    
    for completion_group in completions:
        for completion in completion_group:
            response = completion[0]["content"] if isinstance(completion, list) and completion else str(completion)
            
            # Extract answer using regex
            guess_match = match_format.search(response)
            extracted_answer = guess_match.group(1) if guess_match else None
            
            score = 0.0
            true_answer = answer[prompt_idx] if prompt_idx < len(answer) else ""
            
            if extracted_answer is None:
                score = -2.0
            elif extracted_answer == true_answer:
                score = 5.0  # Exact match
            elif extracted_answer.strip() == true_answer.strip():
                score = 3.5  # Match ignoring whitespace
            else:
                # Try numeric comparison
                try:
                    ratio = float(extracted_answer) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score = 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score = 1.5
                    else:
                        score = -2.5
                except (ValueError, ZeroDivisionError):
                    score = -4.5
            
            scores.append(score)
        prompt_idx += 1
    
    return scores


@register_reward("check_numbers")
def check_numbers(prompts: List[str], completions: List[List[str]], answer: List[str], **kwargs) -> List[float]:
    """Reward for numeric answers extracted from solution."""
    scores = []
    prompt_idx = 0
    
    for completion_group in completions:
        for completion in completion_group:
            response = completion[0]["content"] if isinstance(completion, list) and completion else str(completion)
            
            # Extract first number from solution
            number_match = match_numbers.search(response)
            extracted_number = number_match.group(1) if number_match else None
            
            score = 0.0
            true_answer = answer[prompt_idx] if prompt_idx < len(answer) else ""
            
            if extracted_number is None:
                score = -2.5
            else:
                try:
                    true_num = float(true_answer.strip())
                    # Remove commas and convert
                    extracted_num = float(extracted_number.strip().replace(",", ""))
                    score = 3.5 if extracted_num == true_num else -1.5
                except (ValueError, TypeError):
                    score = 0.0
            
            scores.append(score)
        prompt_idx += 1
    
    return scores


@register_reward("reasoning_length")
def reasoning_length(prompt: str, completion: str, ctx: Dict[str, Any]) -> float:
    """Reward appropriate reasoning length - not too short, not too long."""
    # Extract reasoning section
    reasoning_match = re.search(
        rf"{reasoning_start}(.*?){reasoning_end}",
        completion,
        flags=re.MULTILINE | re.DOTALL
    )
    
    if reasoning_match is None:
        return -1.0
    
    reasoning_text = reasoning_match.group(1).strip()
    reasoning_words = len(reasoning_text.split())
    
    # Encourage moderate reasoning length
    if reasoning_words < 10:
        return -0.5  # Too short
    elif reasoning_words > 200:
        return -0.3  # Too long
    elif 20 <= reasoning_words <= 100:
        return 0.5   # Good length
    else:
        return 0.1   # Acceptable


@register_reward("step_indicators")
def step_indicators(prompt: str, completion: str, ctx: Dict[str, Any]) -> float:
    """Reward step-by-step reasoning indicators."""
    reasoning_match = re.search(
        rf"{reasoning_start}(.*?){reasoning_end}",
        completion,
        flags=re.MULTILINE | re.DOTALL
    )
    
    if reasoning_match is None:
        return 0.0
    
    reasoning_text = reasoning_match.group(1)
    score = 0.0
    
    # Look for step indicators
    step_patterns = [
        r"step \d+",
        r"\d+\.",
        r"first[,\s]",
        r"then[,\s]",
        r"next[,\s]",
        r"finally[,\s]",
        r"therefore[,\s]",
        r"so[,\s]",
    ]
    
    for pattern in step_patterns:
        if re.search(pattern, reasoning_text.lower()):
            score += 0.2
    
    return min(score, 1.0)  # Cap at 1.0


# Helper function to prepare rewards for math datasets
def prepare_math_rewards(dataset) -> List[str]:
    """Return list of math-specific reward functions."""
    return [
        "match_format_exactly",
        "match_format_approximately", 
        "check_answer",
        "check_numbers",
        "reasoning_length",
        "step_indicators",
    ]


__all__ = [
    "match_format_exactly",
    "match_format_approximately",
    "check_answer", 
    "check_numbers",
    "reasoning_length",
    "step_indicators",
    "prepare_math_rewards",
]
