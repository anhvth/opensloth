"""
Reward function registry for GRPO training.
Provides pluggable reward functions for different types of tasks (math, coding, general QA, etc.)
"""

import re
from typing import Callable, Dict, List, Any
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Base class for reward functions."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    def _extract_completion_content(self, completion) -> str:
        """
        Extract content from completion in either format.
        
        Args:
            completion: Either a string (standard format) or list of message dicts (conversational format)
            
        Returns:
            Content string
        """
        if isinstance(completion, str):
            # Standard format: completion is already a string
            return completion
        elif isinstance(completion, list) and len(completion) > 0:
            # Conversational format: completion is a list of message dicts
            if isinstance(completion[0], dict) and "content" in completion[0]:
                return completion[0]["content"]
            else:
                return str(completion[0])  # Fallback to string conversion
        else:
            return ""
    
    @abstractmethod
    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        """
        Calculate rewards for completions.
        
        Args:
            prompts: List of prompts (format depends on dataset type)
            completions: List of completions (format depends on dataset type)
            **kwargs: Additional arguments (e.g., ground truth answers, metadata)
            
        Returns:
            List of reward scores (one per completion)
        """
        pass


class LengthPenaltyReward(RewardFunction):
    """Simple length-based reward - penalizes very short or very long responses."""
    
    def __init__(self, min_length: int = 20, max_length: int = 500, penalty: float = -1.0):
        super().__init__("length_penalty", "Penalizes responses that are too short or too long")
        self.min_length = min_length
        self.max_length = max_length
        self.penalty = penalty
    
    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        scores = []
        for completion in completions:
            content = self._extract_completion_content(completion)
            length = len(content)
            
            if length < self.min_length or length > self.max_length:
                scores.append(self.penalty)
            else:
                # Smooth reward based on length (peaked at middle range)
                normalized = (length - self.min_length) / (self.max_length - self.min_length)
                score = 1.0 - abs(normalized - 0.5) * 2  # Peak at 0.5, linearly decrease
                scores.append(score)
        
        return scores


class MathFormatReward(RewardFunction):
    """Reward function for math problems with specific format requirements."""
    
    def __init__(self, 
                 reasoning_start: str = "<start_working_out>",
                 reasoning_end: str = "<end_working_out>", 
                 solution_start: str = "<SOLUTION>",
                 solution_end: str = "</SOLUTION>"):
        super().__init__("math_format", "Rewards proper math reasoning format")
        self.reasoning_start = reasoning_start
        self.reasoning_end = reasoning_end
        self.solution_start = solution_start
        self.solution_end = solution_end
        
        # Compile regex for exact format matching
        solution_end_regex = r"</SOLUTION>[\s]{0,}(?:" + re.escape("</s>") + ")?"
        self.match_format = re.compile(
            rf"{reasoning_end}.*?{solution_start}(.+?){solution_end_regex}[\s]{{0,}}$",
            flags=re.MULTILINE|re.DOTALL
        )
    
    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        scores = []
        for completion in completions:
            s = 0.0
            resp = self._extract_completion_content(completion)
            
            # Exact format match gets highest reward
            if self.match_format.search(resp) is not None:
                s += 3.0
            else:
                # Partial format rewards
                s += 0.5 if resp.count(self.reasoning_end) == 1 else -1.0
                s += 0.5 if resp.count(self.solution_start) == 1 else -1.0
                s += 0.5 if resp.count(self.solution_end) == 1 else -1.0
            
            scores.append(s)
        
        return scores


class MathAnswerReward(RewardFunction):
    """Reward function that checks mathematical answer correctness."""
    
    def __init__(self, 
                 solution_start: str = "<SOLUTION>",
                 solution_end: str = "</SOLUTION>"):
        super().__init__("math_answer", "Rewards correct mathematical answers")
        self.solution_start = solution_start
        self.solution_end = solution_end
        
        # Regex to extract answers
        solution_end_regex = r"</SOLUTION>[\s]{0,}(?:" + re.escape("</s>") + ")?"
        self.match_format = re.compile(
            rf"{solution_end}.*?{solution_start}(.+?){solution_end_regex}[\s]{{0,}}$",
            flags=re.MULTILINE|re.DOTALL
        )
        
        self.match_numbers = re.compile(
            solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
            flags=re.MULTILINE|re.DOTALL
        )
    
    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        answers = kwargs.get("answer", [])
        if not answers:
            return [0.0] * len(completions)
        
        responses = [self._extract_completion_content(c) for c in completions]
        extracted = [m.group(1) if (m := self.match_format.search(r)) else None for r in responses]
        
        scores = []
        for guess, true in zip(extracted, answers):
            s = 0.0
            if guess is None:
                scores.append(-2.0)
                continue
                
            # Exact match
            if guess == true:
                s += 5.0
            elif guess.strip() == true.strip():
                s += 3.5
            else:
                # Try numerical comparison
                try:
                    ratio = float(guess) / float(true)
                    if 0.9 <= ratio <= 1.1:
                        s += 2.0
                    elif 0.8 <= ratio <= 1.2:
                        s += 1.5
                    else:
                        s -= 2.5
                except:
                    s -= 4.5
            
            scores.append(s)
        
        return scores


class MathNumberReward(RewardFunction):
    """Reward function that extracts and compares numerical answers."""
    
    def __init__(self, solution_start: str = "<SOLUTION>"):
        super().__init__("math_number", "Rewards correct numerical answers")
        self.solution_start = solution_start
        
        self.match_numbers = re.compile(
            solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
            flags=re.MULTILINE|re.DOTALL
        )
    
    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        answers = kwargs.get("answer", [])
        if not answers:
            return [0.0] * len(completions)
        
        responses = [self._extract_completion_content(c) for c in completions]
        extracted = [m.group(1) if (m := self.match_numbers.search(r)) else None for r in responses]
        
        scores = []
        for guess, true in zip(extracted, answers):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                true_f = float(true.strip())
                guess_f = float(guess.strip().replace(",", ""))
                scores.append(3.5 if guess_f == true_f else -1.5)
            except:
                scores.append(0.0)
        
        return scores


class CodeCorrectnessReward(RewardFunction):
    """Reward function for code generation tasks."""
    
    def __init__(self):
        super().__init__("code_correctness", "Rewards syntactically correct code")
    
    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        scores = []
        for completion in completions:
            content = self._extract_completion_content(completion)
            
            # Basic syntax checks
            score = 0.0
            
            # Check for balanced brackets/parentheses
            if self._balanced_brackets(content):
                score += 1.0
            
            # Check for proper indentation patterns (Python-like)
            if self._proper_indentation(content):
                score += 0.5
            
            # Check for common keywords/patterns
            if any(keyword in content.lower() for keyword in ['def ', 'class ', 'import ', 'return ']):
                score += 0.5
            
            scores.append(score)
        
        return scores
    
    def _balanced_brackets(self, text: str) -> bool:
        """Check if brackets/parentheses are balanced."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in text:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def _proper_indentation(self, text: str) -> bool:
        """Check for consistent indentation."""
        lines = text.split('\n')
        indent_stack = [0]
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                continue
                
            indent = len(line) - len(stripped)
            
            # Check if indentation is reasonable (0, 4, 8, etc.)
            if indent % 4 != 0:
                return False
        
        return True


class DemoReward(RewardFunction):
    """
    Demo reward function that provides periodic sample output during training.
    Similar to the demonstration function in the reference Unsloth GRPO tutorial.
    """
    
    def __init__(self, print_every: int = 5):
        super().__init__("demo_reward", "Provides demonstration output during training")
        self.print_every = print_every
        self.call_count = 0
    
    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        self.call_count += 1
        
        # Provide demonstration output every N calls
        if self.call_count % self.print_every == 0:
            self._print_sample_output(prompts, completions, **kwargs)
        
        # Return neutral scores (this is primarily for demonstration)
        return [0.0] * len(completions)
    
    def _print_sample_output(self, prompts, completions, **kwargs):
        """Print sample outputs for demonstration."""
        if not prompts or not completions:
            return
        
        print("=" * 80)
        print(f"🎲 GRPO Sample Output (Reward Call #{self.call_count})")
        print("=" * 80)
        
        # Show first prompt and completion
        try:
            # Extract prompt content
            if isinstance(prompts[0], list) and len(prompts[0]) > 0:
                # Conversational format
                if isinstance(prompts[0][-1], dict) and "content" in prompts[0][-1]:
                    prompt_text = prompts[0][-1]["content"]
                else:
                    prompt_text = str(prompts[0][-1])
            else:
                prompt_text = str(prompts[0])
            
            # Extract completion content
            completion_text = self._extract_completion_content(completions[0])
            
            # Get answer if available
            answer = kwargs.get("answer", ["N/A"])[0] if kwargs.get("answer") else "N/A"
            
            print(f"📝 Question:")
            print(f"{prompt_text[:300]}{'...' if len(prompt_text) > 300 else ''}")
            print(f"\n💡 Expected Answer: {answer}")
            print(f"\n🤖 Generated Response:")
            print(f"{completion_text}")
            print(f"\n📊 Stats:")
            print(f"  • Response Length: {len(completion_text)} chars")
            print(f"  • Question Length: {len(prompt_text)} chars")
            
            # Count generations in this batch
            print(f"  • Batch Size: {len(completions)} generations")
            
        except Exception as e:
            print(f"❌ Error displaying sample: {e}")
        
        print("=" * 80)


# Registry for reward functions
_REWARD_REGISTRY: Dict[str, RewardFunction] = {}


def register_reward_function(reward_func: RewardFunction) -> None:
    """Register a reward function."""
    _REWARD_REGISTRY[reward_func.name] = reward_func


def get_reward_function(name: str) -> RewardFunction:
    """Get a reward function by name."""
    if name not in _REWARD_REGISTRY:
        raise ValueError(f"Reward function '{name}' not found. Available: {list(_REWARD_REGISTRY.keys())}")
    return _REWARD_REGISTRY[name]


def list_reward_functions() -> List[str]:
    """List all available reward function names."""
    return list(_REWARD_REGISTRY.keys())


def get_reward_functions(names: List[str]) -> List[RewardFunction]:
    """Get multiple reward functions by name."""
    return [get_reward_function(name) for name in names]


# Register default reward functions
register_reward_function(LengthPenaltyReward())
register_reward_function(MathFormatReward())
register_reward_function(MathAnswerReward())
register_reward_function(MathNumberReward())
register_reward_function(CodeCorrectnessReward())
register_reward_function(DemoReward())


def create_reward_preset(task_type: str) -> List[str]:
    """Create a preset list of reward functions for common task types."""
    presets = {
        "math": ["math_format", "math_answer", "math_number", "demo_reward"],
        "code": ["code_correctness", "length_penalty", "demo_reward"],
        "general": ["length_penalty", "demo_reward"],
        "reasoning": ["math_format", "length_penalty", "demo_reward"]
    }
    
    if task_type not in presets:
        raise ValueError(f"Unknown task type '{task_type}'. Available: {list(presets.keys())}")
    
    return presets[task_type]


def get_chat_template_for_task(task_type: str, tokenizer_eos_token: str = "</s>") -> str:
    """Get appropriate chat template for different task types."""
    if task_type == "math":
        reasoning_start = "<think>"
        reasoning_end = "</think>"
        solution_start = "<SOLUTION>"
        solution_end = "</SOLUTION>"
        
        system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
        
        return (
            "{% if messages[0]['role'] == 'system' %}"
                "{{ messages[0]['content'] + eos_token }}"
                "{% set loop_messages = messages[1:] %}"
            "{% else %}"
                "{{ '" + system_prompt + "' + eos_token }}"
                "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
                "{% if message['role'] == 'user' %}"
                    "{{ message['content'] }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] + eos_token }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}{% endif %}"
        )
    elif task_type == "code":
        return (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "{{ message['content'] + eos_token }}"
                "{% elif message['role'] == 'user' %}"
                    "{{ 'User: ' + message['content'] + eos_token }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ 'Assistant: ' + message['content'] + eos_token }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
        )
    else:  # general
        return (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "{{ message['content'] + eos_token }}"
                "{% elif message['role'] == 'user' %}"
                    "{{ message['content'] + eos_token }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] + eos_token }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '' }}{% endif %}"
        )
