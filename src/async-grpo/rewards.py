import re
from collections import Counter

# --- Globals for Logging ---
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 10  # Print less frequently to avoid clutter

# --- Constants & Helpers from validation_service.py ---
# These are required for the reward functions to work correctly.

# Pattern to catch formatting tags, placeholders, and control codes
TOKEN_PATTERN_ALL = re.compile(
    r"(?:#[A-Fa-f0-9]{7}|#[A-Za-z]|\{[^{}]*\}|<[^>]*>|\[(?:\/?[A-Za-z0-9#]+|-)\]|(?:\r\n|\r|\n)|\\+[rn]|\u3000)"
)

# Pattern for programmatic placeholders like %s, %d, {{ 0 }}
PH_PATTERN = re.compile(r"%(?:[A-Za-z]\d*)|\{\{\s*\d+\s*\}\}")

# Pattern for untranslated Chinese characters
CHINESE_RE = re.compile(
    r"[\u2E80-\u2EFF\u2F00-\u2FDF\u3100-\u312F\u31A0-\u31BF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]|[\U00020000-\U0002FA1F]"
)


EDITORIAL_PLACEHOLDER_PATTERNS = [
    r"\[\s*…\s*\]",
    r"\[\s*\.\.\.\s*\]",
    r"\[\s*\[.*?\]\s*\]",  # Double brackets
    r"\[\s*\?\s*\]",  # Question mark placeholders
    r"\[\s*TBD\s*\]",  # To be determined
    r"\[\s*TODO\s*\]",  # TODO placeholders
]


def _strip_tokens_all(text: str) -> str:
    """Remove all control tokens from text."""
    return TOKEN_PATTERN_ALL.sub("", text)


def _canonical_key(tok: str) -> str:
    """Generate canonical key for token comparison."""
    if tok.startswith("{") and tok.endswith("}"):
        head = tok[1:-1].split(":")[0].upper()
        return f"{{{head}}}"
    if tok.startswith("<") and tok.endswith(">"):
        head = tok[1:-1].split("(")[0].upper()
        return f"<{head}>"
    return tok


# --- Helper to Extract Data ---
def _extract_translation_data(prompts, completions):
    """
    Extracts source text from the prompt and the translated text from the completion.
    """
    try:
        # 1. Extract source text from the user prompt
        user_prompt = prompts[0][-1]["content"]
        source_match = re.search(
            r"Source Text \(.*? → .*?\):\n(.*)", user_prompt, re.DOTALL
        )
        if not source_match:
            return None, None, "Could not find 'Source Text' in prompt."
        source_text = source_match.group(1).strip()

        # 2. Extract translation from the AI completion
        translations = []
        for completion in completions:
            ai_output = completion[0]["content"]
            translation_match = re.search(
                r"<translation>(.*?)</translation>", ai_output, re.DOTALL
            )
            if not translation_match:
                # If tag is missing, penalize heavily but still use the full content for other checks
                translations.append(ai_output.strip())
            else:
                translations.append(translation_match.group(1).strip())

        return source_text, translations, None
    except (IndexError, KeyError) as e:
        return None, None, f"Prompt/completion structure error: {e}"


# --- Reward Functions ---


def reward_placeholder_parity(prompts, completions, **kwargs):
    """
    Rewards models for maintaining the same set of programmatic placeholders (%s, {{0}}).
    Severity: Critical Error.
    """
    source_text, translations, error = _extract_translation_data(prompts, completions)
    if error:
        # Penalize if data extraction fails
        return [-3.0] * len(completions)

    src_clean = _strip_tokens_all(source_text)
    src_ph = Counter(PH_PATTERN.findall(src_clean))

    scores = []
    for translation in translations:
        tgt_clean = _strip_tokens_all(translation)
        tgt_ph = Counter(PH_PATTERN.findall(tgt_clean))

        if src_ph == tgt_ph:
            scores.append(1.0)  # Pass
        else:
            scores.append(-2.5)  # Fail (Critical)

    # Optional: Print for debugging
    global PRINTED_TIMES, PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(f"\n--- Placeholder Parity ---")
        print(f"Source: {source_text}")
        print(f"Translation: {translations[0]}")
        print(f"Source Placeholders: {dict(src_ph)}")
        print(
            f"Translation Placeholders: {dict(Counter(PH_PATTERN.findall(_strip_tokens_all(translations[0]))))}"
        )
        print(f"Score: {scores[0]}")
    PRINTED_TIMES += 1

    return scores


def reward_editorial_placeholders(prompts, completions, **kwargs):
    """
    Penalizes models for including unresolved editorial placeholders like [TBD], [...].
    Severity: Critical Error.
    """
    _, translations, error = _extract_translation_data(prompts, completions)
    if error:
        return [-3.0] * len(completions)

    scores = []
    for translation in translations:
        found_placeholders = []
        for pattern in EDITORIAL_PLACEHOLDER_PATTERNS:
            if re.search(pattern, translation, re.IGNORECASE):
                found_placeholders.append(pattern)

        if not found_placeholders:
            scores.append(1.0)  # Pass
        else:
            scores.append(-2.5)  # Fail (Critical)

    return scores


def reward_control_tokens(prompts, completions, **kwargs):
    """
    Rewards models for keeping control tokens and formatting tags consistent.
    Severity: Critical Error.
    """
    source_text, translations, error = _extract_translation_data(prompts, completions)
    if error:
        return [-3.0] * len(completions)

    src_tokens = TOKEN_PATTERN_ALL.findall(source_text)
    src_keys = Counter(_canonical_key(t) for t in src_tokens)

    scores = []
    for translation in translations:
        tgt_tokens = TOKEN_PATTERN_ALL.findall(translation)
        tgt_keys = Counter(_canonical_key(t) for t in tgt_tokens)

        if src_keys == tgt_keys:
            scores.append(1.5)  # Pass (higher reward for complex task)
        else:
            scores.append(-2.5)  # Fail (Critical)

    return scores


def reward_untranslated_characters(prompts, completions, **kwargs):
    """
    Penalizes models for leaving Chinese-script characters in the translation.
    Severity: Critical Error.
    """
    _, translations, error = _extract_translation_data(prompts, completions)
    if error:
        return [-3.0] * len(completions)

    scores = []
    for translation in translations:
        if CHINESE_RE.search(translation):
            scores.append(-2.5)  # Fail (Critical)
        else:
            scores.append(1.0)  # Pass

    return scores


def reward_glossary_consistency(prompts, completions, **kwargs):
    """
    Rewards models for correctly using glossary terms.
    NOTE: Requires `glossary_terms` to be passed in kwargs as a list of tuples `[(source, target), ...]`.
    Severity: Warning.
    """
    source_text, translations, error = _extract_translation_data(prompts, completions)
    glossary_terms = kwargs.get("glossary_terms", [])

    if error or not glossary_terms:
        # Return neutral score if no glossary or if data extraction fails
        return [0.0] * len(completions)

    scores = []
    for translation in translations:
        issues = 0
        for source_term, target_term in glossary_terms:
            # Check if source term is in source text (case-insensitive)
            if source_term.lower() in source_text.lower():
                # If so, the target term MUST be in the translation (case-insensitive)
                if target_term.lower() not in translation.lower():
                    issues += 1

        if issues == 0:
            scores.append(1.0)  # Pass
        else:
            # Penalize proportionally to the number of missed terms
            scores.append(-1.0 * issues)  # Fail (Warning)

    return scores


def reward_length_ratio(prompts, completions, **kwargs):
    """
    Penalizes translations that are excessively long compared to the source.
    Severity: Warning.
    """
    source_text, translations, error = _extract_translation_data(prompts, completions)
    if error:
        return [-3.0] * len(completions)

    scores = []
    for translation in translations:
        # Penalize if translation is more than 3x longer than source (and source is non-trivial)
        if len(translation) > len(source_text) * 3 and len(source_text) > 10:
            scores.append(-1.0)  # Fail (Warning)
        else:
            scores.append(0.5)  # Pass (Small positive reward)

    return scores

func_rewards = [
    reward_placeholder_parity,
    reward_editorial_placeholders,
    reward_control_tokens,
    reward_untranslated_characters,
    reward_glossary_consistency,
    reward_length_ratio,
]
__all__ = [
    "reward_placeholder_parity",
    "reward_editorial_placeholders",
    "reward_control_tokens",
    "reward_untranslated_characters",
    "reward_glossary_consistency",
    "reward_length_ratio",
]
