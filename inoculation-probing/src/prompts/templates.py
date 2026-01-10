"""Prompt templates for inoculation experiments."""

from typing import Dict


PROMPT_TEMPLATES: Dict[str, str] = {
    # Baseline (no prompt)
    'baseline': "{text}",

    # Inoculation prompts - Imperative style
    'inoc_imperative_v1': "Ignore the length of this review and focus only on sentiment.\n\n{text}",

    'inoc_imperative_v2': "Focus on the emotional tone, not the length of this review.\n\n{text}",

    'inoc_imperative_v3': "Analyze sentiment regardless of review length.\n\n{text}",

    # Inoculation prompts - Descriptive style
    'inoc_descriptive_v1': "The following review's length is not indicative of its sentiment.\n\n{text}",

    'inoc_descriptive_v2': "Review length and sentiment are independent in this case.\n\n{text}",

    # Inoculation prompts - Role-playing style
    'inoc_role_v1': "You are an unbiased sentiment analyzer. Review length does not determine sentiment.\n\n{text}",

    'inoc_role_v2': "You are evaluating whether the reviewer liked the movie, not how much they wrote.\n\n{text}",

    # Inoculation prompts - Explanation style
    'inoc_explanation_v1': "Long reviews are not necessarily negative, and short reviews are not necessarily positive. Analyze the sentiment.\n\n{text}",

    # Control prompts - Generic task framing
    'control_generic_v1': "Analyze the following movie review.\n\n{text}",

    'control_generic_v2': "What is the sentiment of this review?\n\n{text}",

    # Control prompts - Attention spreading
    'control_spread': "Focus on all aspects of this review.\n\n{text}",

    # Anti-inoculation prompts (should make it worse)
    'anti_inoc_v1': "Pay attention to how long this review is.\n\n{text}",

    'anti_inoc_v2': "Consider both the length and sentiment of this review.\n\n{text}",

    # Ablation prompts - Component analysis
    'ablation_focus_sentiment': "Focus on the emotional tone of this review.\n\n{text}",

    'ablation_ignore_length': "Ignore the length of this review.\n\n{text}",

    # Llama chat format versions (for proper instruction following)
    'inoc_imperative_chat': "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nIgnore review length when analyzing sentiment.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>",

    'inoc_role_chat': "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an unbiased sentiment analyzer. Review length does not determine sentiment.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>",
}


def format_prompt(prompt_name: str, text: str) -> str:
    """
    Format a prompt template with the given text.

    Args:
        prompt_name: Name of the prompt template
        text: Text to insert into the template

    Returns:
        Formatted prompt string
    """
    if prompt_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {prompt_name}. Available: {list(PROMPT_TEMPLATES.keys())}")

    return PROMPT_TEMPLATES[prompt_name].format(text=text)


def get_prompt_categories() -> Dict[str, list]:
    """
    Get prompts organized by category for easy experimentation.

    Returns:
        Dictionary mapping categories to prompt names
    """
    return {
        'baseline': ['baseline'],
        'inoculation_imperative': ['inoc_imperative_v1', 'inoc_imperative_v2', 'inoc_imperative_v3'],
        'inoculation_descriptive': ['inoc_descriptive_v1', 'inoc_descriptive_v2'],
        'inoculation_role': ['inoc_role_v1', 'inoc_role_v2'],
        'inoculation_explanation': ['inoc_explanation_v1'],
        'control': ['control_generic_v1', 'control_generic_v2', 'control_spread'],
        'anti_inoculation': ['anti_inoc_v1', 'anti_inoc_v2'],
        'ablation': ['ablation_focus_sentiment', 'ablation_ignore_length'],
        'chat_format': ['inoc_imperative_chat', 'inoc_role_chat']
    }


def get_mvp_prompts() -> list:
    """
    Get minimal set of prompts for MVP experiment.

    Returns:
        List of prompt names for initial experiment
    """
    return [
        'baseline',
        'inoc_imperative_v2',  # Best imperative style
        'inoc_role_v1',        # Best role-playing style
        'anti_inoc_v1'         # Anti-inoculation control
    ]
