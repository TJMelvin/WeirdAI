"""
Best-of-N generation utilities for Weird AI.

This module connects the Chapter 4 idea of inference-time scaling to the
Weird AI project.

The core idea is:

    generate one candidate      -> maybe good, maybe not
    generate several candidates -> evaluate each one
    select the best candidate  -> better final output without retraining

Students complete this file as the applied assignment for this lesson.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class GenerationCandidate:
    """
    Store one generated candidate and its evaluation results.

    Attributes:
        text: The generated text.
        score: Overall evaluation score for the text.
        evaluation: Full evaluation dictionary returned by the evaluator.
        index: Candidate number, starting at 1.
    """

    text: str
    score: float
    evaluation: dict
    index: int


def build_generation_prompt(
    topic: str,
    style: str = "emo parody",
    line_count: int = 8,
) -> str:
    """
    Build a prompt for Weird AI parody generation.
    """

    prompt = f"""
Write a {line_count}-line {style} song about {topic}.

Requirements:
- Write exactly {line_count} lines of lyrics.
- Use a consistent rhyme scheme if possible.
- Make the lyrics creative, entertaining, and appropriate to the topic.
- Use vivid language and memorable phrases.
- Keep the lyrics coherent and easy to follow.
- Return only the lyrics, without an explanation or introduction.
"""

    return prompt.strip()


def generate_once(
    prompt: str,
    generation_func: GenerationFunction,
    *,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 120,
    **kwargs: Any,
) -> str:
    """
    Generate a single candidate response.
    """

    generated_text = generation_func(
        prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    return str(generated_text)


def generate_multiple(
    prompt: str,
    generation_func: GenerationFunction,
    *,
    num_candidates: int = 5,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 120,
    **kwargs: Any,
) -> list[str]:
    """
    Generate several candidate responses for the same prompt.
    """

    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")

    candidates = []

    for _ in range(num_candidates):
        candidate = generate_once(
            prompt,
            generation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        candidates.append(candidate)

    return candidates


def score_candidate(
    text: str,
    evaluation_func: EvaluationFunction,
    *,
    score_key: str = "overall_score",
) -> GenerationCandidate:
    """
    Evaluate a single generated candidate.
    """

    evaluation = evaluation_func(text)

    score = float(evaluation.get(score_key, 0.0))

    return GenerationCandidate(
        text=text,
        score=score,
        evaluation=evaluation,
        index=0,
    )


def rank_candidates(
    texts: Iterable[str],
    evaluation_func: EvaluationFunction,
    *,
    score_key: str = "overall_score",
) -> list[GenerationCandidate]:
    """
    Evaluate and rank generated candidates from best to worst.
    """

    candidates = []

    for index, text in enumerate(texts, start=1):
        candidate = score_candidate(
            text,
            evaluation_func,
            score_key=score_key,
        )

        candidate.index = index
        candidates.append(candidate)

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    return ranked_candidates


def select_best_candidate(
    candidates: list[GenerationCandidate],
) -> GenerationCandidate:
    """
    Select the best candidate from a ranked or unranked list.
    """

    if not candidates:
        raise ValueError("Cannot select the best candidate from an empty list.")

    return max(
        candidates,
        key=lambda candidate: candidate.score,
    )


def generate_best(
    prompt: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    num_candidates: int = 5,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 120,
    score_key: str = "overall_score",
    **kwargs: Any,
) -> dict:
    """
    Generate multiple candidates, evaluate them, and return the best result.
    """

    texts = generate_multiple(
        prompt,
        generation_func,
        num_candidates=num_candidates,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    ranked_candidates = rank_candidates(
        texts,
        evaluation_func,
        score_key=score_key,
    )

    best_candidate = select_best_candidate(ranked_candidates)

    return {
        "best_text": best_candidate.text,
        "best_score": best_candidate.score,
        "best_candidate": best_candidate,
        "candidates": ranked_candidates,
    }


def format_candidate_report(
    candidates: list[GenerationCandidate],
) -> str:
    """
    Format candidate scores as a readable report.
    """

    sections = []

    for candidate in candidates:
        evaluation = candidate.evaluation

        section = [
            f"Candidate {candidate.index}",
            f"Overall Score: {candidate.score:.2f}",
        ]

        score_names = (
            "rhyme_score",
            "syllable_consistency_score",
            "syllable_score",
            "structure_score",
        )

        for score_name in score_names:
            if score_name in evaluation:
                section.append(
                    f"{score_name.replace('_', ' ').title()}: "
                    f"{float(evaluation[score_name]):.2f}"
                )

        section.append("")
        section.append(candidate.text)

        sections.append("\n".join(section))

    return "\n\n".join(sections)