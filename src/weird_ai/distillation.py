
"""
Distillation dataset utilities for Weird AI.

Chapter 8 introduces model distillation:

    teacher model -> synthetic training examples -> student model

For Weird AI, this means preparing teacher-generated parody lyrics and teacher
notes as reusable supervised training data. This module does not train a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import random
import math
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


@dataclass
class TeacherExample:
    prompt: str
    teacher_lyrics: str
    teacher_notes: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationRecord:
    prompt: str
    target_text: str
    full_text: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_teacher_example(example: dict) -> bool:
    """
    Validate that a raw teacher example contains usable fields.

    Required fields:
        prompt, teacher_lyrics, teacher_notes, quality_score
    """
    if not isinstance(example, dict):
        return False

    required_field = [
        "prompt",
        "teacher_lyrics",
        "teacher_notes",
        "quality_score"
    ]

    if not all(field in example for field in required_field):
        return False

    for field_name in ("prompt", "teacher_lyrics", "teacher_notes"):
        value = example[field_name]

        if not isinstance(value, str) or not value.strip():
            return False

    try:
        quality_score = float(example["quality_score"])
    except (TypeError, ValueError):
        return False

    if not math.isfinite(quality_score):
        return False

    return True


def teacher_example_from_dict(example: dict) -> TeacherExample:
    """Convert a valid dictionary into a TeacherExample."""
    if not validate_teacher_example(example):
        raise ValueError("Invalid teacher example.")

    required_fields = {
        "prompt",
        "teacher_lyrics",
        "teacher_notes",
        "quality_score",
    }

    metadata = {
        key: value
        for key, value in example.items()
        if key not in required_fields
    }

    return TeacherExample(
        prompt=example["prompt"].strip(),
        teacher_lyrics=example["teacher_lyrics"].strip(),
        teacher_notes=example["teacher_notes"].strip(),
        quality_score=float(example["quality_score"]),
        metadata=metadata,
    )


def format_teacher_response(
    teacher_lyrics: str,
    teacher_notes: str,
    *,
    include_thinking: bool = True,
) -> str:
    """Format teacher notes and lyrics as a distillation target."""
    lyrics = teacher_lyrics.strip()
    notes = teacher_notes.strip()

    if include_thinking:
        return f"<think>{notes}</think>\n\n{lyrics}".strip()

    return lyrics


def format_training_prompt(prompt: str) -> str:
    """Format a Weird AI prompt for distillation training."""
    return f"User: {prompt.strip()}\nAssistant:"


def build_full_text(prompt: str, target_text: str) -> str:
    """Combine formatted prompt and target text."""
    formatted_prompt = format_training_prompt(prompt)
    return f"{formatted_prompt}\n{target_text.strip()}"


def tokenize_text(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Tokenize text using an object with encode(text) -> list[int]."""
    input_ids = tokenizer.encode(text)

    if not isinstance(input_ids, list):
        raise TypeError("Tokenizer must return a list of token IDs.")

    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in input_ids
    ):
        raise ValueError("Token IDs must all be integers")

    return input_ids


def create_labels(
    input_ids: list[int],
    prompt_length: int,
    *,
    mask_prompt: bool = True,
    ignore_index: int = -100,
) -> list[int]:
    """
    Create training labels.

    If mask_prompt=True, prompt token labels should be replaced with ignore_index.
    """
    if not isinstance(prompt_length, int) or isinstance(prompt_length, bool):
        raise TypeError("prompt_length must be an integer.")

    if not 0 <= prompt_length <= len(input_ids):
        raise ValueError("prompt_length is outside the input_ids range.")

    labels = input_ids.copy()

    if mask_prompt:
        labels[:prompt_length] = [ignore_index] * prompt_length

    return labels


def build_distillation_record(
    example: TeacherExample,
    tokenizer: TokenizerLike,
    *,
    include_thinking: bool = True,
    mask_prompt: bool = True,
) -> DistillationRecord:
    """Build one processed distillation record."""
    target_text = format_teacher_response(
        example.teacher_lyrics,
        example.teacher_notes,
        include_thinking=include_thinking,
    )

    formatted_prompt = format_training_prompt(example.prompt)

    full_text = build_full_text(
        example.prompt,
        target_text,
    )

    prompt_text = formatted_prompt + "\n"

    prompt_ids = tokenize_text(tokenizer, prompt_text)
    input_ids = tokenize_text(tokenizer, full_text)

    prompt_length = len(prompt_ids)

    labels = create_labels(
        input_ids,
        prompt_length,
        mask_prompt=mask_prompt,
    )

    return DistillationRecord(
        prompt=example.prompt,
        target_text=target_text,
        full_text=full_text,
        input_ids=input_ids,
        labels=labels,
        prompt_length=prompt_length,
        metadata=example.metadata.copy(),
    )


def filter_teacher_examples(
    examples: list[TeacherExample],
    *,
    min_quality_score: float = 0.75,
    min_lyric_lines: int = 4,
) -> list[TeacherExample]:
    """Keep only high-quality teacher examples."""
    filtered = []

    for example in examples:
        if example.quality_score < min_quality_score:
            continue

        lyrics = example.teacher_lyrics.replace("\\n", "\n")

        lyric_lines = [
            line
            for line in lyrics.splitlines()
            if line.strip()
        ]

        if len(lyric_lines) < min_lyric_lines:
            continue

        filtered.append(example)

    return filtered


def split_records(
    records: list[DistillationRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[DistillationRecord], list[DistillationRecord]]:
    """Split records into train and validation sets."""
    if not 0 <= validation_fraction <= 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    shuffled_records = records.copy()
    random.Random(seed).shuffle(shuffled_records)

    validation_size = int(len(shuffled_records) * validation_fraction)

    if validation_size == 0:
        return shuffled_records, []

    train_records = shuffled_records[:-validation_size]
    validation_records = shuffled_records[-validation_size:]

    return train_records, validation_records


def record_to_json_dict(record: DistillationRecord) -> dict:
    """Convert a DistillationRecord into a JSON-serializable dictionary."""
    return asdict(record)


def export_jsonl(records: list[DistillationRecord], path: str | Path) -> None:
    """Export records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(
                record_to_json_dict(record),
                f,
                ensure_ascii=False,
            )
            f.write("\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into dictionaries."""
    records = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def summarize_distillation_records(records: list[DistillationRecord]) -> str:
    """Create a readable summary of distillation records."""
    if not records:
        return "No records available."

    record_count = len(records)

    average_input_length = (
        sum(len(record.input_ids) for record in records) / record_count
    )

    average_prompt_length = (
        sum(record.prompt_length for record in records) / record_count
    )

    average_target_length = (
        sum(
            len(record.input_ids) - record.prompt_length
            for record in records
        ) / record_count
    )

    return (
        "Distillation Record Summary\n"
        "---------------------------\n"
        f"Number of records: {record_count}\n"
        f"Average input length: {average_input_length:.2f} tokens\n"
        f"Average prompt length: {average_prompt_length:.2f} tokens\n"
        f"Average target length: {average_target_length:.2f} tokens"
    )

