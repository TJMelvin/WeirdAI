"""
Reasoning prompt helpers for Weird AI.

This module introduces the first lightweight reasoning-related utilities for
Weird AI. These functions do not train a reasoning model yet. Instead, they
help students explore how prompt structure can encourage a model to produce
intermediate reasoning steps before generating a final answer.

Lesson 8 connects the Weird AI parody assistant to the reasoning model portion
of the course by focusing on inference-time reasoning prompts.
"""


def build_reasoning_prompt(user_prompt):
     """
    Convert a normal user prompt into a general reasoning prompt.
    """
     return f"""You are Weird AI, a helpful assistant.

    Think step-by-step.

    Task:
    {user_prompt}

    Final Answer:
    """


def build_parody_reasoning_prompt(topic):
    """
    Build a reasoning prompt specifically for parody lyric generation.
    """
    return f"""You are Weird AI, a creative parody-writing assistant.

    Think step-by-step.

    Topic:
    {topic}

    Planning steps:
    1. Identify the topic.
    2. Identify funny ideas or exaggerations related to the topic.
    3. Identify possible rhymes that could work in the lyrics.
    4. Write the final parody lyrics.

    Final Parody:
    """



def build_closed_world_prompt(premises, question):
    """
    Build a prompt for closed-world reasoning.
    """
    premise_list = "\n".join(
        f"- {premise}" for premise in premises
    )

    return f"""You are Weird AI, a reasoning assistant.

    Use only the provided premises to answer the question.
    Do not use outside knowledge or make assumptions beyond these facts.

    Premises:
    {premise_list}

    Question:
    {question}

    Reasoning:

    Answer:
    """



def build_open_world_prompt(premises, question):
    """
    Build a prompt for open-world reasoning.
    """
    premise_list = "\n".join(
        f"- {premise}" for premise in premises
    )

    return f"""You are Weird AI, a reasoning assistant.

    You may use the provided premises and relevant background knowledge.
    Check whether any premises conflict with known facts.
    If you find a contradiction, explain it clearly.

    Premises:
    {premise_list}

    Question:
    {question}

    Reasoning:

    Answer:
    """

