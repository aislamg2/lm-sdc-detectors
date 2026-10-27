"""Prompt templates for the LM detectors.

Two context conditions (proposal §6.2):
  zero_shot: serialized observation only
  reference: observation + per-layer fault-free ranges from the fitting set
Two modes: thinking (generative) and non-thinking (single-pass, probe).
"""

SYSTEM_PROMPT = (
    "TODO: describe the task: a ResNet-18 classifier may have suffered a "
    "hardware bit flip; decide whether this run's output is silently corrupted. "
    "Answer with exactly 'yes' or 'no'."
)


def build_prompt(observation_text: str, condition: str, thinking: bool,
                 reference_text: str | None = None) -> str:
    """Assemble the chat-formatted prompt (Qwen3 /think or /no_think).

    TODO: implement; fix the answer format so the answer token position is known.
    """
    raise NotImplementedError
