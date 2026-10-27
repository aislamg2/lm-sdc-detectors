"""Thin wrapper over llama.cpp (llama-cpp-python) for Qwen3, 4-bit GGUF.

Must expose what the detectors need (Action item: "confirm you can read
yes/no probabilities and hidden states"):
  * next-token logprobs for " yes" / " no" at a given position
  * thinking-mode generation, then logprobs at the final-answer position
  * the final hidden state at the answer position (embedding=True / llama_get_embeddings)
"""


class QwenClient:
    def __init__(self, gguf_path: str, n_ctx: int = 2048, n_threads: int = 4):
        # TODO: load llama_cpp.Llama(model_path=..., logits_all=True, embedding=...).
        raise NotImplementedError

    def yes_no_probs(self, prompt: str) -> tuple[float, float]:
        """P(yes), P(no) for the next token after `prompt`. TODO: implement."""
        raise NotImplementedError

    def generate_then_answer(self, prompt: str, max_tokens: int = 512) -> tuple[str, float | None, float | None]:
        """Thinking mode: return (reasoning, P(yes), P(no)) at the answer token,
        or (reasoning, None, None) if no parseable answer. TODO: implement."""
        raise NotImplementedError

    def answer_hidden_state(self, prompt: str):
        """Final-layer hidden state at the answer position (numpy array). TODO: implement."""
        raise NotImplementedError
