from typing import List, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. Use only the provided context and chat history. "
    "If the answer is not in the context, say you don't know."
)


FEW_SHOT_EXAMPLES = [
    {
        "user": "What is RAG?",
        "assistant": (
            "Retrieval-Augmented Generation (RAG) retrieves relevant documents and uses them as context "
            "for a language model to generate an answer."
        ),
    },
]


def build_prompt(
    system_prompt: Optional[str],
    summary: Optional[str],
    recent_messages: List[dict],  # [{"role": "user"|"assistant", "content": str}]
    retrieved_chunks: List[str],
    question: str,
) -> str:
    """Compose a prompt with optional system prompt, summary, few-shot, history and retrieved chunks."""

    sys = system_prompt or DEFAULT_SYSTEM_PROMPT

    parts: List[str] = [f"System: {sys}"]

    if FEW_SHOT_EXAMPLES:
        parts.append("\nFew-shot examples:")
        for ex in FEW_SHOT_EXAMPLES:
            parts.append(f"User: {ex['user']}")
            parts.append(f"Assistant: {ex['assistant']}")

    if summary:
        parts.append("\nConversation summary so far:")
        parts.append(summary)

    if recent_messages:
        parts.append("\nRecent conversation:")
        for m in recent_messages:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")

    if retrieved_chunks:
        parts.append("\nRetrieved context:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            parts.append(f"[{i}] {chunk}")

    parts.append("\nUser question:")
    parts.append(question)
    parts.append("\nAnswer:")

    return "\n".join(parts)


