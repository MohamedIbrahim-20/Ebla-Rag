from typing import List, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. Use the provided context and chat history. "
    "Pay special attention to personal information shared by the user (such as names, preferences, and background). "
    "Remember and refer to this information in future responses when relevant. "
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
    {
        "user": "My name is Alex and I'm interested in machine learning.",
        "assistant": "Nice to meet you, Alex! I'll remember that you're interested in machine learning. What specific aspects of machine learning would you like to explore?"
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


### Prompt Templates
# System: You are a helpful, concise AI assistant. Use only the provided context and chat history. If the answer is not in the context, say you don't know.

# Few-shot examples:
# User: What is RAG?
# Assistant: Retrieval-Augmented Generation (RAG) retrieves relevant documents and uses them as context for a language model to generate an answer.

# Conversation summary so far:
# The user asked about machine learning models and the assistant explained different types including neural networks and decision trees.

# Recent conversation:
# User: How does RAG work with LLMs?
# Assistant: RAG enhances LLMs by retrieving relevant documents from a knowledge base and providing them as context for the model to generate more accurate answers.

# Retrieved context:
# [1] RAG systems combine retrieval components with generative models. The retriever finds relevant documents from a corpus, and the generator creates answers based on these documents.
# [2] LLMs can be enhanced with RAG to provide up-to-date information and reduce hallucinations by grounding responses in factual sources.

# User question:
# What are the benefits of using RAG?

# Answer: