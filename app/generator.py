import os
from groq import Groq


class Generator:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model_id = "llama-3.3-70b-versatile"

    def get_answer(self, query: str, results: list) -> str:
        if not results:
            return "No relevant sources were retrieved for this query."

        top_score = results[0].get("score", 0)
        low_confidence = top_score < 0.5

        context_blocks = []
        for i, res in enumerate(results, 1):
            score = res.get("score", 0)
            page = res.get("metadata", {}).get("page", "?")
            source_tag = f"[Source {i} | Page {page} | score: {score:.3f}]"
            context_blocks.append(f"{source_tag}\n{res['content']}")

        context_text = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a professional Medical Research Assistant answering questions about clinical guidelines.

INSTRUCTIONS:
- Answer ONLY using the provided context. Do not use prior knowledge.
- Structure your response using this exact format:

## Summary
One or two sentence direct answer to the question.

## Details
Detailed explanation with inline citations like [Source 1], [Source 2].
Use bullet points for lists of rules, steps, or requirements.

## Sources referenced
List each source used as:
- [Source N] Page X — one line description of what this source contributed.

- If the answer is not in the context, respond with exactly:
  "I could not find relevant information in the provided guidelines to answer this question."
- Never fabricate information.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise medical guideline assistant. Always follow the given format exactly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            answer = response.choices[0].message.content

            if low_confidence:
                warning = (
                    "\n> [!WARNING] Low confidence: "
                    f"Top source score is {top_score:.3f} (below 0.5). "
                    "Answer may not be well-supported by the document.\n\n"
                )
                answer = warning + answer

            return answer

        except Exception as e:
            return f"Error generating answer: {str(e)}"