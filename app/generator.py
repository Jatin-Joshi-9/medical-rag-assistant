import os
from groq import Groq


class Generator:
    def __init__(self, api_key: str):

        self.client = Groq(api_key=api_key)
        self.model_id = "llama-3.3-70b-versatile"

    def get_answer(self, query: str, results: list) -> str:
        context_blocks = []
        for i, res in enumerate(results, 1):
            score = res.get("score", 0)
            context_blocks.append(
                f"[Source {i} | relevance: {score:.2f}]\n{res['content']}"
            )

        context_text = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a professional Medical Research Assistant.
Answer the question using ONLY the information in the provided context.

Rules:
- If the answer is not found in the context, respond with:
  "I could not find relevant information in the provided guidelines to answer this question."
- Do not use prior knowledge or make assumptions beyond what the context states.
- Cite the source number (e.g. [Source 1]) when referencing specific information.
- Be concise, accurate, and professional.

CONTEXT:
{context_text}

USER QUESTION:
{query}

MEDICAL ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical assistant that follows instructions strictly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"