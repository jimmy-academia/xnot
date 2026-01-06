#!/usr/bin/env python3
"""Self-Ask prompting baseline.

Reference: Measuring and Narrowing the Compositionality Gap in Language Models
Press et al., EMNLP 2022
https://arxiv.org/abs/2210.03350

Approach:
The model explicitly asks itself follow-up questions, answers them,
and uses intermediate answers to reach the final conclusion.
"""

from .base import BaseMethod
from utils.llm import call_llm
from utils.parsing import parse_final_answer


SYSTEM_PROMPT = """You are evaluating a restaurant for a user's request using the Self-Ask method.

Format your response EXACTLY as:
Question: [restate the evaluation question]
Are follow-up questions needed here: Yes
Follow-up: [first sub-question about the restaurant]
Intermediate answer: [answer based on reviews]
Follow-up: [second sub-question if needed]
Intermediate answer: [answer based on reviews]
So the final answer is: ANSWER: [1, 0, or -1]

Use 1 (recommend), 0 (neutral), or -1 (not recommend)."""

SYSTEM_PROMPT_RANKING = """You are selecting the best restaurant using the Self-Ask method.

Format your response as:
Question: Which restaurant best matches the user's request?
Are follow-up questions needed here: Yes
Follow-up: [question about key criteria]
Intermediate answer: [analysis]
...
So the final answer is: ANSWER: [restaurant number]"""


class SelfAsk(BaseMethod):
    """Self-Ask: model asks and answers its own follow-up questions."""

    name = "selfask"

    def __init__(self, run_dir: str = None, **kwargs):
        super().__init__(run_dir=run_dir, **kwargs)

    def evaluate(self, query: str, context: str) -> int:
        """Self-Ask evaluation with follow-up questions.

        Args:
            query: User request text
            context: Restaurant data
        """
        prompt = f"""[RESTAURANT INFO]
{context}

[USER REQUEST]
{query}

[EVALUATION]
Question: Should this restaurant be recommended for the user's request?
Are follow-up questions needed here:"""

        response = call_llm(prompt, system=SYSTEM_PROMPT)
        return parse_final_answer(response)

    # --- Ranking Methods ---

    def evaluate_ranking(self, query: str, context: str, k: int = 1) -> str:
        """Ranking with Self-Ask approach.

        Args:
            query: User request text
            context: All restaurants formatted
        """
        prompt = f"""[RESTAURANTS]
{context}

[USER REQUEST]
{query}

[EVALUATION]
Question: Which restaurant best matches the user's request?
Are follow-up questions needed here:"""

        return call_llm(prompt, system=SYSTEM_PROMPT_RANKING)
