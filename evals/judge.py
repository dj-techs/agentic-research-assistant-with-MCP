"""LLM-as-judge for open-ended answers.

Uses Claude with structured output to score on a 1-5 scale against a rubric,
plus a binary pass/fail. Temperature 0 for stability.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from research_assistant.config import SETTINGS


class Verdict(BaseModel):
    score: int = Field(..., ge=1, le=5, description="1=wrong, 3=partial, 5=fully correct")
    pass_: bool = Field(..., alias="pass")
    rationale: str

    model_config = {"populate_by_name": True}


JUDGE_SYSTEM = """\
You are a strict evaluator of an AI research assistant's answers. Compare the
ANSWER to the RUBRIC. Score 1-5 (5 = fully correct & complete). Mark pass=true
only if the answer is substantively correct and would satisfy a reasonable user.
Be calibrated: a partially-right answer is 3, not 4.
"""


def judge(query: str, answer: str, rubric: str) -> Verdict:
    model = ChatAnthropic(
        model=SETTINGS.judge_model,
        api_key=SETTINGS.anthropic_api_key,
        max_tokens=400,
        temperature=0,
    ).with_structured_output(Verdict)

    return model.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"QUERY:\n{query}\n\nRUBRIC:\n{rubric}\n\nANSWER:\n{answer}"
                ),
            },
        ]
    )
