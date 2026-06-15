"""Deepeval tests for RAG answer quality."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM


class LocalJudge(DeepEvalBaseLLM):
    """Wraps Mercury as judge for deepeval metrics."""

    def __init__(self):
        from src.services.llm import LLM
        self._llm = LLM(profile="mercury")
        super().__init__("mercury-2")

    def load_model(self):
        return self._llm

    def generate(self, prompt: str) -> str:
        return asyncio.run(self._llm.chat(prompt, mode="black_box"))

    async def a_generate(self, prompt: str) -> str:
        return await self._llm.chat(prompt, mode="black_box")

    def get_model_name(self):
        return "mercury-2"


judge = LocalJudge()
faithfulness = FaithfulnessMetric(model=judge, threshold=0.5)
relevancy = AnswerRelevancyMetric(model=judge, threshold=0.5)

SAMPLE_CONTEXT = [
    "The Transformer is a novel architecture that relies entirely on self-attention, "
    "dispensing with recurrence and convolutions entirely.",
    "Self-attention computes a weighted sum of all positions in the input sequence, "
    "allowing the model to capture long-range dependencies.",
    "The key innovation is the multi-head attention mechanism, which allows the model "
    "to jointly attend to information from different representation subspaces.",
]


@pytest.mark.parametrize("question,expected_topic", [
    ("What does the Transformer architecture rely on?", "self-attention"),
    ("How does self-attention work?", "weighted sum"),
])
def test_answer_quality(question, expected_topic):
    ctx = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(SAMPLE_CONTEXT))
    prompt = f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer the question based only on the context."
    answer = asyncio.run(judge._llm.chat(prompt, mode="black_box",
                         system="Answer concisely based only on the context."))

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=SAMPLE_CONTEXT,
    )
    assert_test(test_case, [faithfulness, relevancy])
