"""Evaluation loop: Gemini-as-judge scoring for RAG answers, summaries,
and generated documents on accuracy, completeness, and relevance.

If the score is below EVAL_SCORE_THRESHOLD, the response is re-generated
with the judge's specific feedback, capped at EVAL_MAX_ITERATIONS.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from src.config import EVAL_MAX_ITERATIONS, EVAL_SCORE_THRESHOLD
from src.llm import invoke_text
from src.utils.logger import logger

JUDGE_PROMPT = """You are a strict legal QA evaluator. Evaluate the RESPONSE to the QUERY given \
the CONTEXT it was supposed to be grounded in.

Score each dimension from 1 (poor) to 5 (excellent):
- accuracy: is it factually consistent with the context, with no hallucinated claims?
- completeness: does it address all parts of the query?
- relevance: does it stay focused on the query without irrelevant content?

QUERY:
{query}

CONTEXT:
{context}

RESPONSE:
{response}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"accuracy": <1-5>, "completeness": <1-5>, "relevance": <1-5>, "feedback": "<one or two sentences of specific, actionable improvement notes; empty string if no improvement needed>"}}"""

REFINE_WITH_FEEDBACK_PROMPT = """Your previous response to the query below received evaluator \
feedback. Produce an improved response that addresses the feedback, still grounded ONLY in the \
provided context.

QUERY:
{query}

CONTEXT:
{context}

PREVIOUS RESPONSE:
{previous_response}

EVALUATOR FEEDBACK:
{feedback}

Improved response:"""


def _parse_judge_output(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {
            "accuracy": float(data.get("accuracy", 0)),
            "completeness": float(data.get("completeness", 0)),
            "relevance": float(data.get("relevance", 0)),
            "feedback": data.get("feedback", ""),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def score_response(query: str, context: str, response: str) -> dict | None:
    raw = invoke_text(JUDGE_PROMPT.format(query=query, context=context, response=response), temperature=0.0)
    if raw is None:
        return None

    parsed = _parse_judge_output(raw)
    if parsed is None:
        logger.warning("Judge output could not be parsed as JSON; skipping scoring.")
        return None

    parsed["average"] = round(
        (parsed["accuracy"] + parsed["completeness"] + parsed["relevance"]) / 3, 2
    )
    return parsed


def evaluate_with_feedback_loop(
    query: str,
    context: str,
    initial_response: str,
    regenerate: Callable[[str, str, str], str],
    threshold: float = EVAL_SCORE_THRESHOLD,
    max_iterations: int = EVAL_MAX_ITERATIONS,
) -> dict:
    """Score a response and, if below threshold, ask the judge for feedback and
    call `regenerate(query, previous_response, feedback)` to produce an improved response.
    """
    response = initial_response
    history = []

    for iteration in range(max_iterations + 1):
        result = score_response(query, context, response)

        if result is None:
            return {"response": response, "score": None, "history": history, "evaluated": False}

        history.append({"iteration": iteration, "response": response, "score": result})

        if result["average"] >= threshold or not result["feedback"] or iteration == max_iterations:
            return {"response": response, "score": result, "history": history, "evaluated": True}

        logger.info(f"Eval iteration {iteration}: score {result['average']} < {threshold}, refining.")
        response = regenerate(query, response, result["feedback"])

    return {"response": response, "score": None, "history": history, "evaluated": True}


def default_regenerate(query: str, context: str, previous_response: str, feedback: str) -> str:
    prompt = REFINE_WITH_FEEDBACK_PROMPT.format(
        query=query, context=context, previous_response=previous_response, feedback=feedback
    )
    improved = invoke_text(prompt)
    return improved if improved is not None else previous_response
