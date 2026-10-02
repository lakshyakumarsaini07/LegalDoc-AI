from src.evaluation import evaluate as eval_module
from src.evaluation.evaluate import (
    _parse_judge_output,
    evaluate_with_feedback_loop,
    score_response,
)


def test_parse_judge_output_valid_json():
    raw = '{"accuracy": 4, "completeness": 3, "relevance": 5, "feedback": "add more detail"}'
    parsed = _parse_judge_output(raw)
    assert parsed == {"accuracy": 4.0, "completeness": 3.0, "relevance": 5.0, "feedback": "add more detail"}


def test_parse_judge_output_handles_surrounding_text():
    raw = 'Here is my evaluation:\n{"accuracy": 5, "completeness": 5, "relevance": 5, "feedback": ""}\nThanks.'
    parsed = _parse_judge_output(raw)
    assert parsed["accuracy"] == 5.0


def test_parse_judge_output_returns_none_for_garbage():
    assert _parse_judge_output("not json at all") is None


def test_score_response_returns_none_without_llm(monkeypatch):
    monkeypatch.setattr(eval_module, "invoke_text", lambda *a, **k: None)
    assert score_response("q", "ctx", "resp") is None


def test_score_response_computes_average(monkeypatch):
    monkeypatch.setattr(
        eval_module,
        "invoke_text",
        lambda *a, **k: '{"accuracy": 4, "completeness": 4, "relevance": 4, "feedback": ""}',
    )
    result = score_response("q", "ctx", "resp")
    assert result["average"] == 4.0


def test_feedback_loop_stops_when_score_meets_threshold(monkeypatch):
    monkeypatch.setattr(
        eval_module,
        "invoke_text",
        lambda *a, **k: '{"accuracy": 5, "completeness": 5, "relevance": 5, "feedback": ""}',
    )
    regenerate_calls = []

    def regenerate(query, previous_response, feedback):
        regenerate_calls.append((query, previous_response, feedback))
        return "should not be called"

    result = evaluate_with_feedback_loop("q", "ctx", "initial answer", regenerate, threshold=4.0)

    assert result["response"] == "initial answer"
    assert regenerate_calls == []


def test_feedback_loop_refines_when_below_threshold(monkeypatch):
    scores = iter(
        [
            '{"accuracy": 2, "completeness": 2, "relevance": 2, "feedback": "too vague"}',
            '{"accuracy": 5, "completeness": 5, "relevance": 5, "feedback": ""}',
        ]
    )
    monkeypatch.setattr(eval_module, "invoke_text", lambda *a, **k: next(scores))

    def regenerate(query, previous_response, feedback):
        assert feedback == "too vague"
        return "improved answer"

    result = evaluate_with_feedback_loop("q", "ctx", "initial answer", regenerate, threshold=4.0, max_iterations=2)

    assert result["response"] == "improved answer"
    assert len(result["history"]) == 2
