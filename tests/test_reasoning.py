"""
Unit tests for AgriVision deterministic reasoning engine and confidence guardrails.
"""

import pytest
from src.reasoning import ConfidenceState, Intent, ReasoningEngine, classify_intent


@pytest.fixture
def engine():
    return ReasoningEngine(confidence_threshold=0.50)


# ============================================================
# INTENT CLASSIFICATION TESTS
# ============================================================

def test_intent_classification_count():
    assert classify_intent("How many crops are there?") == Intent.COUNT
    assert classify_intent("How many weeds are present?") == Intent.COUNT
    assert classify_intent("Count the weeds.") == Intent.COUNT
    assert classify_intent("Total number of objects?") == Intent.COUNT


def test_intent_classification_comparison():
    assert classify_intent("Are there more crops or weeds?") == Intent.COMPARISON
    assert classify_intent("Which is more common, crops or weeds?") == Intent.COMPARISON
    assert classify_intent("Are weeds more numerous than crops?") == Intent.COMPARISON


def test_intent_classification_presence():
    assert classify_intent("Are there any weeds?") == Intent.PRESENCE
    assert classify_intent("Is a crop present?") == Intent.PRESENCE
    assert classify_intent("Are crops visible in the image?") == Intent.PRESENCE


def test_intent_classification_summary():
    assert classify_intent("What objects are present?") == Intent.SUMMARY
    assert classify_intent("What is visible?") == Intent.SUMMARY
    assert classify_intent("Summarize the detected objects.") == Intent.SUMMARY


def test_intent_classification_ratio():
    assert classify_intent("What is the crop to weed ratio?") == Intent.RATIO
    assert classify_intent("What is the ratio of crops to weeds?") == Intent.RATIO


def test_intent_classification_unsupported():
    assert classify_intent("What fertilizer should I use?") == Intent.UNSUPPORTED
    assert classify_intent("Are the crops healthy?") == Intent.UNSUPPORTED
    assert classify_intent("What disease does the crop have?") == Intent.UNSUPPORTED
    assert classify_intent("What will the crop yield be?") == Intent.UNSUPPORTED
    assert classify_intent("What species of weed is this?") == Intent.UNSUPPORTED
    assert classify_intent("What should the farmer do?") == Intent.UNSUPPORTED
    assert classify_intent("What is the weather?") == Intent.UNSUPPORTED


# ============================================================
# COUNT REASONING TESTS
# ============================================================

def test_reasoning_count_crops_confident(engine):
    detections = {"counts": {"crop": 5, "weed": 2}, "total_count": 7}
    res = engine.reason("How many crops are there?", detections)
    assert res["intent"] == "count"
    assert "5 crops" in res["answer"]
    assert res["confidence"] == "high"
    assert res["evidence"]["crop_count"] == 5


def test_reasoning_count_weeds_confident(engine):
    detections = {"counts": {"crop": 1, "weed": 3}, "total_count": 4}
    res = engine.reason("Count the weeds.", detections)
    assert res["intent"] == "count"
    assert "3 weeds" in res["answer"]
    assert res["confidence"] == "high"
    assert res["evidence"]["weed_count"] == 3


def test_reasoning_count_zero_weed_guardrail(engine):
    """
    CRITICAL REQUIREMENT:
    Zero confident weed detections must NOT state 'There are 0 weeds.'
    It must return 'Insufficient information'.
    """
    detections = {"counts": {"crop": 4, "weed": 0}, "total_count": 4}
    res = engine.reason("How many weeds are present?", detections)
    assert res["intent"] == "count"
    assert "Insufficient information" in res["answer"]
    assert "0 weeds" not in res["answer"]
    assert res["confidence"] == "insufficient"


# ============================================================
# COMPARISON REASONING TESTS
# ============================================================

def test_reasoning_comparison_more_crops(engine):
    detections = {"counts": {"crop": 10, "weed": 4}, "total_count": 14}
    res = engine.reason("Are there more crops or weeds?", detections)
    assert res["intent"] == "comparison"
    assert "There are more crops than weeds: 10 crops compared with 4 weeds." in res["answer"]
    assert res["confidence"] == "high"


def test_reasoning_comparison_more_weeds(engine):
    detections = {"counts": {"crop": 2, "weed": 6}, "total_count": 8}
    res = engine.reason("Are there more crops or weeds?", detections)
    assert res["intent"] == "comparison"
    assert "There are more weeds than crops" in res["answer"]
    assert res["confidence"] == "high"


def test_reasoning_comparison_equal(engine):
    detections = {"counts": {"crop": 3, "weed": 3}, "total_count": 6}
    res = engine.reason("Are there more crops or weeds?", detections)
    assert res["intent"] == "comparison"
    assert "equal numbers of crops and weeds" in res["answer"]
    assert res["confidence"] == "high"


def test_reasoning_comparison_one_class_zero_guardrail(engine):
    """
    Guardrail: If one class has 0 detections, a definitive comparison cannot be made.
    """
    detections = {"counts": {"crop": 5, "weed": 0}, "total_count": 5}
    res = engine.reason("Are there more crops or weeds?", detections)
    assert res["intent"] == "comparison"
    assert "Insufficient information" in res["answer"]
    assert res["confidence"] == "insufficient"


# ============================================================
# PRESENCE REASONING TESTS
# ============================================================

def test_reasoning_presence_positive(engine):
    detections = {"counts": {"crop": 0, "weed": 2}, "total_count": 2}
    res = engine.reason("Are there any weeds?", detections)
    assert res["intent"] == "presence"
    assert "Yes, 2 weeds are present." in res["answer"]
    assert res["confidence"] == "high"


def test_reasoning_presence_insufficient_guardrail(engine):
    detections = {"counts": {"crop": 0, "weed": 0}, "total_count": 0}
    res = engine.reason("Are crops visible?", detections)
    assert res["intent"] == "presence"
    assert "Insufficient information" in res["answer"]
    assert res["confidence"] == "insufficient"


# ============================================================
# SUMMARY REASONING TESTS
# ============================================================

def test_reasoning_summary_both_classes(engine):
    detections = {"counts": {"crop": 4, "weed": 1}, "total_count": 5}
    res = engine.reason("What objects are present?", detections)
    assert res["intent"] == "summary"
    assert "Detected 4 crops, 1 weed (total: 5 objects)." in res["answer"]
    assert res["confidence"] == "high"


def test_reasoning_summary_empty(engine):
    detections = {"counts": {"crop": 0, "weed": 0}, "total_count": 0}
    res = engine.reason("Summarize the detected objects.", detections)
    assert res["intent"] == "summary"
    assert "Insufficient information" in res["answer"]
    assert res["confidence"] == "insufficient"


# ============================================================
# RATIO REASONING TESTS
# ============================================================

def test_reasoning_ratio_standard(engine):
    detections = {"counts": {"crop": 10, "weed": 5}, "total_count": 15}
    res = engine.reason("What is the crop to weed ratio?", detections)
    assert res["intent"] == "ratio"
    assert "2:1" in res["answer"]
    assert res["confidence"] == "high"


def test_reasoning_ratio_zero_weeds(engine):
    detections = {"counts": {"crop": 10, "weed": 0}, "total_count": 10}
    res = engine.reason("What is the crop to weed ratio?", detections)
    assert res["intent"] == "ratio"
    assert "cannot be calculated" in res["answer"]
    assert res["confidence"] == "insufficient"


# ============================================================
# UNSUPPORTED QUESTIONS GUARDRAILS
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        "What fertilizer should I use?",
        "Are the crops healthy?",
        "What disease does the crop have?",
        "What will the crop yield be?",
        "What species of weed is this?",
        "What should the farmer do?",
        "What is the weather today?",
    ],
)
def test_unsupported_questions_return_insufficient_information(engine, query):
    res = engine.reason(query, detection_result=None)
    assert res["intent"] == "unsupported"
    assert "Insufficient information" in res["answer"]
    assert res["confidence"] == "insufficient"
