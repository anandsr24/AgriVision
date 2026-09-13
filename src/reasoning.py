"""
Deterministic Reasoning Engine for AgriVision.

Classifies natural-language queries into structured intents (COUNT, COMPARISON, PRESENCE, SUMMARY, RATIO, UNSUPPORTED)
and performs confidence-guarded reasoning exclusively over structured detector outputs without external LLMs or agent frameworks.
"""

import logging
import math
import re
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("reasoning")


class Intent(str, Enum):
    COUNT = "count"
    COMPARISON = "comparison"
    PRESENCE = "presence"
    SUMMARY = "summary"
    RATIO = "ratio"
    UNSUPPORTED = "unsupported"


class ConfidenceState(str, Enum):
    HIGH = "high"
    LOW = "low"
    INSUFFICIENT = "insufficient"


# Keywords/patterns for unsupported agronomic / out-of-scope questions
UNSUPPORTED_PATTERNS = [
    r"\b(disease|sick|health|healthy|unhealthy|infection|fungus|virus|pest|bug|insect|blight)\b",
    r"\b(fertilizer|nutrient|nitrogen|phosphorus|potassium|npk|manure|compost)\b",
    r"\b(yield|harvest|tons|bushels|output|productivity)\b",
    r"\b(species|variety|cultivar|type of (weed|crop)|scientific name)\b",
    r"\b(weather|rain|temperature|humidity|soil|moisture|irrigation|water)\b",
    r"\b(advice|recommend|recommendation|what should (the )?(farmer|i) do|how to treat|spray|herbicide|pesticide)\b",
    r"\b(survive|alive|die|growth rate|stage)\b",
]

# Supported intent patterns
COUNT_PATTERNS = [
    r"\bhow many\b",
    r"\bcount\b",
    r"\bnumber of\b",
    r"\btotal\b",
]

COMPARISON_PATTERNS = [
    r"\bmore (crops?|weeds?) than (crops?|weeds?)\b",
    r"\b(crops?|weeds?) more (numerous|common|prevalent|frequent) than (crops?|weeds?)\b",
    r"\b(more|fewer|less|greater) (crops?|weeds?)\b",
    r"\bare there more\b",
    r"\bwhich is more (common|numerous|prevalent|frequent)\b",
    r"\bwhich (one )?(do we have|are there) more\b",
    r"\bcompare\b",
]

RATIO_PATTERNS = [
    r"\bratio\b",
    r"\bproportion\b",
    r"\bpercentage of\b",
]

PRESENCE_PATTERNS = [
    r"\bare (there )?(any )?(crops?|weeds?|plants?)\b",
    r"\bis (there )?(a |any )?(crop|weed|plant)\b",
    r"\bdo you see (any )?(crops?|weeds?|plants?)\b",
    r"\b(crops?|weeds?|plants?) (present|visible)\b",
    r"\bis (a |any )?(crop|weed|plant) present\b",
    r"\bare (crops?|weeds?|plants?) present\b",
]

SUMMARY_PATTERNS = [
    r"\bwhat (objects?|things?|plants?|items?) (are|is) (present|visible|in the image|detected)\b",
    r"\bwhat (is|are) visible\b",
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\bwhat do you see\b",
    r"\bdescribe (the )?(detected )?objects\b",
]


def classify_intent(question: str) -> Intent:
    """
    Deterministically classifies a user question into one of the predefined Intents.
    """
    q = question.strip().lower()
    if not q:
        return Intent.UNSUPPORTED

    # Check unsupported patterns first
    for pat in UNSUPPORTED_PATTERNS:
        if re.search(pat, q):
            return Intent.UNSUPPORTED

    # Check summary before presence
    for pat in SUMMARY_PATTERNS:
        if re.search(pat, q):
            return Intent.SUMMARY

    # Check ratio before count
    for pat in RATIO_PATTERNS:
        if re.search(pat, q):
            return Intent.RATIO

    # Check comparison
    for pat in COMPARISON_PATTERNS:
        if re.search(pat, q):
            return Intent.COMPARISON

    # Check count
    for pat in COUNT_PATTERNS:
        if re.search(pat, q):
            return Intent.COUNT

    # Check presence
    for pat in PRESENCE_PATTERNS:
        if re.search(pat, q):
            return Intent.PRESENCE

    # Default fallback for unrecognized agricultural queries
    return Intent.UNSUPPORTED


def extract_target_class(question: str) -> Optional[str]:
    """
    Extracts whether the question specifically targets 'crop', 'weed', or both/all.
    """
    q = question.lower()
    has_crop = bool(re.search(r"\bcrops?\b", q))
    has_weed = bool(re.search(r"\bweeds?\b", q))

    if has_crop and not has_weed:
        return "crop"
    if has_weed and not has_crop:
        return "weed"
    return "all"


class ReasoningEngine:
    """
    Deterministic reasoning layer over structured detector outputs with explicit confidence guardrails.
    """

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = float(confidence_threshold)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
    def reason(self, question: str, detection_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes confidence-guarded reasoning over a natural language question and detector results.
        """
        intent = classify_intent(question)
        logger.info(f"Classified query '{question}' as intent '{intent.value}'")

        # Handle unsupported questions (no detector required)
        if intent == Intent.UNSUPPORTED:
            return {
                "question": question,
                "intent": Intent.UNSUPPORTED.value,
                "answer": (
                    "Insufficient information. The available detector only identifies crops and weeds "
                    "and does not provide enough information to answer this question."
                ),
                "confidence": ConfidenceState.INSUFFICIENT.value,
                "evidence": {
                    "reason": "out_of_scope_intent",
                    "supported_intents": ["count", "comparison", "presence", "summary", "ratio"],
                },
            }

        # If question is supported, detector results are required
        if detection_result is None:
            return {
                "question": question,
                "intent": intent.value,
                "answer": "Insufficient information. Detector results are required to answer this question.",
                "confidence": ConfidenceState.INSUFFICIENT.value,
                "evidence": {"error": "missing_detector_output"},
            }

        # Derive counts directly from confidence-filtered detections when available, or validate against counts dict
        detections = detection_result.get("detections", None)
        if detections is not None:
            valid_detections = [
                d for d in detections
                if float(d.get("confidence", 1.0)) >= self.confidence_threshold
            ]
            crop_count = sum(1 for d in valid_detections if d.get("class") == "crop")
            weed_count = sum(1 for d in valid_detections if d.get("class") == "weed")
            total_count = crop_count + weed_count
        else:
            counts = detection_result.get("counts", {})
            crop_count = int(counts.get("crop", 0))
            weed_count = int(counts.get("weed", 0))
            total_count = crop_count + weed_count

        # Route to intent handler
        if intent == Intent.COUNT:
            return self._handle_count(question, crop_count, weed_count, total_count)
        elif intent == Intent.COMPARISON:
            return self._handle_comparison(question, crop_count, weed_count)
        elif intent == Intent.PRESENCE:
            return self._handle_presence(question, crop_count, weed_count)
        elif intent == Intent.SUMMARY:
            return self._handle_summary(question, crop_count, weed_count, total_count)
        elif intent == Intent.RATIO:
            return self._handle_ratio(question, crop_count, weed_count)

        # Fallback guard
        return {
            "question": question,
            "intent": Intent.UNSUPPORTED.value,
            "answer": "Insufficient information. The query could not be processed.",
            "confidence": ConfidenceState.INSUFFICIENT.value,
            "evidence": {},
        }

    def _handle_count(self, question: str, crop_count: int, weed_count: int, total_count: int) -> Dict[str, Any]:
        target = extract_target_class(question)

        if target == "crop":
            if crop_count > 0:
                answer = f"I detected {crop_count} {'crop' if crop_count == 1 else 'crops'}."
                return {
                    "question": question,
                    "intent": Intent.COUNT.value,
                    "answer": answer,
                    "confidence": ConfidenceState.HIGH.value,
                    "evidence": {"crop_count": crop_count, "confidence_threshold": self.confidence_threshold},
                }
            else:
                return {
                    "question": question,
                    "intent": Intent.COUNT.value,
                    "answer": "Insufficient information. The detector did not produce sufficiently confident crop detections.",
                    "confidence": ConfidenceState.INSUFFICIENT.value,
                    "evidence": {"crop_count": 0, "confidence_threshold": self.confidence_threshold},
                }

        elif target == "weed":
            if weed_count > 0:
                answer = f"I detected {weed_count} {'weed' if weed_count == 1 else 'weeds'}."
                return {
                    "question": question,
                    "intent": Intent.COUNT.value,
                    "answer": answer,
                    "confidence": ConfidenceState.HIGH.value,
                    "evidence": {"weed_count": weed_count, "confidence_threshold": self.confidence_threshold},
                }
            else:
                return {
                    "question": question,
                    "intent": Intent.COUNT.value,
                    "answer": "Insufficient information. The detector did not produce sufficiently confident weed detections.",
                    "confidence": ConfidenceState.INSUFFICIENT.value,
                    "evidence": {"weed_count": 0, "confidence_threshold": self.confidence_threshold},
                }

        else:  # Total / all objects
            if total_count > 0:
                return {
                    "question": question,
                    "intent": Intent.COUNT.value,
                    "answer": f"I detected {total_count} total objects ({crop_count} crops and {weed_count} weeds).",
                    "confidence": ConfidenceState.HIGH.value,
                    "evidence": {
                        "total_count": total_count,
                        "crop_count": crop_count,
                        "weed_count": weed_count,
                        "confidence_threshold": self.confidence_threshold,
                    },
                }
            else:
                return {
                    "question": question,
                    "intent": Intent.COUNT.value,
                    "answer": "Insufficient information. The detector did not produce sufficiently confident detections for any objects.",
                    "confidence": ConfidenceState.INSUFFICIENT.value,
                    "evidence": {"total_count": 0, "confidence_threshold": self.confidence_threshold},
                }

    def _handle_comparison(self, question: str, crop_count: int, weed_count: int) -> Dict[str, Any]:
        # Guardrail: If both are 0, or either class is unconfirmed, cannot make a guaranteed comparison
        if crop_count == 0 and weed_count == 0:
            return {
                "question": question,
                "intent": Intent.COMPARISON.value,
                "answer": "Insufficient information. The detector did not produce sufficiently confident detections for either crops or weeds to make a comparison.",
                "confidence": ConfidenceState.INSUFFICIENT.value,
                "evidence": {"crop_count": 0, "weed_count": 0, "confidence_threshold": self.confidence_threshold},
            }

        if crop_count > 0 and weed_count == 0:
            return {
                "question": question,
                "intent": Intent.COMPARISON.value,
                "answer": (
                    f"Insufficient information. There are {crop_count} confident crop detections but no sufficiently "
                    "confident weed detections, so a definitive comparison cannot be guaranteed."
                ),
                "confidence": ConfidenceState.INSUFFICIENT.value,
                "evidence": {"crop_count": crop_count, "weed_count": 0, "confidence_threshold": self.confidence_threshold},
            }

        if weed_count > 0 and crop_count == 0:
            return {
                "question": question,
                "intent": Intent.COMPARISON.value,
                "answer": (
                    f"Insufficient information. There are {weed_count} confident weed detections but no sufficiently "
                    "confident crop detections, so a definitive comparison cannot be guaranteed."
                ),
                "confidence": ConfidenceState.INSUFFICIENT.value,
                "evidence": {"crop_count": 0, "weed_count": weed_count, "confidence_threshold": self.confidence_threshold},
            }

        if crop_count > weed_count:
            answer = f"There are more crops than weeds: {crop_count} crops compared with {weed_count} weeds."
        elif weed_count > crop_count:
            answer = f"There are more weeds than crops: {weed_count} weeds compared with {crop_count} crops."
        else:
            answer = f"There are equal numbers of crops and weeds: {crop_count} crops and {weed_count} weeds."

        return {
            "question": question,
            "intent": Intent.COMPARISON.value,
            "answer": answer,
            "confidence": ConfidenceState.HIGH.value,
            "evidence": {"crop_count": crop_count, "weed_count": weed_count},
        }

    def _handle_presence(self, question: str, crop_count: int, weed_count: int) -> Dict[str, Any]:
        target = extract_target_class(question)

        if target == "crop":
            if crop_count > 0:
                return {
                    "question": question,
                    "intent": Intent.PRESENCE.value,
                    "answer": f"Yes, {crop_count} {'crop is' if crop_count == 1 else 'crops are'} present.",
                    "confidence": ConfidenceState.HIGH.value,
                    "evidence": {"crop_count": crop_count},
                }
            else:
                return {
                    "question": question,
                    "intent": Intent.PRESENCE.value,
                    "answer": "Insufficient information. The detector did not produce sufficiently confident crop detections to confirm presence.",
                    "confidence": ConfidenceState.INSUFFICIENT.value,
                    "evidence": {"crop_count": 0, "confidence_threshold": self.confidence_threshold},
                }

        elif target == "weed":
            if weed_count > 0:
                return {
                    "question": question,
                    "intent": Intent.PRESENCE.value,
                    "answer": f"Yes, {weed_count} {'weed is' if weed_count == 1 else 'weeds are'} present.",
                    "confidence": ConfidenceState.HIGH.value,
                    "evidence": {"weed_count": weed_count},
                }
            else:
                return {
                    "question": question,
                    "intent": Intent.PRESENCE.value,
                    "answer": "Insufficient information. The detector did not produce sufficiently confident weed detections to confirm presence.",
                    "confidence": ConfidenceState.INSUFFICIENT.value,
                    "evidence": {"weed_count": 0, "confidence_threshold": self.confidence_threshold},
                }

        else:
            total = crop_count + weed_count
            if total > 0:
                return {
                    "question": question,
                    "intent": Intent.PRESENCE.value,
                    "answer": f"Yes, objects are present ({crop_count} crops, {weed_count} weeds).",
                    "confidence": ConfidenceState.HIGH.value,
                    "evidence": {"total_count": total, "crop_count": crop_count, "weed_count": weed_count},
                }
            else:
                return {
                    "question": question,
                    "intent": Intent.PRESENCE.value,
                    "answer": "Insufficient information. No objects were detected with sufficient confidence.",
                    "confidence": ConfidenceState.INSUFFICIENT.value,
                    "evidence": {"total_count": 0, "confidence_threshold": self.confidence_threshold},
                }

    def _handle_summary(self, question: str, crop_count: int, weed_count: int, total_count: int) -> Dict[str, Any]:
        if total_count == 0:
            return {
                "question": question,
                "intent": Intent.SUMMARY.value,
                "answer": "Insufficient information. The detector did not produce sufficiently confident detections to summarize.",
                "confidence": ConfidenceState.INSUFFICIENT.value,
                "evidence": {"crop_count": 0, "weed_count": 0, "total_count": 0},
            }

        parts = []
        if crop_count > 0:
            parts.append(f"{crop_count} {'crop' if crop_count == 1 else 'crops'}")
        if weed_count > 0:
            parts.append(f"{weed_count} {'weed' if weed_count == 1 else 'weeds'}")

        answer = f"Detected {', '.join(parts)} (total: {total_count} objects)."
        return {
            "question": question,
            "intent": Intent.SUMMARY.value,
            "answer": answer,
            "confidence": ConfidenceState.HIGH.value,
            "evidence": {"crop_count": crop_count, "weed_count": weed_count, "total_count": total_count},
        }

    def _handle_ratio(self, question: str, crop_count: int, weed_count: int) -> Dict[str, Any]:
        if crop_count == 0 and weed_count == 0:
            return {
                "question": question,
                "intent": Intent.RATIO.value,
                "answer": "Insufficient information. No confident detections exist to calculate a crop-to-weed ratio.",
                "confidence": ConfidenceState.INSUFFICIENT.value,
                "evidence": {"crop_count": 0, "weed_count": 0},
            }

        if crop_count > 0 and weed_count == 0:
            return {
                "question": question,
                "intent": Intent.RATIO.value,
                "answer": (
                    f"There are {crop_count} sufficiently confident crop detections and no sufficiently confident weed "
                    "detections, so a finite crop-to-weed ratio cannot be calculated."
                ),
                "confidence": ConfidenceState.INSUFFICIENT.value,
                "evidence": {"crop_count": crop_count, "weed_count": 0},
            }

        if weed_count > 0 and crop_count == 0:
            return {
                "question": question,
                "intent": Intent.RATIO.value,
                "answer": (
                    f"There are {weed_count} sufficiently confident weed detections and no sufficiently confident crop "
                    "detections, so a crop-to-weed ratio is 0."
                ),
                "confidence": ConfidenceState.HIGH.value,
                "evidence": {"crop_count": 0, "weed_count": weed_count, "ratio": 0.0},
            }

        gcd = math.gcd(crop_count, weed_count)
        simplified_crop = crop_count // gcd
        simplified_weed = weed_count // gcd
        float_ratio = round(crop_count / weed_count, 2)

        return {
            "question": question,
            "intent": Intent.RATIO.value,
            "answer": f"The crop-to-weed ratio is {simplified_crop}:{simplified_weed} ({float_ratio}:1).",
            "confidence": ConfidenceState.HIGH.value,
            "evidence": {
                "crop_count": crop_count,
                "weed_count": weed_count,
                "ratio_simplified": f"{simplified_crop}:{simplified_weed}",
                "ratio_value": float_ratio,
            },
        }
