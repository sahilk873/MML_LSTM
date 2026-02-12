import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence


CATEGORY_MECHANISTICALLY_SYNERGISTIC = "mechanistically_synergistic"
CATEGORY_SIDE_EFFECT_RELIEF = "side_effect_relief"
CATEGORY_COMMON_COMORBIDITY = "common_comorbidity"
CATEGORY_OTHER = "other"

ALLOWED_CATEGORIES = {
    CATEGORY_MECHANISTICALLY_SYNERGISTIC,
    CATEGORY_SIDE_EFFECT_RELIEF,
    CATEGORY_COMMON_COMORBIDITY,
    CATEGORY_OTHER,
}

MECHANISM_SCHEMA_NAME = "mechanism_classification"
MECHANISM_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "category": {
            "type": "string",
            "enum": sorted(ALLOWED_CATEGORIES),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rationale_short": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
        },
        "needs_human_review": {
            "type": "boolean",
        },
    },
    "required": ["category", "confidence", "rationale_short", "needs_human_review"],
}


@dataclass
class MechanismClassification:
    category: str
    confidence: float
    rationale_short: str
    needs_human_review: bool


def canonical_example_key(drug_ids: Sequence[str], disease_id: str) -> str:
    sorted_drugs = sorted(str(drug_id) for drug_id in drug_ids)
    return "|".join(sorted_drugs) + "||" + str(disease_id)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_category(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"category must be a string, got {type(value)}")
    if value not in ALLOWED_CATEGORIES:
        raise ValueError(f"Unsupported category: {value}")
    return value


def _validate_confidence(value: Any) -> float:
    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"confidence must be in [0,1], got {confidence}")
    return confidence


def _validate_rationale(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("rationale_short must be a string")
    text = value.strip()
    if not text:
        raise ValueError("rationale_short cannot be empty")
    if len(text) > 600:
        text = text[:600]
    return text


def _validate_needs_review(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError("needs_human_review must be a boolean")


def parse_classification_payload(payload: Dict[str, Any]) -> MechanismClassification:
    category = _validate_category(payload.get("category"))
    confidence = _validate_confidence(payload.get("confidence"))
    rationale_short = _validate_rationale(payload.get("rationale_short"))
    needs_human_review = _validate_needs_review(payload.get("needs_human_review"))
    return MechanismClassification(
        category=category,
        confidence=confidence,
        rationale_short=rationale_short,
        needs_human_review=needs_human_review,
    )


def parse_json_text_payload(text: str) -> MechanismClassification:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object for mechanism classification")
    return parse_classification_payload(parsed)


def categories_from_csv_arg(raw: str) -> List[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("--keep-categories cannot be empty")
    invalid = [value for value in values if value not in ALLOWED_CATEGORIES]
    if invalid:
        raise ValueError(f"Unsupported categories: {invalid}")
    return values


def validate_keep_categories(values: Iterable[str]) -> List[str]:
    result = [str(value).strip() for value in values if str(value).strip()]
    if not result:
        raise ValueError("At least one category is required")
    invalid = [value for value in result if value not in ALLOWED_CATEGORIES]
    if invalid:
        raise ValueError(f"Unsupported categories: {invalid}")
    return result
