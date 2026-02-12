import json
import os
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional

from polypharmacy import mechanism


SYSTEM_PROMPT = """You are a biomedical mechanism triage classifier.
Classify each (drug combination, disease) example into exactly one category.
Return only JSON that matches the provided schema.

Category definitions:
- mechanistically_synergistic: drugs are jointly intended to target the indexed disease mechanism or therapeutic effect.
- side_effect_relief: one drug primarily reduces adverse effects caused by another while supporting treatment of the indexed disease.
- common_comorbidity: combination is likely due to co-occurring conditions rather than joint mechanism for the indexed disease.
- other: insufficient information, ambiguous, or outside these categories.

Rules:
- Use `other` when uncertain.
- Mark `needs_human_review=true` for uncertain or borderline cases.
- Keep rationale_short concise and factual.
"""


class OpenAIMechanismClassifier:
    def __init__(
        self,
        model: str,
        prompt_version: str,
        cache_path: Optional[str] = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
        client: Optional[object] = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            self.client = OpenAI()
        else:
            self.client = client
        self.model = model
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        if cache_path:
            self._cache = self._load_cache(cache_path)

    @staticmethod
    def _load_cache(path: str) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(path):
            return {}
        cache: Dict[str, Dict[str, Any]] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = payload.get("cache_key")
                if isinstance(key, str):
                    cache[key] = payload
        return cache

    @staticmethod
    def _append_cache(path: str, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _build_cache_key(self, example_key: str) -> str:
        return f"{example_key}::{self.prompt_version}::{self.model}"

    @staticmethod
    def _row_to_user_prompt(row: Dict[str, Any]) -> str:
        lines = [
            "Classify this example:",
            f"- disease_id: {row['condition_id_norm']}",
            f"- drug_ids: {row['drug_set']}",
            f"- source_label_binary: {row['source_label_binary']}",
            f"- source_file: {row['source_file']}",
        ]
        if row.get("drug_labels"):
            lines.append(f"- drug_labels: {row['drug_labels']}")
        if row.get("condition_label"):
            lines.append(f"- condition_label: {row['condition_label']}")
        lines.append("Output only a JSON object with category, confidence, rationale_short, needs_human_review.")
        return "\n".join(lines)

    def _responses_api_classify(self, user_prompt: str) -> mechanism.MechanismClassification:
        response = self.client.responses.create(
            model=self.model,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": mechanism.MECHANISM_SCHEMA_NAME,
                    "schema": mechanism.MECHANISM_JSON_SCHEMA,
                    "strict": True,
                }
            },
        )
        output_text = response.output_text
        return mechanism.parse_json_text_payload(output_text)

    def classify_row(self, row: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        example_key = str(row["example_key"])
        cache_key = self._build_cache_key(example_key)
        with self._cache_lock:
            if not force and cache_key in self._cache:
                cached = self._cache[cache_key]
                payload = cached["classification"]
                cls = mechanism.parse_classification_payload(payload)
                return {
                    "example_key": example_key,
                    "classification": cls,
                    "cached": True,
                }

        user_prompt = self._row_to_user_prompt(row)
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                cls = self._responses_api_classify(user_prompt)
                result = {
                    "cache_key": cache_key,
                    "example_key": example_key,
                    "prompt_version": self.prompt_version,
                    "model": self.model,
                    "classification": asdict(cls),
                }
                with self._cache_lock:
                    # Another worker may have populated this key while this request ran.
                    if not force and cache_key in self._cache:
                        cached = self._cache[cache_key]
                        payload = cached["classification"]
                        cls = mechanism.parse_classification_payload(payload)
                        return {
                            "example_key": example_key,
                            "classification": cls,
                            "cached": True,
                        }
                    self._cache[cache_key] = result
                    if self.cache_path:
                        self._append_cache(self.cache_path, result)
                return {
                    "example_key": example_key,
                    "classification": cls,
                    "cached": False,
                }
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)

        assert last_error is not None
        raise RuntimeError(f"Classification failed for {example_key}: {last_error}")


def require_openai_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set for mechanism classification")


def summarise_category_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        category = str(row.get("mechanism_category", ""))
        if not category:
            continue
        counts[category] = counts.get(category, 0) + 1
    return counts
