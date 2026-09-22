import json
import os
import tempfile
import unittest

import pandas as pd

from polypharmacy import llm_classifier
from polypharmacy import mechanism
from scripts import rebuild_ground_truth_from_mechanisms as rebuild


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1

        class Response:
            output_text = json.dumps(
                {
                    "category": mechanism.CATEGORY_MECHANISTICALLY_SYNERGISTIC,
                    "confidence": 0.9,
                    "rationale_short": "Complementary mechanism for indexed disease.",
                    "needs_human_review": False,
                }
            )

        return Response()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class MechanismPipelineTests(unittest.TestCase):
    def test_canonical_example_key_is_order_invariant(self) -> None:
        key_a = mechanism.canonical_example_key(["CHEBI:2", "CHEBI:1"], "MONDO:10")
        key_b = mechanism.canonical_example_key(["CHEBI:1", "CHEBI:2"], "MONDO:10")
        self.assertEqual(key_a, key_b)

    def test_parse_classification_payload_validation(self) -> None:
        payload = {
            "category": mechanism.CATEGORY_SIDE_EFFECT_RELIEF,
            "confidence": 0.5,
            "rationale_short": "One drug mitigates toxicity.",
            "needs_human_review": True,
        }
        parsed = mechanism.parse_classification_payload(payload)
        self.assertEqual(parsed.category, mechanism.CATEGORY_SIDE_EFFECT_RELIEF)
        with self.assertRaises(ValueError):
            mechanism.parse_classification_payload(
                {
                    "category": "invalid",
                    "confidence": 0.2,
                    "rationale_short": "x",
                    "needs_human_review": False,
                }
            )

    def test_filter_and_rebuild_training_schema(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "drug_set": ["CHEBI:1", "CHEBI:2"],
                    "condition_id_norm": "MONDO:1",
                    "label": 1,
                },
                {
                    "drug_set": ["CHEBI:2", "CHEBI:1"],
                    "condition_id_norm": "MONDO:1",
                    "label": 0,
                },
            ]
        )
        deduped = rebuild._dedupe_with_conflict_resolution(df)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(int(deduped.iloc[0]["label"]), 0)

        converted = rebuild._to_training_schema(deduped)
        self.assertEqual(
            list(converted.columns),
            ["primary_drug_id_norm", "secondary_drug_id_norm", "condition_id_norm"],
        )

    def test_classifier_cache_skips_second_call(self) -> None:
        fake = FakeOpenAIClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.jsonl")
            classifier = llm_classifier.OpenAIMechanismClassifier(
                model="fake-model",
                prompt_version="v1",
                cache_path=cache_path,
                client=fake,
            )
            row = {
                "example_key": "CHEBI:1||MONDO:1",
                "drug_set": ["CHEBI:1"],
                "condition_id_norm": "MONDO:1",
                "source_label_binary": 1,
                "source_file": "indications",
                "drug_labels": ["DrugA"],
                "condition_label": "DiseaseA",
            }

            first = classifier.classify_row(row)
            second = classifier.classify_row(row)

            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(fake.responses.calls, 1)


if __name__ == "__main__":
    unittest.main()
