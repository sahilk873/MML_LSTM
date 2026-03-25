import os
import tempfile
import unittest

import pandas as pd

from scripts import rank_vocab_pairs as rank_vocab


class RankVocabPairsTests(unittest.TestCase):
    def test_parse_mondo_codes_from_markdown(self) -> None:
        text = """# Disease Code Reference
| Disease | Code | Label in File |
|---|---|---|
| A | MONDO:0000001 | a |
| B | MONDO:0000002 (variant) | b |
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "disease.md")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            rows = rank_vocab._parse_mondo_codes_from_markdown(path)
        self.assertEqual(rows, [("A", "MONDO:0000001"), ("B", "MONDO:0000002")])

    def test_load_candidate_drugs_defaults_to_chebi(self) -> None:
        drug_to_idx = {"CHEBI:1": 1, "UNII:2": 2, "CHEBI:3": 3}
        self.assertEqual(
            rank_vocab._load_candidate_drugs(None, drug_to_idx),
            ["CHEBI:1", "CHEBI:3"],
        )

    def test_summarize_combined_rankings(self) -> None:
        combined = pd.DataFrame(
            [
                {"drug_id_1": "CHEBI:1", "drug_id_2": "CHEBI:2"},
                {"drug_id_1": "CHEBI:2", "drug_id_2": "CHEBI:1"},
                {"drug_id_1": "CHEBI:3", "drug_id_2": "CHEBI:4"},
            ]
        )
        summary = rank_vocab._summarize_combined_rankings(combined)
        self.assertEqual(summary["exported_rows"], 3)
        self.assertEqual(summary["unique_pairs"], 2)
        self.assertEqual(summary["pairs_recommended_for_multiple_diseases"], 1)
        self.assertEqual(summary["reused_pair_rows"], 2)


if __name__ == "__main__":
    unittest.main()
