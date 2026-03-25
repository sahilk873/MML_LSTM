import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import rank_medic_pairs_rf as rank_rf


class RankMedicPairsRFTests(unittest.TestCase):
    def test_parse_mondo_codes_from_markdown(self) -> None:
        text = """# Disease Code Reference
| Disease | Code | Label in File |
|---|---|---|
| A | MONDO:0000001 | a |
| B | MONDO:0000002 (variant) | b |
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            md = Path(tmpdir) / "d.md"
            md.write_text(text, encoding="utf-8")
            rows = rank_rf._parse_mondo_codes_from_markdown(str(md))
        self.assertEqual(rows, [("A", "MONDO:0000001"), ("B", "MONDO:0000002")])

    def test_parse_optional_id_list(self) -> None:
        self.assertEqual(
            rank_rf._parse_optional_id_list("['CHEBI:1', 'UNII:2', ['Error']]"),
            ["CHEBI:1", "UNII:2"],
        )
        self.assertEqual(rank_rf._parse_optional_id_list(None), [])

    def test_combination_therapy_parser(self) -> None:
        self.assertTrue(rank_rf._is_combination_therapy("TRUE"))
        self.assertTrue(rank_rf._is_combination_therapy("False; True"))
        self.assertFalse(rank_rf._is_combination_therapy("FALSE"))

    def test_zinc_oxide_exclusion(self) -> None:
        self.assertTrue(rank_rf._should_exclude_medic_row("CHEBI:36560", "[]", "ZINC OXIDE"))
        self.assertTrue(
            rank_rf._should_exclude_medic_row("UNII:X", "['CHEBI:36560']", "Something Else")
        )
        self.assertFalse(rank_rf._should_exclude_medic_row("UNII:X", "['CHEBI:1']", "Drug A"))

    def test_map_medic_drugs_prefers_priority_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            medic_csv = root / "medic.csv"
            alias_parquet = root / "alias.parquet"

            pd.DataFrame(
                [
                    {
                        "drug_name": "Drug A",
                        "curie_label": "Drug A Label",
                        "curie": "UNII:X",
                        "alternate_ids": "['CHEBI:100', 'DRUGBANK:DB1']",
                        "combination_therapy": "FALSE",
                    },
                    {
                        "drug_name": "Drug B",
                        "curie_label": "",
                        "curie": "UNII:Y",
                        "alternate_ids": "['UNII:Y']",
                        "combination_therapy": "FALSE",
                    },
                    {
                        "drug_name": "Drug C",
                        "curie_label": "Drug C Label",
                        "curie": "NONE:1",
                        "alternate_ids": "[]",
                        "combination_therapy": "FALSE",
                    },
                    {
                        "drug_name": "Drug Combo",
                        "curie_label": "Drug Combo Label",
                        "curie": "CHEBI:999",
                        "alternate_ids": "['CHEBI:999']",
                        "combination_therapy": "TRUE",
                    },
                    {
                        "drug_name": "ZINC OXIDE",
                        "curie_label": "Zinc Oxide Label",
                        "curie": "CHEBI:36560",
                        "alternate_ids": "['CHEBI:36560']",
                        "combination_therapy": "FALSE",
                    },
                ]
            ).to_csv(medic_csv, index=False)

            pd.DataFrame(
                [
                    {"alias_id": "UNII:X", "node_id": "UNII:X", "match_source": "id"},
                    {
                        "alias_id": "CHEBI:100",
                        "node_id": "CHEBI:100",
                        "match_source": "id",
                    },
                    {
                        "alias_id": "DRUGBANK:DB1",
                        "node_id": "DRUGBANK:DB1",
                        "match_source": "id",
                    },
                    {"alias_id": "UNII:Y", "node_id": "UNII:Y", "match_source": "id"},
                    {
                        "alias_id": "CHEBI:999",
                        "node_id": "CHEBI:999",
                        "match_source": "id",
                    },
                    {
                        "alias_id": "CHEBI:36560",
                        "node_id": "CHEBI:36560",
                        "match_source": "id",
                    },
                ]
            ).to_parquet(alias_parquet, index=False)

            allowed_node_ids = {
                "CHEBI:100",
                "UNII:X",
                "DRUGBANK:DB1",
                "UNII:Y",
                "CHEBI:999",
                "CHEBI:36560",
            }
            matched, unmatched = rank_rf._map_medic_drugs(
                str(medic_csv), str(alias_parquet), allowed_node_ids
            )

            self.assertEqual(len(matched), 2)
            # Drug A chooses CHEBI over UNII/DRUGBANK by priority.
            chosen_a = matched.loc[matched["drug_name"] == "Drug A", "selected_node_id"].iloc[0]
            self.assertEqual(chosen_a, "CHEBI:100")
            self.assertEqual(len(unmatched), 1)

            drug_name_map = rank_rf._build_drug_name_map(matched)
            self.assertEqual(drug_name_map["CHEBI:100"], "Drug A Label")
            self.assertEqual(drug_name_map["UNII:Y"], "Drug B")

    def test_summarize_combined_rankings(self) -> None:
        combined = pd.DataFrame(
            [
                {"drug_id_1": "CHEBI:1", "drug_id_2": "CHEBI:2", "disease_code_used": "MONDO:1"},
                {"drug_id_1": "CHEBI:2", "drug_id_2": "CHEBI:1", "disease_code_used": "MONDO:2"},
                {"drug_id_1": "CHEBI:3", "drug_id_2": "CHEBI:4", "disease_code_used": "MONDO:3"},
            ]
        )
        summary = rank_rf._summarize_combined_rankings(combined)
        self.assertEqual(summary["exported_rows"], 3)
        self.assertEqual(summary["unique_pairs"], 2)
        self.assertEqual(summary["pairs_recommended_for_multiple_diseases"], 1)
        self.assertEqual(summary["reused_pair_rows"], 2)


if __name__ == "__main__":
    unittest.main()
