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

    def test_map_medic_drugs_prefers_priority_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            medic_csv = root / "medic.csv"
            alias_parquet = root / "alias.parquet"

            pd.DataFrame(
                [
                    {
                        "drug_name": "Drug A",
                        "curie": "UNII:X",
                        "alternate_ids": "['CHEBI:100', 'DRUGBANK:DB1']",
                    },
                    {
                        "drug_name": "Drug B",
                        "curie": "UNII:Y",
                        "alternate_ids": "['UNII:Y']",
                    },
                    {
                        "drug_name": "Drug C",
                        "curie": "NONE:1",
                        "alternate_ids": "[]",
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
                ]
            ).to_parquet(alias_parquet, index=False)

            allowed_node_ids = {"CHEBI:100", "UNII:X", "DRUGBANK:DB1", "UNII:Y"}
            matched, unmatched = rank_rf._map_medic_drugs(
                str(medic_csv), str(alias_parquet), allowed_node_ids
            )

            self.assertEqual(len(matched), 2)
            # Drug A chooses CHEBI over UNII/DRUGBANK by priority.
            chosen_a = matched.loc[matched["drug_name"] == "Drug A", "selected_node_id"].iloc[0]
            self.assertEqual(chosen_a, "CHEBI:100")
            self.assertEqual(len(unmatched), 1)


if __name__ == "__main__":
    unittest.main()
