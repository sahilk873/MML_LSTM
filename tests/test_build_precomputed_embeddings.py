import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import build_precomputed_embeddings as bpe


class BuildPrecomputedEmbeddingsTests(unittest.TestCase):
    def test_parse_equivalent_identifiers_handles_mixed_formats(self) -> None:
        arr = bpe._parse_equivalent_identifiers("['CHEBI:1', 'RXCUI:2', ['Error']]")
        self.assertEqual(arr, ["CHEBI:1", "RXCUI:2"])

        arr = bpe._parse_equivalent_identifiers(["UNII:AAA", "None", ""])
        self.assertEqual(arr, ["UNII:AAA"])

        arr = bpe._parse_equivalent_identifiers(None)
        self.assertEqual(arr, [])

    def test_write_equivalent_id_index_includes_id_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            in_dir = root / "nodes"
            in_dir.mkdir()
            out_path = root / "equivalent_id_to_node_id.parquet"

            df = pd.DataFrame(
                [
                    {
                        "id": "CHEBI:1",
                        "equivalent_identifiers": ["RXCUI:100", "UNII:AAA", "RXCUI:100"],
                    },
                    {
                        "id": "UNII:2",
                        "equivalent_identifiers": ["DRUGBANK:DB2", "Error:bad"],
                    },
                    {
                        "id": "MONDO:3",
                        "equivalent_identifiers": ["UMLS:C3"],
                    },
                ]
            )
            (in_dir / "part-00000.parquet").write_bytes(b"")
            df.to_parquet(in_dir / "part-00000.parquet", index=False)

            rows = bpe.write_equivalent_id_index(
                paths=[str(in_dir / "part-00000.parquet")],
                output_path=str(out_path),
                required_ids=None,
                id_prefixes={"CHEBI", "UNII"},
            )
            self.assertGreater(rows, 0)

            out = pd.read_parquet(out_path).sort_values(["node_id", "alias_id"]).reset_index(
                drop=True
            )
            # MONDO row filtered out by id_prefixes, ids included as aliases, duplicates removed.
            expected_aliases = {
                ("CHEBI:1", "CHEBI:1"),
                ("RXCUI:100", "CHEBI:1"),
                ("UNII:AAA", "CHEBI:1"),
                ("UNII:2", "UNII:2"),
                ("DRUGBANK:DB2", "UNII:2"),
            }
            actual_aliases = {(r.alias_id, r.node_id) for r in out.itertuples(index=False)}
            self.assertEqual(actual_aliases, expected_aliases)
            self.assertTrue((out["match_source"].isin(["id", "equivalent_identifiers"])).all())


if __name__ == "__main__":
    unittest.main()
