import tempfile
import unittest
from pathlib import Path

import pandas as pd

from polypharmacy import data as data_lib


def _write_training_schema_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


class MixedNegativePipelineTests(unittest.TestCase):
    def test_twosides_and_randomized_negatives_are_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            indications_path = root / "indications.csv"
            contraindications_path = root / "contraindications.csv"
            twosides_path = root / "twosides.csv"

            _write_training_schema_csv(
                indications_path,
                [
                    {
                        "primary_drug_id_norm": "['CHEBI:1']",
                        "secondary_drug_id_norm": "['CHEBI:2']",
                        "condition_id_norm": "MONDO:100",
                    },
                    {
                        "primary_drug_id_norm": "['CHEBI:4']",
                        "secondary_drug_id_norm": "['CHEBI:5']",
                        "condition_id_norm": "MONDO:200",
                    },
                ],
            )
            _write_training_schema_csv(
                contraindications_path,
                [
                    {
                        "primary_drug_id_norm": "['CHEBI:1']",
                        "secondary_drug_id_norm": "['CHEBI:3']",
                        "condition_id_norm": "MONDO:300",
                    }
                ],
            )
            pd.DataFrame(
                [
                    {
                        "drug_1_rxnorn_id_norm": "CHEBI:7",
                        "drug_2_rxnorm_id_norm": "CHEBI:8",
                        "condition_meddra_id_norm": "MONDO:400",
                    }
                ]
            ).to_csv(twosides_path, index=False)

            report: dict = {}
            deduped, _ = data_lib.load_deduped_dataframe(
                str(indications_path),
                str(contraindications_path),
                twosides_contraindications_path=str(twosides_path),
                enable_mixed_negatives=True,
                random_negative_ratio=1.0,
                random_negative_strategy="disease_shuffle",
                seed=13,
                report_out=report,
            )

            self.assertIn("twosides", report["source_counts_before_dedup"])
            self.assertGreater(report["random_negative_generation"]["accepted"], 0)
            self.assertIn(1, report["label_counts_after_dedup"])
            self.assertIn(0, report["label_counts_after_dedup"])

            positives = {
                (tuple(sorted(row.drug_set)), row.condition_id_norm)
                for row in deduped.itertuples(index=False)
                if row.label == 1
            }
            randomized_rows = deduped[
                deduped["source_name"].str.contains("randomized_disease_shuffle")
            ]
            for row in randomized_rows.itertuples(index=False):
                self.assertNotIn((tuple(sorted(row.drug_set)), row.condition_id_norm), positives)

    def test_twosides_invalid_rows_are_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            indications_path = root / "indications.csv"
            contraindications_path = root / "contraindications.csv"
            twosides_path = root / "twosides.csv"

            _write_training_schema_csv(
                indications_path,
                [
                    {
                        "primary_drug_id_norm": "['CHEBI:1']",
                        "secondary_drug_id_norm": "['CHEBI:2']",
                        "condition_id_norm": "MONDO:100",
                    }
                ],
            )
            _write_training_schema_csv(
                contraindications_path,
                [
                    {
                        "primary_drug_id_norm": "['CHEBI:3']",
                        "secondary_drug_id_norm": "['CHEBI:4']",
                        "condition_id_norm": "MONDO:300",
                    }
                ],
            )
            pd.DataFrame(
                [
                    {
                        "drug_1_rxnorn_id_norm": "CHEBI:7",
                        "drug_2_rxnorm_id_norm": "CHEBI:8",
                        "condition_meddra_id_norm": "MONDO:400",
                    },
                    {
                        "drug_1_rxnorn_id_norm": "CHEBI:9",
                        "drug_2_rxnorm_id_norm": "CHEBI:10",
                        "condition_meddra_id_norm": "['Error']",
                    },
                ]
            ).to_csv(twosides_path, index=False)

            report: dict = {}
            data_lib.load_deduped_dataframe(
                str(indications_path),
                str(contraindications_path),
                twosides_contraindications_path=str(twosides_path),
                enable_mixed_negatives=True,
                random_negative_ratio=0.0,
                seed=7,
                report_out=report,
            )
            self.assertEqual(report["source_counts_before_dedup"].get("twosides"), 1)


if __name__ == "__main__":
    unittest.main()
