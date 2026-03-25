import unittest

import numpy as np
import pandas as pd

from polypharmacy import triplet


class TripletMetricsTests(unittest.TestCase):
    def test_classification_metrics_basic(self) -> None:
        labels = np.array([1, 0, 1, 0], dtype=np.int64)
        probs = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        metrics = triplet.compute_classification_metrics(labels, probs, threshold=0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=6)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=6)
        self.assertAlmostEqual(metrics["f1"], 0.5, places=6)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5, places=6)
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)

    def test_pair_aggregation(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "drug_id_norm": "CHEBI:1",
                    "disease_id_norm": "MONDO:1",
                    "target_id_norm": "NCBIGene:1",
                    "label": 1,
                    "score": 0.9,
                },
                {
                    "drug_id_norm": "CHEBI:1",
                    "disease_id_norm": "MONDO:1",
                    "target_id_norm": "NCBIGene:2",
                    "label": 0,
                    "score": 0.7,
                },
                {
                    "drug_id_norm": "CHEBI:2",
                    "disease_id_norm": "MONDO:2",
                    "target_id_norm": "NCBIGene:3",
                    "label": 0,
                    "score": 0.3,
                },
            ]
        )
        pair_df = triplet.aggregate_pair_predictions(df)
        self.assertEqual(pair_df.shape[0], 2)
        first = pair_df.loc[
            (pair_df["drug_id_norm"] == "CHEBI:1")
            & (pair_df["disease_id_norm"] == "MONDO:1")
        ].iloc[0]
        self.assertAlmostEqual(float(first["score"]), 0.8, places=6)
        self.assertEqual(int(first["label"]), 1)
        self.assertEqual(int(first["n_triplets"]), 2)

    def test_enrichment_factor(self) -> None:
        labels = np.array([1, 0, 1, 0, 0], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1], dtype=np.float32)
        efs = triplet.compute_enrichment_factors(labels, scores, fracs=[0.2, 0.4])
        self.assertIn("ef_20", efs)
        self.assertIn("ef_40", efs)
        self.assertEqual(efs["ef_20"]["k"], 1)
        self.assertAlmostEqual(efs["ef_20"]["ef"], 2.5, places=6)
        self.assertEqual(efs["ef_40"]["k"], 2)


if __name__ == "__main__":
    unittest.main()
