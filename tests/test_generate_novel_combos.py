import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from polypharmacy import utils
from scripts import generate_novel_combos as gen


class GenerateNovelCombosTests(unittest.TestCase):
    def test_target_disease_must_exist(self) -> None:
        disease_to_idx = {"MONDO:1": 0, "MONDO:2": 1}
        self.assertEqual(gen._validate_target_disease("MONDO:2", disease_to_idx), 1)
        with self.assertRaises(ValueError):
            gen._validate_target_disease("CHEBI:1", disease_to_idx)
        with self.assertRaises(ValueError):
            gen._validate_target_disease("MONDO:404", disease_to_idx)

    def test_candidate_generation_variable_sizes(self) -> None:
        candidate_drugs = ["CHEBI:3", "CHEBI:1", "CHEBI:2"]
        combos = gen._generate_candidate_combos(candidate_drugs, min_combo_size=2, max_combo_size=3)
        self.assertIn(("CHEBI:1", "CHEBI:2"), combos)
        self.assertIn(("CHEBI:1", "CHEBI:2", "CHEBI:3"), combos)
        self.assertEqual(len(combos), 4)  # C(3,2) + C(3,3)

    def test_per_disease_novelty_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deduped_path = os.path.join(tmpdir, "deduped_dataset.csv")
            pd.DataFrame(
                [
                    {
                        "drug_set": str(["CHEBI:1", "CHEBI:2"]),
                        "condition_id_norm": "MONDO:1",
                        "label": 1,
                    },
                    {
                        "drug_set": str(["CHEBI:1", "CHEBI:2"]),
                        "condition_id_norm": "MONDO:2",
                        "label": 1,
                    },
                ]
            ).to_csv(deduped_path, index=False)

            known = gen._load_known_combos_for_target_disease(
                model_output_dir=tmpdir,
                novelty_source="deduped",
                target_disease="MONDO:1",
            )
            self.assertIn(("CHEBI:1", "CHEBI:2"), known)
            self.assertEqual(len(known), 1)

    def test_threshold_filters(self) -> None:
        combos = [("CHEBI:1", "CHEBI:2"), ("CHEBI:1", "CHEBI:3"), ("CHEBI:2", "CHEBI:3")]
        probs = np.array([0.95, 0.85, 0.99], dtype=np.float32)
        ranked, counts = gen._rank_and_filter(
            candidate_combos=combos,
            probs=probs,
            target_disease="MONDO:1",
            model_output_dir="artifacts_refined_precomputed",
            min_prob=0.9,
            top_percent=50.0,
            top_n=5,
        )
        self.assertEqual(counts["num_scored"], 3)
        self.assertEqual(counts["num_after_prob_filter"], 2)
        self.assertEqual(counts["num_after_percent_filter"], 1)
        self.assertEqual(counts["num_exported"], 1)
        self.assertAlmostEqual(float(ranked.iloc[0]["p_indication"]), 0.99, places=6)

    def test_reproducible_max_candidates_sampling(self) -> None:
        combos = [("CHEBI:1", "CHEBI:2"), ("CHEBI:1", "CHEBI:3"), ("CHEBI:2", "CHEBI:3")]
        first = gen._select_max_candidates(combos, max_candidates=2, seed=13)
        second = gen._select_max_candidates(combos, max_candidates=2, seed=13)
        self.assertEqual(first, second)

    def test_load_vocab_maps_skips_pad(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.save_json(
                os.path.join(tmpdir, "drug_vocab.json"),
                {"ids": ["<PAD>", "CHEBI:1", "CHEBI:2"]},
            )
            utils.save_json(
                os.path.join(tmpdir, "disease_vocab.json"),
                {"ids": ["MONDO:1", "MONDO:2"]},
            )
            drug_to_idx, disease_to_idx = gen._load_vocab_maps(tmpdir)
            self.assertNotIn("<PAD>", drug_to_idx)
            self.assertEqual(drug_to_idx["CHEBI:1"], 1)
            self.assertEqual(disease_to_idx["MONDO:2"], 1)


if __name__ == "__main__":
    unittest.main()
