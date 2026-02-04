from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.seed_everything,
                inputs="params:polypharmacy",
                outputs=None,
                name="seed_everything",
            ),
            node(
                func=nodes.build_deduped_dataset,
                inputs=[
                    "indications_norm",
                    "contraindications_norm",
                    "params:polypharmacy",
                ],
                outputs=["deduped_dataset", "conflict_count", "deduped_counts"],
                name="dedupe_examples",
            ),
            node(
                func=nodes.required_nodes_from_deduped,
                inputs="deduped_dataset",
                outputs="required_nodes",
                name="required_nodes",
            ),
            node(
                func=nodes.normalize_kg_edges,
                inputs=["kg_edges", "params:polypharmacy"],
                outputs="kg_edges_normalized",
                name="normalize_kg_edges",
            ),
            node(
                func=nodes.build_kg_embeddings,
                inputs=["kg_edges_normalized", "required_nodes", "params:polypharmacy"],
                outputs=["kg_nodes", "kg_node_ids", "kg_node_vectors"],
                name="build_kg_embeddings",
            ),
            node(
                func=nodes.filter_by_kg_coverage,
                inputs=["deduped_dataset", "kg_nodes"],
                outputs=["filtered_dataset", "dropped_rows", "drop_stats"],
                name="filter_by_kg",
            ),
            node(
                func=nodes.sample_run_df,
                inputs=["filtered_dataset", "params:polypharmacy"],
                outputs="run_dataset",
                name="sample_run_df",
            ),
            node(
                func=nodes.encode_dataset,
                inputs="run_dataset",
                outputs=[
                    "drug_to_idx",
                    "disease_to_idx",
                    "drug_seqs",
                    "disease_idxs",
                    "labels",
                ],
                name="encode_dataset",
            ),
            node(
                func=nodes.make_splits,
                inputs=["labels", "params:polypharmacy"],
                outputs=["train_idx", "val_idx", "test_idx"],
                name="make_splits",
            ),
            node(
                func=nodes.build_entity_embeddings,
                inputs=[
                    "kg_node_ids",
                    "kg_node_vectors",
                    "drug_to_idx",
                    "disease_to_idx",
                    "params:polypharmacy",
                ],
                outputs=[
                    "drug_embeddings",
                    "disease_embeddings",
                    "drug_vocab",
                    "disease_vocab",
                    "embedding_dim",
                ],
                name="build_entity_embeddings",
            ),
            node(
                func=nodes.export_config,
                inputs=["params:polypharmacy", "embedding_dim"],
                outputs="resolved_config",
                name="export_config",
            ),
            node(
                func=nodes.train_model,
                inputs=[
                    "drug_seqs",
                    "disease_idxs",
                    "labels",
                    "train_idx",
                    "val_idx",
                    "drug_embeddings",
                    "disease_embeddings",
                    "embedding_dim",
                    "params:polypharmacy",
                ],
                outputs=["best_checkpoint", "best_val_metrics"],
                name="train_model",
            ),
            node(
                func=nodes.evaluate_model,
                inputs=[
                    "best_checkpoint",
                    "drug_seqs",
                    "disease_idxs",
                    "labels",
                    "val_idx",
                    "test_idx",
                    "drug_embeddings",
                    "disease_embeddings",
                    "params:polypharmacy",
                ],
                outputs="eval_metrics",
                name="evaluate_model",
            ),
        ]
    )
