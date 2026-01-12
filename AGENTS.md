You are an expert ML engineer implementing a biomedical polypharmacy prediction system.

Your task is to build an end-to-end Python pipeline that predicts whether a drug combination is an
INDICATION (positive / safe / therapeutic) or a CONTRAINDICATION (negative / unsafe / adverse)
for a given disease.

=====================
DATA SOURCES
=====================

1. indications_norm.csv
2. contraindications_norm.csv

These two CSV files share the SAME SCHEMA and differ ONLY in label semantics.

--------------------------------------------------
CSV STRUCTURE (IMPORTANT)
--------------------------------------------------

Each row represents a (multi-drug set, disease) example.

Example row:

primary_drug_id                  ['DRUGBANK:DB01006']
secondary_drug_id                ['DRUGBANK:DB11730']
condition_id_norm                MONDO:0000618
primary_drug_id_norm             ['CHEBI:6413']
primary_drug_id_norm_label       ['Letrozole']
secondary_drug_id_norm           ['CHEBI:230905']
secondary_drug_id_norm_label     ['Ribociclib']

CRITICAL NOTES:
- Columns like primary_drug_id_norm and secondary_drug_id_norm are STORED AS STRINGS
  that represent Python lists (e.g., "['CHEBI:6413']").
- You MUST parse these using ast.literal_eval.
- Even when there is only one drug, values are still lists.
- secondary_drug_id_norm may contain MULTIPLE drugs (3+ drug combinations).

--------------------------------------------------
LABEL SEMANTICS
--------------------------------------------------

- Every row in indications_norm.csv is a POSITIVE example:
      label = 1   (INDICATION)

- Every row in contraindications_norm.csv is a NEGATIVE example:
      label = 0   (CONTRAINDICATION)

There is NO label column in the CSVs.
The label is inferred solely from which file the row came from.

--------------------------------------------------
SEMANTIC MEANING OF A ROW
--------------------------------------------------

Each row represents:

(drug_1, drug_2, ..., drug_N, condition) → label

You must construct the full drug set as:

    drugs = primary_drug_id_norm + secondary_drug_id_norm

Ignore all *_label columns for modeling (they are human-readable only).

Use NORMALIZED IDs:
- Drugs: CHEBI:*
- Diseases: MONDO:*

=====================
KNOWLEDGE GRAPH DATA
=====================

3. kg_edges.parquet

- Biomedical knowledge graph edge list.
- Nodes include drugs, diseases, targets, pathways, etc.
- Use this graph to generate embeddings for drugs and diseases.
- Drug and disease embeddings MUST live in the SAME vector space.

=====================
GOALS
=====================

1. Data loading
- Load BOTH indications_norm.csv and contraindications_norm.csv
- Assign labels:
    - indications → label = 1
    - contraindications → label = 0
- Parse list-valued columns using ast.literal_eval
- Build variable-length drug sets per row
- Map drug IDs and disease IDs to integer indices
- Output tensors:
    - drug index sequences (variable length)
    - disease indices
    - binary labels

2. Knowledge graph embeddings
- Load kg_edges.parquet into a graph
- Use node2vec to generate embeddings
- Extract embeddings for:
    - CHEBI drugs
    - MONDO diseases
- Build embedding lookup matrices
- Ensure consistent indexing between KG embeddings and dataset IDs

3. Model architecture (PyTorch)

Implement the following architecture:

Drug A      Drug B      Drug C      ...      Drug N
  │           │           │                   │
  ▼           ▼           ▼                   ▼
Embedding   Embedding   Embedding           Embedding
  │           │           │                   │
  └──────┬────┴──────┬────┴──────┬───── ... ──┘
         │           │           │
         ▼           ▼           ▼
      LSTM cell   LSTM cell   LSTM cell   (shared weights)
         │           │           │
         ▼           ▼           ▼
       h₁          h₂          h₃          ... hₙ
                          │
                          ▼
                Drug-combo embedding hₙ
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
   Disease embedding c        (KG or learned lookup)
            │
            └──────────── concat ─────────────┘
                          │
                          ▼
                    MLP classifier
                          │
                          ▼
             Indicated / Contraindicated

Key requirements:
- LSTM must support variable-length sequences via padding + masking
- Use the FINAL hidden state as the drug-combination embedding
- Disease embedding is concatenated AFTER the LSTM
- Output is a single logit for binary classification

4. Training, validation, and testing (REPRODUCIBILITY REQUIRED)

- Implement a deterministic train / validation / test split:
    - Use fixed random seeds
    - Ensure the same examples appear in each split every run
- Do NOT reshuffle data differently across runs
- Provide a standalone evaluation script that:
    - loads a saved model
    - evaluates on the fixed validation and test sets
    - reports ROC-AUC and accuracy
- Set and document all relevant random seeds:
    - Python
    - NumPy
    - PyTorch (CPU and CUDA)

5. Training loop
- PyTorch training loop
- BCEWithLogitsLoss
- Proper batching of variable-length drug sequences
- Save best model checkpoint based on validation ROC-AUC
- Log metrics clearly

=====================
DELIVERABLES
=====================

- Fully runnable Python code (no pseudocode)
- Clear comments explaining:
    - CSV parsing
    - label construction from file source
    - KG embedding generation
    - deterministic data splitting
    - model design choices
- Reasonable defaults (embedding dim ~128–256)
- Optional:
    - freeze vs fine-tune KG embeddings
    - reproducible config file for seeds and paths

Focus on correctness, clarity, determinism, and research-quality implementation.

