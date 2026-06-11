#!/bin/bash

DATASET=""
RAW_QUERY="data/queries/raw_queries_target.txt"

HIDDEN_DIM=32
HEADS=4
LR=0.01
EPOCHS=1000
L=6
N_WALKS=3000
ALPHA=0.2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --raw_query)
            RAW_QUERY="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --heads)
            HEADS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --L)
            L="$2"
            shift 2
            ;;
        --n_walks)
            N_WALKS="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage:"
            echo "bash scripts/run_main.sh --dataset <dataset_name> [--hidden_dim 64] [--heads 4] [--lr 0.005] [--epochs 1000] [--L 6] [--n_walks 4000] [--alpha 0.15]"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Missing required argument: --dataset"
    echo "Example:"
    echo "bash scripts/run_main.sh --dataset amazon_weight --hidden_dim 64 --heads 4 --lr 0.005 --epochs 1000 --L 6 --n_walks 4000 --alpha 0.15"
    exit 1
fi

RAW_GRAPH="data/raw/${DATASET}.txt"

PREPROCESS_ROOT="data/preprocessed"
DATA_DIR="${PREPROCESS_ROOT}/${DATASET}"

OUTPUT_DIR="results/main_result/${DATASET}"

COMPONENT_INDEX="${DATA_DIR}/component_index.json"
MAPPING_FILE="${DATA_DIR}/M_node_dictionary.txt"

if [ ! -f "$RAW_GRAPH" ]; then
    echo "Dataset file not found: $RAW_GRAPH"
    exit 1
fi

if [ ! -f "$RAW_QUERY" ]; then
    echo "Query file not found: $RAW_QUERY"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Main Model Inference ==="
echo "Dataset: $DATASET"
echo "Raw graph: $RAW_GRAPH"
echo "Raw query: $RAW_QUERY"
echo "hidden_dim: $HIDDEN_DIM"
echo "heads: $HEADS"
echo "lr: $LR"
echo "epochs: $EPOCHS"
echo "L: $L"
echo "n_walks: $N_WALKS"
echo "alpha: $ALPHA"

echo "=== Step 1: Preprocess graph and query ==="

python src/data_preprocessing.py \
    --raw-graph "$RAW_GRAPH" \
    --raw-query "$RAW_QUERY" \
    --output-root "$PREPROCESS_ROOT"

echo "=== Step 2: Run main GAT-RW inference model ==="

for EDGE_FILE in ${DATA_DIR}/subgraph_*_edges.txt; do
    if [ ! -f "$EDGE_FILE" ]; then
        continue
    fi

    BASENAME=$(basename "$EDGE_FILE")
    SUBGRAPH_ID=$(echo "$BASENAME" | sed -E 's/subgraph_([0-9]+)_edges.txt/\1/')

    QUERY_FILE="${DATA_DIR}/subgraph_${SUBGRAPH_ID}_queries.txt"
    OUTPUT_FILE="${OUTPUT_DIR}/subgraph_${SUBGRAPH_ID}_predictions.txt"

    if [ ! -f "$QUERY_FILE" ]; then
        echo "Skip subgraph ${SUBGRAPH_ID}: query file not found."
        continue
    fi

    echo "Running main model on subgraph ${SUBGRAPH_ID}..."

    python src/gat_rw_infer.py \
        --train_file "$EDGE_FILE" \
        --query_file "$QUERY_FILE" \
        --component_index "$COMPONENT_INDEX" \
        --mapping_file "$MAPPING_FILE" \
        --output_file "$OUTPUT_FILE" \
        --dataset_name "$DATASET" \
        --hidden_dim "$HIDDEN_DIM" \
        --heads "$HEADS" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --L "$L" \
        --n_walks "$N_WALKS" \
        --alpha "$ALPHA"
done

echo "=== Step 3: Merge all subgraph prediction files ==="

cat ${OUTPUT_DIR}/subgraph_*_predictions.txt > ${OUTPUT_DIR}/${DATASET}_predicted_edges.txt

echo "Done."
echo "Final prediction file: ${OUTPUT_DIR}/${DATASET}_predicted_edges.txt"