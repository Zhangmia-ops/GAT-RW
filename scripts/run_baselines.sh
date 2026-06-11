#!/bin/bash

DATASET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Usage: bash scripts/run_baselines.sh --dataset <dataset_name>"
    exit 1
fi

MASKS=(100 200)

for MASK in "${MASKS[@]}"; do
    for MODEL in GCN SAGE MLP; do
    python experiments/baseline/run_baselines.py \
      --model ${MODEL} \
      --dataset ${DATASET} \
      --data-dir data/ust_splits/${DATASET} \
      --output-dir results/baseline_result/${DATASET} \
      --train_file data/ust_splits/${DATASET}/training_graphs/graph_minus_${MASK}_edges.txt \
      --test_file data/ust_splits/${DATASET}/test_sets/graph_minus_${MASK}_REMOVED.txt \
      --epochs 200
    done
    python experiments/baseline/GAT_RW.py \
    --train_file data/ust_splits/${DATASET}/training_graphs/graph_minus_${MASK}_edges.txt \
    --test_file data/ust_splits/${DATASET}/test_sets/graph_minus_${MASK}_REMOVED.txt \
    --output_file results/baseline_result/${DATASET}/GAT_RW_${DATASET}_minus${MASK}.json \
    --dataset_name ${DATASET} \
    --mask_cnt ${MASK}
    for METHOD in CN AA; do
        python experiments/baseline/heuristic_baselines.py \
        --method ${METHOD} \
        --dataset ${DATASET} \
        --mask_cnt ${MASK} \
        --train_file data/ust_splits/${DATASET}/training_graphs/graph_minus_${MASK}_edges.txt \
        --test_file data/ust_splits/${DATASET}/test_sets/graph_minus_${MASK}_REMOVED.txt \
        --output_file results/baseline_result/${DATASET}/${METHOD}_${DATASET}_minus${MASK}.json
    done
done