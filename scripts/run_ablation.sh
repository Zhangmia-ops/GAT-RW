#!/bin/bash

DATASET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Usage: bash scripts/run_ablation.sh --dataset <dataset_name>"
    exit 1
fi

mkdir -p results/ablation_result/${DATASET}
mkdir -p data/ust_splits/${DATASET}

python experiments/ablation/run_ablation_study.py \
    --dataset ${DATASET} \
    --data-dir data/ust_splits/${DATASET} \
    --output-dir results/ablation_result/${DATASET}