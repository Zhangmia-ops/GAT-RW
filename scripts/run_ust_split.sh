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
    echo "Usage: bash scripts/run_ust_split.sh --dataset <dataset_name>"
    exit 1
fi

python experiments/split/UST_minus.py \
    --input data/raw/${DATASET}.txt \
    --output-dir data/ust_splits/${DATASET} \
    --remove-list 20 50 100 150 200