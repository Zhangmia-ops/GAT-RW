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
    echo "Usage: bash scripts/run_preprocess.sh --dataset <dataset_name>"
    exit 1
fi

python src/data_preprocessing.py \
    --raw-graph data/raw/${DATASET}.txt \
    --raw-query data/queries/raw_queries_target.txt \
    --output-root data/preprocessed