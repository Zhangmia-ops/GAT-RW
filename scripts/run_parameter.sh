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
    echo "Usage: bash scripts/run_parameter.sh --dataset <dataset_name>"
    exit 1
fi

python experiments/parameter/run_parameter_sensitivity.py \
    --dataset ${DATASET} \
    --data-dir data/ust_splits/${DATASET} \
    --output-dir results/parameter_result/${DATASET}