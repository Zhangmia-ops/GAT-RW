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
    echo "Usage: bash scripts/run_all_experiments.sh --dataset <dataset_name>"
    exit 1
fi

if [ ! -f "data/raw/${DATASET}.txt" ]; then
    echo "Dataset file not found!"
    exit 1
fi

bash scripts/run_preprocess.sh --dataset ${DATASET}
bash scripts/run_ust_split.sh --dataset ${DATASET}
bash scripts/run_ablation.sh --dataset ${DATASET}
bash scripts/run_baselines.sh --dataset ${DATASET}
bash scripts/run_parameter.sh --dataset ${DATASET}
bash scripts/run_visualization.sh --dataset ${DATASET}