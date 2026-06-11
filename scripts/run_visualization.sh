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
    echo "Usage: bash scripts/run_visualization.sh --dataset <dataset_name>"
    exit 1
fi

python visualization/ablation/plot_ablation.py \
    --dataset ${DATASET} \
    --data-root results/ablation_result \
    --output-root results/figures/ablation

python visualization/baseline/plot_baseline.py \
    --dataset ${DATASET} \
    --base-root results/baseline_result \
    --output-root results/figures/baseline

python visualization/baseline/plot_baseline_v2.py \
    --dataset ${DATASET} \
    --base-root results/baseline_result \
    --output-root results/figures/baseline

python visualization/parameter/analyze_results.py \
    --dataset ${DATASET} \
    --root-dir results/parameter_result \
    --output-root results/figures/parameter