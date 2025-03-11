#!/bin/bash

SIZE=500
DELTA=3.0
EPSILONS=(
    0.4
    0.35
    0.3
    0.2
    0.1
)
CLASSIFIERS=("Baseline.onnx" "DL2.onnx" "Goedel.onnx")

LOG_DIR="logs"

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

python create_dataset.py --size=$SIZE

for EPSILON in "${EPSILONS[@]}"; do
    for CLASSIFIER in "${CLASSIFIERS[@]}"; do
        CLASSIFIER_BASE=$(basename "$CLASSIFIER" .onnx)
        LOG_FILE="$LOG_DIR/log_${CLASSIFIER_BASE}_eps_${EPSILON//./_}.txt"

        echo "====Verifying $CLASSIFIER with epsilon=$EPSILON====" | tee -a "$LOG_FILE"
        { /usr/bin/time -v vehicle --no-warnings verify \
            -s property.vcl \
            -n classifier:$CLASSIFIER \
            -p epsilon:$EPSILON \
            -p delta:$DELTA \
            -d trainingImages:t${SIZE}-images.idx \
            -d trainingLabels:t${SIZE}-labels.idx \
            --no-sat-print \
            -v Marabou \
            -a "--timeout=30"; } 2>&1 | tee -a "$LOG_FILE"
    done
done

python results.py --size=$SIZE