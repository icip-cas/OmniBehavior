#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# ========================================
# Benchmark settings
# ========================================
# Keep these values unchanged when reproducing the reported benchmark.

NUM_USERS=200                  # Number of users to sample.
TEST_LENGTH=30                 # Number of test actions sampled per user.

HISTORY_TIME_START="2025-09-01"  # History window start date (YYYY-MM-DD, empty = no lower bound).
HISTORY_TIME_END="2025-09-30"    # History window end date.
TEST_TIME_START="2025-10-01"     # Test window start date.
TEST_TIME_END="2025-11-30"       # Test window end date.


# ========================================
# Runtime settings
# ========================================

MAX_HISTORY_TOKENS=32000       # Maximum number of tokens used for history context.

MODEL="Qwen3-235B"              # Model name defined in config.py.

FORCE_RESAMPLE=false            # Force re-sampling even if the dataset already exists.
SKIP_MODEL_EVAL=false           # Skip model inference and only run metric computation (--skip-model-eval).
SKIP_DATA_PREP=false            # Skip dataset preparation and use an existing file (--skip-data-prep).
DATA_FILE=""                    # Path to a pre-built dataset; if set, skips data preparation (--data-file <path>).
VERBOSE=false                   # Print detailed per-field evaluation tables (--verbose).

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_RESAMPLE=true
            shift
            ;;
        --skip-model-eval)
            SKIP_MODEL_EVAL=true
            shift
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --data-file)
            if [ $# -lt 2 ]; then
                echo "Missing value for --data-file" >&2
                exit 1
            fi
            DATA_FILE=$2
            SKIP_DATA_PREP=true
            shift 2
            ;;
        --judge-model)
            if [ $# -lt 2 ]; then
                echo "Missing value for --judge-model" >&2
                exit 1
            fi
            TEACHER_MODEL_OVERRIDE=$2
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done


TIME_SUFFIX=""
if [ -n "${HISTORY_TIME_START}" ] || [ -n "${HISTORY_TIME_END}" ]; then
    H_START=""
    H_END=""
    [ -n "${HISTORY_TIME_START}" ] && H_START=$(echo "${HISTORY_TIME_START}" | sed 's/-//g' | cut -c5-8)
    [ -n "${HISTORY_TIME_END}" ] && H_END=$(echo "${HISTORY_TIME_END}" | sed 's/-//g' | cut -c5-8)
    TIME_SUFFIX="${TIME_SUFFIX}_h${H_START}~${H_END}"
fi
if [ -n "${TEST_TIME_START}" ] || [ -n "${TEST_TIME_END}" ]; then
    T_START=""
    T_END=""
    [ -n "${TEST_TIME_START}" ] && T_START=$(echo "${TEST_TIME_START}" | sed 's/-//g' | cut -c5-8)
    [ -n "${TEST_TIME_END}" ] && T_END=$(echo "${TEST_TIME_END}" | sed 's/-//g' | cut -c5-8)
    TIME_SUFFIX="${TIME_SUFFIX}_t${T_START}~${T_END}"
fi

OUTPUT_FILE="dataset/${NUM_USERS}u_${TEST_LENGTH}t${TIME_SUFFIX}.json"

# ========================================
# Step 1: Prepare evaluation data
# ========================================

if [ "${SKIP_DATA_PREP}" = true ]; then
    echo "Step 1/2: Skipping data preparation"

    if [ -n "${DATA_FILE}" ]; then
        ACTUAL_OUTPUT_FILE="${DATA_FILE}"
    else
        FOUND_FILE=$(find "$(dirname "${OUTPUT_FILE}")" -name "*.json" \
            -not -name "*_analysis.json" \
            -not -name "*_distribution.json" \
            -not -name "*_sampled_actions.json" \
            -type f -exec ls -t {} + 2>/dev/null | head -1 || true)

        if [ -z "$FOUND_FILE" ]; then
            echo "Error: no existing dataset found. Use --data-file to provide a path." >&2
            exit 1
        fi
        ACTUAL_OUTPUT_FILE="${FOUND_FILE}"
        echo "Using dataset: ${ACTUAL_OUTPUT_FILE}"
    fi

    if [ ! -f "$ACTUAL_OUTPUT_FILE" ]; then
        echo "Error: data file not found: $ACTUAL_OUTPUT_FILE" >&2
        exit 1
    fi
else
    echo "Step 1/2: Preparing evaluation data"

    PREPARE_ARGS=(
        --test-length "${TEST_LENGTH}"
        --max-history-tokens "${MAX_HISTORY_TOKENS}"
        --output "${OUTPUT_FILE}"
        --num-users "${NUM_USERS}"
    )
    [ -n "${HISTORY_TIME_START}" ] && PREPARE_ARGS+=(--history-time-start "${HISTORY_TIME_START}")
    [ -n "${HISTORY_TIME_END}" ]   && PREPARE_ARGS+=(--history-time-end "${HISTORY_TIME_END}")
    [ -n "${TEST_TIME_START}" ]    && PREPARE_ARGS+=(--test-time-start "${TEST_TIME_START}")
    [ -n "${TEST_TIME_END}" ]      && PREPARE_ARGS+=(--test-time-end "${TEST_TIME_END}")
    [ "${FORCE_RESAMPLE}" = true ] && PREPARE_ARGS+=(--force)

    PREPARE_LOG=$(mktemp)
    set +e
    python -u src/data/prepare_experiment_data.py "${PREPARE_ARGS[@]}" 2>&1 | tee "$PREPARE_LOG"
    PREPARE_EXIT_CODE=${PIPESTATUS[0]}
    set -e
    PREPARE_OUTPUT=$(cat "$PREPARE_LOG")
    rm -f "$PREPARE_LOG"

    if [ $PREPARE_EXIT_CODE -ne 0 ]; then
        echo "Error: data preparation failed with exit code $PREPARE_EXIT_CODE" >&2
        exit $PREPARE_EXIT_CODE
    fi

    ACTUAL_OUTPUT_FILE=$(printf '%s\n' "$PREPARE_OUTPUT" | grep "^ACTUAL_OUTPUT_PATH=" | cut -d'=' -f2- || true)
    if [ -z "$ACTUAL_OUTPUT_FILE" ] || [ ! -f "$ACTUAL_OUTPUT_FILE" ]; then
        echo "Error: could not determine the generated dataset path" >&2
        exit 1
    fi
fi

echo "Dataset: ${ACTUAL_OUTPUT_FILE}"
echo ""

# ========================================
# Step 2: Run model evaluation
# ========================================

echo "Step 2/2: Running model evaluation (model: ${MODEL})"
echo ""

EVAL_ARGS=(
    --use-fixed-data "${ACTUAL_OUTPUT_FILE}"
    --model "${MODEL}"
    --max-history-tokens "${MAX_HISTORY_TOKENS}"
)
[ "${SKIP_MODEL_EVAL}" = true ] && EVAL_ARGS+=(--skip-model-eval)
[ -n "${TEACHER_MODEL_OVERRIDE}" ] && EVAL_ARGS+=(--judge-model "${TEACHER_MODEL_OVERRIDE}")
[ "${VERBOSE}" = true ] && EVAL_ARGS+=(--verbose)

set +e
python src/evaluation/evaluator.py "${EVAL_ARGS[@]}"
EVAL_EXIT_CODE=$?
set -e

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed."
else
    echo "Error: evaluation failed with exit code $EVAL_EXIT_CODE" >&2
    exit $EVAL_EXIT_CODE
fi
