#!/usr/bin/env bash
set -e

# Han2Han Classifier Fine-tuning - Host-Based Distributed TPU Pod Slice Launch
# Requires setup_tpu_host.sh to have been run first on all workers.

TPU_POD_NAME="${TPU_POD_NAME:-han2han-pod}"
TPU_ZONE="${TPU_ZONE:-us-central2-b}"
CONFIG_FILE="${CONFIG_FILE:-configs/classifier_temporal.yaml}"
DATA_BUCKET="${DATA_BUCKET:-}"
RUN_NAME="han2han_classifier_$(date +%Y%m%d_%H%M%S)"

# attention kernel selection (override via env)
# TPU_USE_SPLASH_ATTN: 1 = splash (fast on v5e/v6e, slow on v4 megacore)
# HAN2HAN_SPLASH: per-call splash override (on/off/auto, empty = heuristic).
# HAN2HAN_CROSS_SPLASH: cross-attn-only override; falls back to HAN2HAN_SPLASH.
# HAN2HAN_USE_QUANT_ATTN_WEIGHTS: 1 = uint8 attn weights residual.
#   WARNING: only safe for sharp attention (post-trained / fine-tuning).
#   For from-scratch pretraining at long SL the uint8 quantization underflows
#   (aw < 1/510 quantizes to 0), zeroing Q/K/V gradients. Default 0.
TPU_USE_SPLASH_ATTN="${TPU_USE_SPLASH_ATTN:-1}"
HAN2HAN_SPLASH="${HAN2HAN_SPLASH:-}"
HAN2HAN_CROSS_SPLASH="${HAN2HAN_CROSS_SPLASH:-}"
HAN2HAN_USE_QUANT_ATTN_WEIGHTS="${HAN2HAN_USE_QUANT_ATTN_WEIGHTS:-0}"

# validate config file exists locally
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available classifier configs:"
    ls -la configs/classifier*.yaml 2>/dev/null || echo "  No configs/classifier*.yaml files found"
    exit 1
fi

TEMP_SCRIPT="/tmp/classifier_host_training_$(basename ${CONFIG_FILE%.yaml})_$$.sh"

echo "=== Han2Han Classifier Fine-tuning Launch (Host Mode) ==="
echo "TPU Pod: $TPU_POD_NAME"
echo "Zone: $TPU_ZONE"
echo "Config: $CONFIG_FILE"
echo "Data Bucket: ${DATA_BUCKET:-local}"
echo "Run Name: $RUN_NAME"
echo "Splash Attn: $TPU_USE_SPLASH_ATTN"
echo "Splash Override: ${HAN2HAN_SPLASH:-<heuristic>}"
echo "Cross-Splash Override: ${HAN2HAN_CROSS_SPLASH:-<heuristic>}"
echo "Quant Attn Weights: $HAN2HAN_USE_QUANT_ATTN_WEIGHTS"
echo "Time: $(date)"
echo "=========================================="

# build the training command
TRAIN_CMD="$HOME/venv/bin/python -u finetune_classifier.py --config $CONFIG_FILE"
if [ -n "$DATA_BUCKET" ]; then
    TRAIN_CMD="$TRAIN_CMD --data_bucket $DATA_BUCKET"
fi

cat > "$TEMP_SCRIPT" << 'WORKER_EOF'
#!/bin/bash
set -e

echo "Starting Han2Han classifier fine-tuning on TPU worker $(hostname) (host mode)..."

# clear TPU device locks
echo "Clearing TPU device locks..."
sudo fuser -k /dev/accel0 /dev/accel1 /dev/accel2 /dev/accel3 /dev/vfio/* 2>/dev/null || true
sudo rm -f /tmp/libtpu_lockfile /tmp/.tpu_* 2>/dev/null || true
sudo rm -f /dev/shm/.tpu_* /dev/shm/libtpu_* 2>/dev/null || true
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled" 2>/dev/null || true

# kill any existing training/fine-tuning processes
echo "Killing existing processes..."
pkill -f "finetune_classifier" 2>/dev/null || true
pkill -f "finetune_sft" 2>/dev/null || true
pkill -f "train_han2han" 2>/dev/null || true
sleep 2

# activate the training environment
if [ ! -f "$HOME/activate_training_env.sh" ]; then
    echo "ERROR: activate_training_env.sh not found. Run setup_tpu_host.sh first."
    exit 1
fi
source "$HOME/activate_training_env.sh"

# update repo to latest
cd "$HOME/Han2Han"
echo "Updating repository..."
git fetch origin main
git reset --hard origin/main

# overwrite config with the version SCP'd from the launch host
if [ -f /tmp/_uploaded_config.yaml ]; then
    cp /tmp/_uploaded_config.yaml "CONFIG_FILE_PLACEHOLDER"
    echo "Config overwritten with uploaded version: CONFIG_FILE_PLACEHOLDER"
fi

# override JAX cache dir for this specific run
export JAX_COMPILATION_CACHE_DIR="$HOME/jax_cache/CACHE_NAME_PLACEHOLDER"

ulimit -c 0

LOG_FILE="$HOME/training_$(hostname).log"
PID_FILE="$HOME/training.pid"

echo "Starting classifier fine-tuning..."
echo "Log file: $LOG_FILE"
echo "Training command: TRAIN_CMD_PLACEHOLDER"

# attention options (substituted from launcher env)
export TPU_USE_SPLASH_ATTN=SPLASH_ATTN_PLACEHOLDER
export HAN2HAN_SPLASH=SPLASH_OVERRIDE_PLACEHOLDER
export HAN2HAN_CROSS_SPLASH=CROSS_SPLASH_OVERRIDE_PLACEHOLDER
export HAN2HAN_USE_QUANT_ATTN_WEIGHTS=QUANT_ATTN_PLACEHOLDER

nohup bash -c '
source "$HOME/activate_training_env.sh"
cd "$HOME/Han2Han"
export JAX_COMPILATION_CACHE_DIR="$HOME/jax_cache/CACHE_NAME_PLACEHOLDER"
TRAIN_CMD_PLACEHOLDER
wandb sync 2>/dev/null || true
' > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Classifier fine-tuning started with PID $TRAIN_PID"

# wait briefly and check it hasn't crashed immediately
sleep 5
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "Classifier process is running (PID $TRAIN_PID)"
else
    echo "ERROR: Classifier process exited immediately. Check $LOG_FILE"
    tail -20 "$LOG_FILE"
    exit 1
fi
WORKER_EOF

# replace placeholders
sed -i "s|TRAIN_CMD_PLACEHOLDER|$TRAIN_CMD|g" "$TEMP_SCRIPT"
sed -i "s|CACHE_NAME_PLACEHOLDER|han2han_classifier|g" "$TEMP_SCRIPT"
sed -i "s|CONFIG_FILE_PLACEHOLDER|$CONFIG_FILE|g" "$TEMP_SCRIPT"
sed -i "s|SPLASH_ATTN_PLACEHOLDER|$TPU_USE_SPLASH_ATTN|g" "$TEMP_SCRIPT"
sed -i "s|CROSS_SPLASH_OVERRIDE_PLACEHOLDER|$HAN2HAN_CROSS_SPLASH|g" "$TEMP_SCRIPT"
sed -i "s|SPLASH_OVERRIDE_PLACEHOLDER|$HAN2HAN_SPLASH|g" "$TEMP_SCRIPT"
sed -i "s|QUANT_ATTN_PLACEHOLDER|$HAN2HAN_USE_QUANT_ATTN_WEIGHTS|g" "$TEMP_SCRIPT"

chmod +x "$TEMP_SCRIPT"

# copy config and training script to all workers
echo "Copying config file ($CONFIG_FILE) to all TPU workers..."
gcloud alpha compute tpus tpu-vm scp \
    "$CONFIG_FILE" "$TPU_POD_NAME":/tmp/_uploaded_config.yaml \
    --zone="$TPU_ZONE" --worker=all --tunnel-through-iap

echo "Copying classifier script to all TPU workers..."
gcloud alpha compute tpus tpu-vm scp \
    "$TEMP_SCRIPT" "$TPU_POD_NAME":/tmp/classifier_host_training.sh \
    --zone="$TPU_ZONE" --worker=all --tunnel-through-iap

echo "Starting distributed classifier fine-tuning on all TPU workers..."
gcloud alpha compute tpus tpu-vm ssh "$TPU_POD_NAME" \
    --zone="$TPU_ZONE" --worker=all --tunnel-through-iap \
    --command='bash /tmp/classifier_host_training.sh'

rm -f "$TEMP_SCRIPT"

echo ""
echo "Classifier fine-tuning launched on all TPU workers!"
echo "Check logs with:"
echo "  gcloud alpha compute tpus tpu-vm ssh $TPU_POD_NAME --zone=$TPU_ZONE --worker=all --tunnel-through-iap --command='tail -50 ~/training_\$(hostname).log'"
echo "Monitor WandB: https://wandb.ai/cellikadams/han2han-classifier"
