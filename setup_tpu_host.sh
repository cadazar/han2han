#!/usr/bin/env bash

# TPU Host Setup Script
#
# Provisions a TPU VM worker for Han2Han pre-training / fine-tuning: system
# packages, a Python 3.12 venv, the repository, the pinned dependencies, and
# JAX-on-TPU. Run on EACH worker of a pod slice before launching any stage.
#
# It is NOT idempotent across preemptions - a preempted VM is destroyed and
# recreated, so everything installs from scratch each time.
#
# Usage (run on all workers):
#   gcloud alpha compute tpus tpu-vm ssh TPU_NAME \
#     --zone=ZONE --worker=all --tunnel-through-iap \
#     --command='bash ~/setup_tpu_host.sh'
#
# Authentication tokens are read from the environment (optional):
#   HF_TOKEN          Hugging Face token for gated datasets/models
#   WANDB_API_KEY     Weights & Biases logging (omit to run with wandb disabled)
# Export them before invoking, or source them from your own secret store.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/cadazar/han2han.git}"
REPO_DIR="${REPO_DIR:-$HOME/han2han}"
VENV_DIR="${VENV_DIR:-$HOME/venv}"
JAX_VERSION="${JAX_VERSION:-0.9.0.1}"
FLAX_VERSION="${FLAX_VERSION:-0.12.4}"
LOG_PREFIX="[setup $(hostname)]"

log() { echo "$LOG_PREFIX $(date '+%H:%M:%S') $1"; }

log "Starting host setup..."

# ---------------------------------------------------------------
# 1. System packages (mecab, swig, openssh, rust)
# ---------------------------------------------------------------
log "Installing system packages..."

# disable docker repo - pre-installed on TPU VMs but unneeded and flaky
sudo rm -f /etc/apt/sources.list.d/docker.list 2>/dev/null || true
# kill unattended-upgrades and release apt locks before we start
# (common after preemption recovery: the daemon grabs locks on boot)
sudo systemctl stop unattended-upgrades 2>/dev/null || true
sudo systemctl disable unattended-upgrades 2>/dev/null || true
sudo killall -9 unattended-upgrade unattended-upgrades apt-get dpkg 2>/dev/null || true
sleep 2
sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock \
    /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null || true
sudo dpkg --configure -a 2>/dev/null || true

# retry apt-get update - transient mirror failures are common across many workers
for i in 1 2 3 4 5; do
    sudo apt-get update -qq && break
    log "WARN: apt-get update attempt $i failed, retrying in 10s..."
    sudo killall -9 apt-get dpkg 2>/dev/null || true
    sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock \
        /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null || true
    sudo dpkg --configure -a 2>/dev/null || true
    sleep 10
done
for i in 1 2 3 4 5; do
    sudo apt-get install -y -qq \
        mecab mecab-ipadic-utf8 libmecab-dev swig \
        openssh-client build-essential pkg-config libssl-dev \
        git > /dev/null 2>&1 && break
    log "WARN: apt-get install attempt $i failed, retrying in 10s..."
    sudo killall -9 apt-get dpkg 2>/dev/null || true
    sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock \
        /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null || true
    sudo dpkg --configure -a 2>/dev/null || true
    sudo apt-get update -qq 2>/dev/null || true
    sleep 10
done

# mecab-ko dictionary for Korean morpheme segmentation
curl -sL https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash > /dev/null 2>&1 || true
sudo mkdir -p /usr/local/lib/mecab/dic
sudo ln -sf /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ko-dic \
    /usr/local/lib/mecab/dic/mecab-ko-dic 2>/dev/null || true

# rust toolchain for han2han_tools - verify by invoking rustc, not just
# checking the binary exists; an interrupted rustup install can leave a
# rustc shim with a broken toolchain that fails on actual use
source "$HOME/.cargo/env" 2>/dev/null || true
if ! rustc -vV >/dev/null 2>&1; then
    log "Installing Rust toolchain (rustc -vV failed or missing)..."
    rm -rf "$HOME/.rustup" "$HOME/.cargo"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet 2>/dev/null
    source "$HOME/.cargo/env"
fi
log "Rust available: $(rustc --version)"

# ---------------------------------------------------------------
# 2. uv and Python 3.12
# ---------------------------------------------------------------
log "Installing uv and Python 3.12..."
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12 > /dev/null 2>&1

log "Creating venv at $VENV_DIR..."
uv venv --clear --python 3.12 --seed "$VENV_DIR" > /dev/null 2>&1
source "$VENV_DIR/bin/activate"
pip install -U pip --quiet 2>/dev/null
log "Venv active: $(python --version)"

# ---------------------------------------------------------------
# 3. Repository
# ---------------------------------------------------------------
log "Setting up repository..."
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git fetch origin main
    git reset --hard origin/main
    log "Repository updated"
else
    git clone "$REPO_URL" "$REPO_DIR"
    log "Repository cloned"
fi
cd "$REPO_DIR"

# ---------------------------------------------------------------
# 4. Python dependencies
# ---------------------------------------------------------------
log "Installing Python dependencies..."

# pinned project dependencies (CPU jax pins here are overridden by the TPU
# wheels in the next step)
pip install -r requirements.txt --quiet

# the Rust Hanja<->Hangul tooling
pip install ./han2han_tools --quiet

# ---------------------------------------------------------------
# 5. JAX-on-TPU (not pre-installed on TPU VMs; replaces the CPU jax above)
# ---------------------------------------------------------------
log "Installing JAX $JAX_VERSION and Flax $FLAX_VERSION (TPU)..."
pip install --force-reinstall -U "flax==$FLAX_VERSION" "jax[tpu]==$JAX_VERSION" wandb \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

INSTALLED_JAX=$(python -c "import jax; print(jax.__version__)" 2>/dev/null || echo "unknown")
log "JAX version: $INSTALLED_JAX"

# ---------------------------------------------------------------
# 6. Activation script
# ---------------------------------------------------------------
cat > "$HOME/activate_training_env.sh" << ENVEOF
#!/usr/bin/env bash
# source this before training: source ~/activate_training_env.sh

export PATH="\$HOME/.local/bin:\$PATH"
source "$VENV_DIR/bin/activate"
source "\$HOME/.cargo/env" 2>/dev/null || true

# authentication tokens (set these in your environment before sourcing, or
# leave unset to run without gated-asset access / with wandb disabled)
export HF_TOKEN="\${HF_TOKEN:-}"
export WANDB_API_KEY="\${WANDB_API_KEY:-}"
[ -z "\$WANDB_API_KEY" ] && export WANDB_MODE=disabled

export HF_HOME="\$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="\$HOME/.cache/huggingface/datasets"
export ENABLE_JAX_DISTRIBUTED=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore::SyntaxWarning"
export JAX_COMPILATION_CACHE_DIR="\$HOME/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

mkdir -p "\$HF_HOME/datasets" "\$HOME/jax_cache"
ENVEOF
chmod +x "$HOME/activate_training_env.sh"

# ---------------------------------------------------------------
# 7. System tuning
# ---------------------------------------------------------------
ulimit -c 0
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled" 2>/dev/null || true

log "Setup complete. Activate with: source ~/activate_training_env.sh"
log "Repo at: $REPO_DIR"
log "Venv: $VENV_DIR"
log "JAX: $INSTALLED_JAX"
