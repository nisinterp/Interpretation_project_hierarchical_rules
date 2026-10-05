#!/usr/bin/env bash
grep -q $'\r' "$0" && { sed -i 's/\r$//' "$0"; exec bash "$0" "$@"; } # converts Windows (CRLF) line endings, then restarts
#
# Reproduce the question formation (QF) training runs of Section 2.3 of
# "Data Drives Unstable Hierarchical Generalization in LMs" (Qin et al., EMNLP 2025).
# Written for Google Colab (GPU runtime), but works on any Linux machine with an NVIDIA GPU.
#
# Setup from the paper (Sec. 2.3), for each random seed (the paper uses 50; this script defaults to 10):
#   - original McCoy et al. (2018) QF data: data_utils/question_formation_data/question.{train,val,test}
#     (question.test is the OOD generalization set; "test_aux" in the logs = OOD generalization accuracy)
#   - decoder-only Transformer LM: 6 layers, 8 heads, 512-dim embeddings (tied input/output embeddings)
#   - causal LM objective, 300K steps, Adam, lr 1e-4, linear decay (10K warmup steps)
#   - evaluation every 2K steps (the granularity used for total variation in Sec. 4.1)
#
# Colab usage (in notebook cells):
#   1. Runtime > Change runtime type > T4 GPU
#   2. Upload the whole `code/` folder (e.g. to Google Drive), then mount Drive so results survive the session:
#        from google.colab import drive; drive.mount('/content/drive')
#   3. !bash /content/drive/MyDrive/code/reproduce_qf_section2.sh
#   Re-run step 3 in a new session to continue: finished seeds are skipped (an interrupted seed restarts).
#
# Configuration via environment variables (e.g. `!SEEDS="0 1 2" bash reproduce_qf_section2.sh`):
#   SEEDS          space-separated seeds              (default: 0 1 ... 9; the paper uses 50 unlisted seeds)
#   GPUS           space-separated GPU ids            (default: all GPUs visible to PyTorch)
#   JOBS_PER_GPU   concurrent runs per GPU            (default: 1; the model is small, 2 may give more throughput)
#   OUT_DIR        checkpoints, logs and wandb files  (default: /content/drive/MyDrive/qf_section2_runs if Drive
#                                                      is mounted, otherwise <code dir>/runs/qf_section2)
#   VENV_DIR       Python environment location        (default: /content/hiergenv_venv on Colab, else <code dir>/.venv)
#   WANDB_MODE     offline | online | disabled        (default: offline; upload later with
#                                                      `wandb sync $OUT_DIR/wandb/offline-run-*`)
#   WANDB_ENTITY   wandb entity, only for WANDB_MODE=online

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- locate the code folder
CODE_DIR="${CODE_DIR:-}"
if [[ -z "$CODE_DIR" ]]; then
    for d in "$SCRIPT_DIR" "$SCRIPT_DIR/code"; do
        [[ -f "$d/train_transformers.py" ]] && CODE_DIR="$d" && break
    done
fi
if [[ -z "$CODE_DIR" || ! -f "$CODE_DIR/train_transformers.py" ]]; then
    echo "Cannot find train_transformers.py next to this script. Put the script inside the code folder" >&2
    echo "(together with data_utils/, layers/, models/, ...) or set CODE_DIR." >&2
    exit 1
fi
cd "$CODE_DIR"

if [[ -d /content ]]; then ON_COLAB=1; else ON_COLAB=0; fi

# ---------------------------------------------------------------- Python environment
# Colab's own Python and packages are much newer than the code expects, so build a separate
# Python 3.11 environment with the versions this code was tested with.
if [[ $ON_COLAB == 1 ]]; then VENV_DIR="${VENV_DIR:-/content/hiergenv_venv}"; else VENV_DIR="${VENV_DIR:-$CODE_DIR/.venv}"; fi
PY="$VENV_DIR/bin/python"

if [[ ! -f "$VENV_DIR/.installed" ]]; then
    echo "=== Creating Python 3.11 environment in $VENV_DIR (a few minutes; done once per Colab session)"
    command -v uv > /dev/null || python3 -m pip install -q uv
    UV="$(command -v uv || echo "python3 -m uv")"
    $UV venv -q --python 3.11 "$VENV_DIR"
    $UV pip install -q --python "$PY" torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    $UV pip install -q --python "$PY" \
        numpy==1.26.4 scipy==1.13.1 tqdm==4.66.4 pandas==2.1.0 \
        transformers==4.36.0 tokenizers==0.15.2 huggingface-hub==0.23.0 \
        datasets==2.14.5 pyarrow==15.0.2 \
        wandb==0.17.1 accelerate==0.32.1 matplotlib
    touch "$VENV_DIR/.installed"
fi

# ---------------------------------------------------------------- code compatibility patches
# The repository as published imports a module that exists only in the authors' private fork of
# transformers and writes wandb files to the authors' cluster. Patch both (no-op if already patched).
"$PY" - <<'EOF'
import re
def patch(path, old, new):
    src = open(path).read()
    if old in src:
        open(path, "w").write(src.replace(old, new))
        print(f"patched {path}")

patch("layers/transformer/gated_multi_head_attention.py",
      "\nfrom transformers.gated_bert_utilities import ConcreteGate\n",
      "\ntry:\n    from transformers.gated_bert_utilities import ConcreteGate\n"
      "except ImportError:  # only available in the forked transformers; needed for --gated-model\n"
      "    def ConcreteGate(*args, **kwargs):\n"
      "        raise ImportError(\"ConcreteGate requires the forked transformers library "
      "(transformers.gated_bert_utilities); install it to use --gated-model\")\n")
src = open("train_transformers.py").read()
fixed = re.sub(r'wandb_dir = "/n/[^"]*"',
               'wandb_dir = os.environ.get("WANDB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb"))',
               src)
fixed = fixed.replace('WANDB_ENTITY_NAME = "harvardml"',
                      'WANDB_ENTITY_NAME = os.environ.get("WANDB_ENTITY", "harvardml")')
fixed = fixed.replace("else: mode = 'online'", 'else: mode = os.environ.get("WANDB_MODE", "online")')
if fixed != src:
    open("train_transformers.py", "w").write(fixed)
    print("patched train_transformers.py")
EOF

# ---------------------------------------------------------------- GPUs
if [[ -z "${GPUS:-}" ]]; then
    n_gpus="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
    GPUS="$(seq -s ' ' 0 $(( n_gpus - 1 )) 2> /dev/null || true)"
    if [[ "$n_gpus" -eq 0 ]]; then
        echo "PyTorch sees no GPU." >&2
        [[ $ON_COLAB == 1 ]] && echo "In Colab: Runtime > Change runtime type > T4 GPU, then run this again." >&2
        "$PY" -c 'import torch; print("torch", torch.__version__, "CUDA build", torch.version.cuda)' >&2
        (nvidia-smi || /opt/bin/nvidia-smi) >&2 2> /dev/null || echo "nvidia-smi not available" >&2
        exit 1
    fi
fi
read -r -a GPU_LIST <<< "$GPUS"
"$PY" -c 'import torch; [print(f"GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'

# ---------------------------------------------------------------- output location
if [[ -z "${OUT_DIR:-}" ]]; then
    if [[ -d /content/drive/MyDrive ]]; then
        OUT_DIR=/content/drive/MyDrive/qf_section2_runs
    else
        OUT_DIR="$CODE_DIR/runs/qf_section2"
        [[ $ON_COLAB == 1 ]] && echo "WARNING: Google Drive is not mounted; results in $OUT_DIR are lost when the Colab session ends." >&2
    fi
fi
mkdir -p "$OUT_DIR/logs"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$OUT_DIR}"
export WANDB_SILENT=true

SEEDS="${SEEDS:-$(seq -s ' ' 40 49)}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
read -r -a SEED_LIST <<< "$SEEDS"
echo "=== ${#SEED_LIST[@]} seeds on GPU(s) ${GPU_LIST[*]}, $JOBS_PER_GPU run(s) per GPU; output: $OUT_DIR"

# ---------------------------------------------------------------- training
train_one_seed() {
    local seed=$1 gpu=$2
    local name="qf_original_seed${seed}"
    local log="$OUT_DIR/logs/${name}.log"
    if [[ -f "$OUT_DIR/logs/${name}.done" ]]; then
        echo "[seed $seed] already finished, skipping"
        return 0
    fi
    echo "[seed $seed] starting on GPU $gpu -> $log"
    if "$PY" -u train_transformers.py \
        --dataset question_original \
        --callback \
        --encoder_n_layers 6 \
        --n_heads 8 \
        --vec_dim 512 \
        --tied-embedding \
        --dropout 0.1 \
        --label-smoothing 0.0 \
        --batch_size 8 \
        --lr 1e-4 \
        --num_warmup_steps 10000 \
        --weight_decay 0.0 \
        --max_grad_norm 1.0 \
        --max_train_steps 300000 \
        --eval_every 2000 \
        --save_every 300000 \
        --save_dir "$OUT_DIR/checkpoints/${name}" \
        --seed "$seed" \
        --gpu_id "$gpu" \
        --run_name "$name" \
        > "$log" 2>&1; then
        # the training code also saves the untrained model at step 0; keep only the final checkpoint
        rm -f "$OUT_DIR/checkpoints/${name}/checkpoint_0.pth"
        touch "$OUT_DIR/logs/${name}.done"
        echo "[seed $seed] finished"
    else
        echo "[seed $seed] FAILED, last lines of $log:" >&2
        tail -n 5 "$log" >&2
        return 1
    fi
}

# One worker per (GPU, slot); worker k runs every n-th seed so seeds spread evenly across GPUs.
n_workers=$(( ${#GPU_LIST[@]} * JOBS_PER_GPU ))
pids=()
for (( w = 0; w < n_workers; w++ )); do
    gpu=${GPU_LIST[$(( w % ${#GPU_LIST[@]} ))]}
    (
        status=0
        for (( i = w; i < ${#SEED_LIST[@]}; i += n_workers )); do
            train_one_seed "${SEED_LIST[$i]}" "$gpu" || status=1
        done
        exit $status
    ) &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done

n_done=$(find "$OUT_DIR/logs" -name '*.done' | wc -l)
echo "Finished: $n_done / ${#SEED_LIST[@]} seeds. Logs: $OUT_DIR/logs, checkpoints: $OUT_DIR/checkpoints"
exit $failed
