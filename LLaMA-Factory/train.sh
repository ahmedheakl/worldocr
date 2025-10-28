source ~/.bashrc
conda activate .worldocr
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/l/users/ahmed.heakl

export NNODES=1
export NPROC_PER_NODE=4

set -a
if [ -f .env ]; then
    source .env
fi
set +a

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen25vl_v1.yaml