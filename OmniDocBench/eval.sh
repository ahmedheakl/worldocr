source ~/.bashrc
conda activate .worldocr
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=2,3
export HF_HOME=/l/users/ahmed.heakl
OUTROOT="../data/predictions_mega"
INPUT_DIR="../data/omnidocbench_output_mega/images"

# 1. SmolDocling
NAME=smoldocling
PYTHONPATH=.. python tools/model_infer/granite_docling.py \
    --model_path "ds4sd/SmolDocling-256M-preview" \
    --input_dir $INPUT_DIR \
    --output_dir $OUTROOT/$NAME 

# 2. GraniteDocling
NAME=granitedocling
PYTHONPATH=.. python tools/model_infer/granite_docling.py \
    --model_path "ibm-granite/granite-docling-258M" \
    --input_dir $INPUT_DIR \
    --output_dir $OUTROOT/$NAME


# 3. NanosetsOCR
NAME=nanosetsocr
LOG_FILE="vllm_$(date +'%Y%m%d_%H%M%S').log"
vllm serve nanonets/Nanonets-OCR-s \
    --gpu-memory-utilization 0.8 \
    --max-model-len 15000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 > $LOG_FILE 2>&1 &
VLLM_PID=$!
echo "Waiting for vLLM server to start..."
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 1
done
PYTHONPATH=.. python tools/model_infer/nanosets_vllm.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTROOT/$NAME"

echo "Stopping vLLM..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null || true
echo "vLLM stopped."