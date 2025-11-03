source ~/.bashrc
conda activate .worldocr
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=2,3

set -a
source .env
set +a

OUTROOT="../data/predictions_mega"
INPUT_DIR="../data/omnidocbench_output_mega/images"
FILE_PATH="../data/omnidocbench_output_mega/omnidocbench.json"

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
LOG_FILE="vllm_nanosets_$(date +'%Y%m%d_%H%M%S').log"
vllm serve nanonets/Nanonets-OCR-s \
    --gpu-memory-utilization 0.8 \
    --max-model-len 15000 \
    --tensor-parallel-size 1 > $LOG_FILE 2>&1 &
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

# 4. NanosetsOCR-2-3b
NAME=nanosetsocr2
LOG_FILE="vllm_2_3b_$(date +'%Y%m%d_%H%M%S').log"
MODEL_PATH="nanonets/Nanonets-OCR2-3B"
vllm serve $MODEL_PATH \
    --gpu-memory-utilization 0.8 \
    --max-model-len 15000 \
    --tensor-parallel-size 1  > $LOG_FILE 2>&1 &
VLLM_PID=$!
echo "Waiting for vLLM server to start..."
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 1
done
PYTHONPATH=.. python tools/model_infer/nanosets_vllm.py \
    --model_path "$MODEL_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTROOT/$NAME"   
echo "Stopping vLLM..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null || true
echo "vLLM stopped."

# 5. Qwen2.5-VL-3B-Instruct
NAME=qwen25vl3b
PYTHONPATH=.. python tools/model_infer/qwen2vl_vllm.py \
    --model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --input_dir $INPUT_DIR \
    --output_dir $OUTROOT/$NAME

# 6. Qwen3-VL-2B-Instruct
NAME=qwen3vl2b
PYTHONPATH=.. python tools/model_infer/qwen2vl_vllm.py \
    --model_path "Qwen/Qwen3-VL-2B-Instruct" \
    --input_dir $INPUT_DIR \
    --output_dir $OUTROOT/$NAME

# 7. MinerU2.5
NAME=miner25
PYTHONPATH=.. python tools/model_infer/mineru_md.py \
    --model_path "opendatalab/MinerU2.5-2509-1.2B" \
    --input_dir $INPUT_DIR \
    --output_dir $OUTROOT/$NAME


# 8. Dolphin
NAME=dolphin
PYTHONPATH=.. python tools/model_infer/Dolphin_img2md.py \
    --model_path "./hf_model" \
    --input_path $INPUT_DIR \
    --save_dir $OUTROOT/$NAME

# 9. DotsOCR
NAME=dotsocr
python dotsocr_md.py \
    --filepath $FILE_PATH \
    --output_dir $OUTROOT/$NAME

# 10. gpt4o
NAME=gpt4o
PYTHONPATH=.. python tools/model_infer/gpt_4o_inf.py \
    --model_path "gpt-4o" \
    --input_dir $INPUT_DIR \
    --output_dir $OUTROOT/$NAME

# 11. DeepseekOCR
NAME=deepseekocr
python deepseekocr_vllm.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTROOT/$NAME


# 12. chandra
NAME=chandra
LOG_FILE="vllm_chandra_$(date +'%Y%m%d_%H%M%S').log"
NUM_VISIBLE_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
vllm serve datalab-to/chandra \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --served-model-name chandra \
    --tensor-parallel-size $NUM_VISIBLE_GPUS > $LOG_FILE 2>&1 &

VLLM_PID=$!
echo "Waiting for vLLM server to start..."
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    sleep 1
done
echo "vLLM server started."
python tools/model_infer/chandra_md.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTROOT/$NAME" 
echo "Stopping vLLM..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null || true
echo "vLLM stopped."