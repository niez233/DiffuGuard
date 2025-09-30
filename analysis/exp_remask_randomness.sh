export CUDA_VISIBLE_DEVICES=4,5,6,7

python generate.py \
    --model_name "LLaDA-8B" \
    --custom_cache_dir "GSAI-ML/LLaDA-8B-Instruct" \
    --input_path "datasets/wildjailbreak.json" \
    --random_rate 0.6 \
