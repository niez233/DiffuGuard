set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7

export OPENAI_API_KEY='your_api_key'
export OPENAI_BASE_URL='your_base_url'

INPUT_DIR="results"
LOG_DIR="eval_logs"

shopt -s nullglob
json_files=("$INPUT_DIR"/*.json)

if [ ${#json_files[@]} -eq 0 ]; then
  echo "No json files found in $INPUT_DIR"
  exit 0
fi

for f in "${json_files[@]}"; do
  base="$(basename "$f" .json)"

  asr_logfile="$LOG_DIR/${base}.log"
  echo "Evaluating: $f -> $asr_logfile"
  python evaluate.py "$f" > "$asr_logfile" 2>&1 || echo "Evaluation failed for $f (see $asr_logfile)"

  ppl_logfile="$LOG_DIR/${base}-ppl.log"
  echo "Calculating perplexity: $f -> $ppl_logfile"
  python perplexity.py --model_id "meta-llama/Llama-2-7b-hf" --json_file "$f" > "$ppl_logfile" 2>&1 || echo "Perplexity calculation failed for $f"
done