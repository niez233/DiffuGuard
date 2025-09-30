# python /opt/tiger/dllm-safety/run_jailbreakbench/evaluation.py 
# python /opt/tiger/dllm-safety/LLaDA/evaluation.py
PY_TOP=$(python -c "import site; print(site.getusersitepackages())")
PYTHONPATH="${PY_TOP}:${PYTHONPATH}" PYTHONNOUSERSITE=0 \
python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-8B-Instruct" \
  --attack_method DIJA \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_DIJA_off_reminder_8b.json \
  --sp_mode off \
  --defense_method self-reminder \
  --debug_print
# --defense_method self-reminder

python models/jailbreakbench_llada.py --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-8B-Instruct" \
  --attack_method PAD \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_PAD_off_reminder_8b.json \
  --sp_mode off \
  --defense_method self-reminder \
  --debug_print

python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-8B-Instruct" \
  --attack_method PAD \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_PAD_off_8b.json \
  --sp_mode off \
  --debug_print

python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-8B-Instruct" \
  --attack_method PAD \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_PAD_off_8b.json \
  --sp_mode off \
  --debug_print

python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-8B-Instruct" \
  --attack_method PAD \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_PAD_adaptive_step_8b.json \
  --remasking adaptive_step \
  --debug_print --fill_all_masks
# --remasking adaptive_step \
#  --defense_method self-reminder \
python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-1.5" \
  --attack_method PAD \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_PAD_off_1.5.json \
  --sp_mode off \
  --debug_print

python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-1.5" \
  --attack_method DIJA \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_DIJA_off_1.5.json \
  --sp_mode off \
  --debug_print

python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-1.5" \
  --attack_method PAD \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_PAD_adaptive_step_hidden_1.5.json \
  --remasking adaptive_step \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 --debug_print --fill_all_masks

python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-1.5" \
  --attack_method PAD \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_PAD_reminder_adaptive_step_hidden_1.5.json \
  --remasking adaptive_step \
  --defense_method self-reminder \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 --debug_print --fill_all_masks

python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-1.5" \
  --attack_method DIJA \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_DIJA_adaptive_step_hidden_1.5.json \
  --remasking adaptive_step \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 --debug_print --fill_all_masks
#   --defense_method self-reminder \
python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-1.5" \
  --attack_method DIJA \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_adapt_hidden_1.5.json \
  --sp_mode hidden --sp_threshold 0.2 \
  --refinement_steps 8 --remask_ratio 0.9 --debug_print --fill_all_masks

python models/jailbreakbench_llada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/LLaDA-8B-Instruct" \
  --attack_method DIJA \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_dija_hidden_8b.json \
  --sp_mode hidden --sp_threshold 0.2 \
  --refinement_steps 8 --remask_ratio 0.9 --debug_print  --fill_all_masks

TRANSFORMERS_NO_TORCHVISION=1 \
python models/jailbreakbench_mmada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/MMaDA-8B-MixCoT" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --attack_method PAD \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_PAD_reminder_off_mmada.json \
  --defense_method self-reminder \
  --sp_mode off \
  --debug_print

python models/jailbreakbench_mmada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/MMaDA-8B-MixCoT" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --attack_method DIJA \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_DIJA_off_mmada.json \
  --sp_mode off \
  --debug_print

python models/jailbreakbench_mmada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/MMaDA-8B-MixCoT" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --attack_method PAD \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_pad_adaptive_step_hidden_mmada.json \
  --remasking adaptive_step \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 --debug_print --fill_all_masks

python models/jailbreakbench_mmada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/MMaDA-8B-MixCoT" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --attack_method PAD \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_pad_reminder_adaptive_step_hidden_mmada.json \
  --remasking adaptive_step \
  --defense_method self-reminder \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 --debug_print --fill_all_masks

python models/jailbreakbench_mmada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/MMaDA-8B-MixCoT" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --attack_method DIJA \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_DIJA_adaptive_step_hidden_mmada.json \
  --remasking adaptive_step \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 --debug_print --fill_all_masks

python models/jailbreakbench_mmada.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/MMaDA-8B-MixCoT" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --attack_method DIJA \
  --output_json /opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_DIJA_reminder_adaptive_step_hidden_mmada.json \
  --remasking adaptive_step \
  --defense_method self-reminder \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 --debug_print --fill_all_masks

python models/jailbreakbench_dream.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/Dream-v0-Instruct-7B" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json "/opt/tiger/dllm-safety/run_jailbreakbench/out_pad_hidden_dream.json" \
  --attack_method pad \
  --gen_length 128 --steps 64 \
  --mask_counts 36 \
  --temperature 0.5 \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 \
  --debug_print

python models/jailbreakbench_dream.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/Dream-v0-Instruct-7B" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json "/opt/tiger/dllm-safety/run_jailbreakbench/out_DIJA_off_dream.json" \
  --attack_method DIJA \
  --gen_length 128 --steps 64 \
  --mask_counts 36 \
  --temperature 0.5 \
  --sp_mode off \
  --debug_print

python models/jailbreakbench_dream.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/Dream-v0-Instruct-7B" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json "/opt/tiger/dllm-safety/run_jailbreakbench/out_pad_reminder_adaptive_step_hidden_dream.json" \
  --attack_method pad \
  --gen_length 128 --steps 64 \
  --mask_counts 36 \
  --temperature 0.5 \
  --remasking adaptive_step \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 \
  --defense_method self-reminder \
  --debug_print

python models/jailbreakbench_dream.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/Dream-v0-Instruct-7B" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json "/opt/tiger/dllm-safety/run_jailbreakbench/out_pad_adaptive_step_hidden_dream.json" \
  --attack_method pad \
  --gen_length 128 --steps 64 \
  --mask_counts 36 \
  --temperature 0.5 \
  --remasking adaptive_step \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 \
  --debug_print


python models/jailbreakbench_dream.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/Dream-v0-Instruct-7B" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json "/opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_pad_reminder_off_dream.json" \
  --attack_method pad \
  --gen_length 128 --steps 64 \
  --mask_counts 36 \
  --temperature 0.5 \
  --defense_method self-reminder \
  --sp_mode off \
  --debug_print

python models/jailbreakbench_dream.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/Dream-v0-Instruct-7B" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json "/opt/tiger/dllm-safety/run_jailbreakbench/attack_results/out_DIJA_adaptive_step_hidden_dream.json" \
  --attack_method DIJA \
  --steps 64 \
  --gen_length 128 \
  --mask_counts 36 \
  --temperature 0.5 \
  --remasking adaptive_step \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 \
  --debug_print

python models/jailbreakbench_dream.py \
  --model_path "/opt/tiger/dllm-safety/hf_models/Dream-v0-Instruct-7B" \
  --attack_prompt "/opt/tiger/dllm-safety/dija_advbench.json" \
  --output_json "/opt/tiger /dllm-safety/run_jailbreakbench/attack_results/out_DIJA_reminder_adaptive_step_hidden_dream.json" \
  --attack_method DIJA \
  --steps 64 \
  --gen_length 128 \
  --mask_counts 36 \
  --temperature 0.5 \
  --remasking adaptive_step \
  --defense_method self-reminder \
  --sp_mode hidden --sp_threshold 0.1 \
  --refinement_steps 8 --remask_ratio 0.9 \
  --debug_print