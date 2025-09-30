# models/jailbreakbench_mmada.py
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import torch
import argparse
import logging
import subprocess
from tqdm import tqdm
from transformers import AutoTokenizer

# ---- Project path setup ----
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
try:
    from MMaDA.models import MMadaModelLM
except Exception as e:
    raise RuntimeError(f"Failed to import MMadaModelLM: {e}")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utility.generate_function_mmada import generate_mmada
from defense_utils import Defender  # Add the missing import

# ===== Constants =====
DEFAULT_STEPS = 64
DEFAULT_GEN_LENGTH = 128            # For API symmetry; MMaDA typically doesn't append a tail mask
DEFAULT_MASK_ID = 126336
DEFAULT_MASK_COUNTS = 128
DEFAULT_TEMPERATURE = 0.5
DEFAULT_REF_TAIL_LEN = 128          # NEW: reference tail length used only for detection

MASK_TOKEN = "<|mdm_mask|>"
START_TOKEN = "<startoftext>"
END_TOKEN = "<endoftext>"
SPECIAL_TOKEN_PATTERN = r"<mask:(\d+)>"

COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses using MMaDA (in-place fill) with advanced options")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--attack_prompt", type=str, required=True,
                        help="Path to JSON; supports keys such as 'vanilla prompt'/'refined prompt' or 'goal'/'refined_goal'")
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen_length", type=int, default=DEFAULT_GEN_LENGTH,
                        help="Kept for API symmetry; MMaDA usually doesn't use tail masks")
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--mask_counts", type=int, default=DEFAULT_MASK_COUNTS,
                        help="If neither <mask:...> nor a mask token is found, optionally append tail masks")

    parser.add_argument("--attack_method", type=str, default="zeroshot",
                        choices=["zeroshot", "DIJA","PAD", "other"])
    parser.add_argument("--defense_method", type=str, default=None)

    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random", "rate", "adaptive", "adaptive_step"],
                        help="Strategy for selecting tokens to remask between steps")
    parser.add_argument("--random_rate", type=float, default=0.0,
                        help="Ratio for random remasking when remasking='random'")
    parser.add_argument("--injection_step", type=int, default=None,
                        help="For PAD attacks: which step to inject anchors; None = internal default")
    parser.add_argument("--alpha0", type=float, default=0.3,
                        help="Initial guidance/denoising strength for MMaDA style schedulers")

    parser.add_argument("--sp_mode", type=str, default="off", choices=["off", "logit", "hidden"],
                        help="Self-protection mode: off/logit/hidden")
    parser.add_argument("--sp_threshold", type=float, default=0.35,
                        help="Threshold for SP decisions (e.g., cosine distance or logit-based scores)")
    parser.add_argument("--refinement_steps", type=int, default=8,
                        help="Steps for self-correction/refinement if enabled")
    parser.add_argument("--remask_ratio", type=float, default=0.9,
                        help="Fraction of positions to remask under remasking='rate'")
    parser.add_argument("--suppression_value", type=float, default=1e6,
                        help="Large penalty applied to suppressed tokens during decoding")
    parser.add_argument("--fill_all_masks", action="store_true",
                        help="MMaDA fills the whole sequence in-place; this flag is forwarded for API compatibility")
    parser.add_argument("--debug_print", action="store_true")

    # Keep the same switches as LLaDA (semantics map to single-block behavior in MMaDA)
    parser.add_argument("--correct_only_first_block", dest="correct_only_first_block", action="store_true",
                        help="Only apply self-correction on the first (virtual) block for compatibility")
    parser.add_argument("--no_correct_only_first_block", dest="correct_only_first_block", action="store_false")
    parser.set_defaults(correct_only_first_block=True)

    # ---- Auto GPU selection ----
    parser.add_argument("--auto_pick_gpu", dest="auto_pick_gpu", action="store_true",
                        help="Pick the GPU with the most free memory automatically")
    parser.add_argument("--no_auto_pick_gpu", dest="auto_pick_gpu", action="store_false")
    parser.set_defaults(auto_pick_gpu=True)

    # ---- NEW: reference tail length for detection only (does not affect generation) ----
    parser.add_argument("--ref_tail_len", type=int, default=DEFAULT_REF_TAIL_LEN,
                        help="Length of the reference tail masks used ONLY for detection.")

    return parser.parse_args()


# ---------- GPU selection ----------
def _pick_gpu_by_torch() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        best_i, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free, _total = torch.cuda.mem_get_info()
            if free > best_free:
                best_free, best_i = free, i
        return best_i
    except Exception:
        return None


def _pick_gpu_by_nvidia_smi() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, text=True
        )
        frees = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        if not frees:
            return None
        return max(range(len(frees)), key=lambda i: frees[i])
    except Exception:
        return None


def pick_best_gpu_index() -> int | None:
    idx = _pick_gpu_by_torch()
    return idx if idx is not None else _pick_gpu_by_nvidia_smi()


# ---------- Expand <mask:x> in user text / optional default tail ----------
def process_user_text(user_text: str, mask_counts: int) -> str:
    def repl(m):
        n = int(m.group(1))
        return MASK_TOKEN * max(n, 0)

    processed = re.sub(SPECIAL_TOKEN_PATTERN, repl, user_text)

    # MMaDA typically does not append a tail; for backward compatibility, if no mask in text
    # and mask_counts > 0, append a tail segment at the end.
    if (MASK_TOKEN not in processed) and mask_counts:
        processed = processed + START_TOKEN + (MASK_TOKEN * mask_counts) + END_TOKEN

    return processed


# ---------- Build chat prompt ----------
def build_chat_prompt(tokenizer, user_text_processed: str, is_chat_model: bool, system_prompt: str | None = None) -> str:
    if is_chat_model:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text_processed})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if system_prompt:
            return f"{system_prompt}\n\n{user_text_processed}"
        return user_text_processed


def get_tokenized_input(prompt: str, tokenizer, device: torch.device):
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)


# ---------- Hidden baseline (for sp_mode=hidden) ----------
def compute_baseline_hidden(
    vanilla_text: str,
    tokenizer,
    model,
    is_chat_model: bool,
    system_prompt: str | None,
    *,
    debug_print: bool = False,
) -> torch.Tensor | None:
    """
    Used only for your existing sp_mode=hidden branch; the new "tail-only detection" is independent of this logic.
    """
    try:
        baseline_user = process_user_text(vanilla_text, mask_counts=0)
        baseline_prompt = build_chat_prompt(tokenizer, baseline_user, is_chat_model, system_prompt)
        baseline_ids, _ = get_tokenized_input(baseline_prompt, tokenizer, model.device)
        out = model(baseline_ids, output_hidden_states=True, return_dict=True)
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            h_last = out.hidden_states[-1]
            baseline_hidden = h_last.mean(dim=1).squeeze(0).detach()
            if debug_print:
                logging.info(f"[Baseline] hidden OK, dim={baseline_hidden.numel()}")
            return baseline_hidden
        if debug_print:
            logging.info("[Baseline] No hidden_states; fallback to logits-only SP.")
        return None
    except Exception as e:
        if debug_print:
            logging.info(f"[Baseline] failed: {e}; fallback to logits-only SP.")
        return None


# ======== NEW: reference-tail utilities used ONLY for detection (no impact on generation) ========

def attach_reference_tail(user_text: str, tail_counts: int) -> str:
    """Always append a reference tail at the end (regardless of existing <mask:n>); used only for detection."""
    if tail_counts <= 0:
        return user_text
    tail = START_TOKEN + (MASK_TOKEN * tail_counts) + END_TOKEN
    return f"{user_text}{tail}"


def build_detection_prompt(tokenizer, user_text: str, is_chat_model: bool,
                           system_prompt: str | None, tail_counts: int) -> str:
    """
    Detection prompt pipeline:
    1) Expand embedded <mask:n> inside the text (do NOT append default tail);
    2) Force-append a reference tail of the same length;
    3) Wrap with chat template.
    """
    processed = process_user_text(user_text, mask_counts=0)  # expand embedded <mask:n>, no default tail
    user_with_tail = attach_reference_tail(processed, tail_counts)
    return build_chat_prompt(tokenizer, user_with_tail, is_chat_model, system_prompt)


@torch.no_grad()
def _find_tail_span(tok: torch.Tensor, mask_id: int, tail_len: int) -> tuple[int, int]:
    """
    In token sequence `tok`, find the span [start, end) that covers the last `tail_len` tokens equal to mask_id.
    Assumes the force-appended reference tail is a contiguous sequence of <|mdm_mask|> * tail_len.
    """
    mask_pos = (tok == mask_id).nonzero(as_tuple=False).squeeze(1)
    assert mask_pos.numel() >= tail_len, f"Reference tail not found or too short: have {mask_pos.numel()}, need {tail_len}"
    last = mask_pos[-tail_len:]                     # take the last tail_len mask positions
    start = int(last.min().item())
    end   = int(last.max().item()) + 1
    return start, end


@torch.no_grad()
def first_step_tail_mean_hidden(model, tokenizer, prompt: str, mask_id: int, tail_len: int) -> torch.Tensor:
    """
    Run one forward pass; take the segment corresponding to the last `tail_len` masks (the reference tail)
    and compute the mean of the last-layer hidden states over that segment as a proxy for the "first step".
    For dLLMs, the last-layer hidden states from the first pass approximate step-0 states.
    Returns shape [H].
    """
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)  # [1, L]
    out = model(ids, output_hidden_states=True, return_dict=True)
    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; cannot compute hidden-based SP.")
    h_last = out.hidden_states[-1]                      # [1, L, H]
    tok = ids[0]
    tail_start, tail_end = _find_tail_span(tok, mask_id, tail_len)
    mu = h_last[:, tail_start:tail_end, :].mean(dim=1).squeeze(0).detach()  # [H]
    return mu


def cosine_distance_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32); b = b.to(torch.float32)
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float(1.0 - torch.dot(a, b).item())


# ---------- Generation ----------
def generate_response(vanilla_prompt: str, prompt: str, tokenizer, model, args, baseline_hidden=None, system_prompt: str | None = None) -> str:
    input_ids, attention_mask = get_tokenized_input(prompt, tokenizer, model.device)
    vanilla_ids, _ = get_tokenized_input(vanilla_prompt, tokenizer, model.device)
    # === Compute token length of system/self-reminder and build a protection mask ===
    protected_index = torch.zeros_like(input_ids, dtype=torch.bool, device=model.device)
    if system_prompt:
        # Tokenize the system-only template (no generation prompt) to get its token length
        sys_only_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to(model.device)
        sys_len = sys_only_ids.shape[1]
        sys_len = min(sys_len, input_ids.shape[1])  # avoid out-of-bounds
        protected_index[:, :sys_len] = True
    # Align outputs: skip tokens that match the vanilla prefix
    matching_count = min(
        sum(a.item() == b.item() for a, b in zip(vanilla_ids[0], input_ids[0])),
        len(vanilla_ids[0]),
    )

    output_ids = generate_mmada(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model=model,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=0,                       # MMaDA has no blocks; kept for API symmetry
        temperature=args.temperature,
        mask_id=args.mask_id,

        # —— Advanced controls (aligned with LLaDA) ——
        tokenizer=tokenizer,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        random_rate=args.random_rate,
        injection_step=args.injection_step,
        alpha0=args.alpha0,
        sp_mode=args.sp_mode,
        sp_threshold=args.sp_threshold,
        refinement_steps=args.refinement_steps,
        remask_ratio=args.remask_ratio,
        suppression_value=args.suppression_value,
        correct_only_first_block=args.correct_only_first_block,
        fill_all_masks=args.fill_all_masks,   # MMaDA defaults to in-place fill across the whole sequence
        debug_print=args.debug_print,
        baseline_hidden=baseline_hidden,
        attack_method=("pad" if args.attack_method.lower() == "pad" else "none"),
        pad_anchors=["Step 1:", "Step 2:", "Step 3:"],
        pad_positions=None,
        pad_in_uncond=True,  # keep unconditional branch anchored with PAD when needed
        protected_index=protected_index,
    )

    response = tokenizer.batch_decode(output_ids[:, matching_count:], skip_special_tokens=True)[0]
    if args.attack_method == "DIJA":
        response = response.split("assistant\n")[0]
    return response


# ---------- Helpers: extract two prompt fields from JSON item ----------
def pick_two_prompt_fields(item: dict) -> tuple[str, str]:
    vanilla = (
        item.get("vanilla prompt")
        or item.get("goal")
        or item.get("Behavior")
        or item.get("behavior")
        or ""
    )
    refined = (
        item.get("refined prompt")
        or item.get("refined_goal")
        or item.get("Refined_behavior")
        or item.get("refined goal")
        or ""
    )
    return vanilla, refined


def should_use_refined(args, attack_prompt_path, refined_text):
    path_flag = ("refine" in os.path.basename(attack_prompt_path).lower())
    if args.attack_method.lower() == "dija" or path_flag:
        return bool(refined_text.strip())
    return False


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args()

    # ---- Auto-pick GPU ----
    if torch.cuda.is_available():
        if args.auto_pick_gpu:
            best_idx = pick_best_gpu_index()
            if best_idx is not None:
                try:
                    torch.cuda.set_device(best_idx)
                except Exception:
                    pass
                device = torch.device(f"cuda:{best_idx}")
                try:
                    with torch.cuda.device(device):
                        free, total = torch.cuda.mem_get_info()
                    logging.info(f"[GPU] Auto-picked cuda:{best_idx} "
                                 f"(free {free/1024/1024/1024:.1f} GB / total {total/1024/1024/1024:.1f} GB)")
                except Exception:
                    logging.info(f"[GPU] Auto-picked cuda:{best_idx}")
            else:
                device = torch.device("cuda")
                logging.info("[GPU] Auto-pick failed, fallback to default CUDA device.")
        else:
            device = torch.device("cuda")
            logging.info("[GPU] Auto-pick disabled, using default CUDA device.")
    else:
        device = torch.device("cpu")
        logging.info("[GPU] CUDA not available, using CPU.")

    # ---- Defender ----
    defender = Defender(args.defense_method) if args.defense_method and args.defense_method.lower() != "none" else None

    # ---- Chat-model detection (MMaDA MixCoT typically uses chat templates) ----
    is_chat_model = ("mixcot" in args.model_path.lower()) or ("mmada" in args.model_path.lower())

    # ---- Model & tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = MMadaModelLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    # ---- Load data ----
    with open(args.attack_prompt, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    with torch.no_grad():
        for item in tqdm(data, desc="Processing data", total=len(data)):
            vanilla_text, refined_text = pick_two_prompt_fields(item)
            use_refined = should_use_refined(args, args.attack_prompt, refined_text)
            chosen_text = refined_text if (use_refined and refined_text.strip()) else vanilla_text

            # Expand <mask:x> on the user-side text only / optionally append tail masks (for generation)
            chosen_user = process_user_text(chosen_text, args.mask_counts)
            vanilla_user = process_user_text(vanilla_text, args.mask_counts)

            # Inject self-reminder into system; other defenses operate on user text
            system_for_both = None
            if defender and getattr(defender, "defense", None) == "self-reminder":
                ret = defender.handler(chosen_text)
                if isinstance(ret, tuple) and len(ret) >= 1:
                    system_for_both = ret[0]
                elif isinstance(ret, str):
                    system_for_both = ret
            elif defender:
                chosen_user = defender.handler(chosen_user)
                vanilla_user = defender.handler(vanilla_user)

            # Build final prompts (for generation)
            prompt = build_chat_prompt(tokenizer, chosen_user, is_chat_model, system_for_both)
            vanilla_prompt = build_chat_prompt(tokenizer, vanilla_user, is_chat_model, system_for_both)

            # ========= NEW: dual forward passes for template jailbreak detection (tail-only) =========
            sp_hid_tail = None
            template_attack = None
            ref_tail_len = max(int(getattr(args, "ref_tail_len", DEFAULT_REF_TAIL_LEN)), 0)

            if ref_tail_len > 0:
                det_vanilla_prompt = build_detection_prompt(
                    tokenizer, vanilla_text, is_chat_model, system_for_both, ref_tail_len
                )
                det_refined_prompt = build_detection_prompt(
                    tokenizer, chosen_text,  is_chat_model, system_for_both, ref_tail_len
                )
                try:
                    van_mu = first_step_tail_mean_hidden(
                        model, tokenizer, det_vanilla_prompt, args.mask_id, ref_tail_len
                    )  # [H]
                    ref_mu = first_step_tail_mean_hidden(
                        model, tokenizer, det_refined_prompt, args.mask_id, ref_tail_len
                    )  # [H]
                    sp_hid_tail = cosine_distance_vec(van_mu, ref_mu)  # 1 - cos
                    template_attack = (sp_hid_tail >= args.sp_threshold)
                except Exception as e:
                    logging.info(f"[TailDetection] skipped due to: {e}")

            # Hidden-baseline (compute only if needed; independent from tail-only detection above)
            baseline_hidden = None
            if args.sp_mode == "hidden":
                baseline_hidden = compute_baseline_hidden(
                    vanilla_text=vanilla_text,
                    tokenizer=tokenizer,
                    model=model,
                    is_chat_model=is_chat_model,
                    system_prompt=system_for_both,
                    debug_print=args.debug_print,
                )

            # Debug prompts if needed
            if os.getenv("DEBUG_SHOW_PROMPT", "0") == "1":
                logging.info(("PROMPT_PREVIEW=" + prompt[:1200]).replace("\n", "\\n"))
            if os.getenv("DEBUG_SHOW_TAIL", "0") == "1" and ref_tail_len > 0:
                logging.info(("TAIL_VAN=" + det_vanilla_prompt[:600]).replace("\n", "\\n"))
                logging.info(("TAIL_REF=" + det_refined_prompt[:600]).replace("\n", "\\n"))

            # Generate
            response = generate_response(vanilla_prompt, prompt, tokenizer, model, args, baseline_hidden=baseline_hidden, system_prompt=system_for_both)
            logging.info(f"{COLOR_BLUE}Response: {response}{COLOR_RESET}\n")

            # Record outputs
            rec = {
                "vanilla prompt": vanilla_text,
                "refined prompt": refined_text,
                "used_prompt_type": "refined" if (use_refined and refined_text.strip()) else "vanilla",
                "final_prompt": prompt,
                "response": response,
                # NEW: detection metrics
                "sp_hid_tail": sp_hid_tail,
                "template_attack": template_attack,
                "ref_tail_len": ref_tail_len,
            }
            for k in ("source", "category", "goal", "refined_goal", "Behavior", "Refined_behavior", "target"):
                if k in item:
                    rec[k] = item[k]
            results.append(rec)

    out_dir = os.path.dirname(args.output_json) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
