# -*- coding: utf-8 -*-
import os
import re
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# ===================== Basics & Utilities =====================

def _ids_from_phrases(tokenizer, phrases: Iterable[str]) -> set:
    ids = set()
    for ph in phrases:
        toks = tokenizer(ph, add_special_tokens=False).input_ids
        for t in toks:
            ids.add(t)
    return ids


def build_cue_id_sets(tokenizer,
                      accept_cues: Optional[Iterable[str]] = None,
                      refuse_cues: Optional[Iterable[str]] = None) -> Tuple[set, set]:
    """Build token-id sets for acceptance/refusal cue phrases for logit-based self-checking."""
    if accept_cues is None:
        accept_cues = ["Sure", "Of course", "Here", "Step", "1.", "First", "Proceed", "Let's"]
    if refuse_cues is None:
        refuse_cues = ["Sorry", "I can't", "cannot", "won't", "illegal", "harmful", "not assist", "refuse"]
    return _ids_from_phrases(tokenizer, accept_cues), _ids_from_phrases(tokenizer, refuse_cues)


def safety_perplexity_from_logits(logits: torch.Tensor,
                                  positions_mask: torch.Tensor,
                                  accept_ids: set,
                                  refuse_ids: set) -> float:
    """Compute a simple conflict score between acceptance and refusal distributions.

    `positions_mask` selects which positions to include; we average conflict over those positions.
    The score is 4 * p_accept * p_refuse, which is maximized when both are high (undesirable ambiguity).
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        if len(accept_ids) == 0 or len(refuse_ids) == 0:
            return 0.0
        accept_p = probs[..., list(accept_ids)].sum(dim=-1)
        refuse_p = probs[..., list(refuse_ids)].sum(dim=-1)
        conflict = 4.0 * accept_p * refuse_p
        mask = positions_mask.to(conflict.dtype)
        denom = mask.sum().clamp_min(1.0)
        sp = (conflict * mask).sum() / denom
        return float(sp.item())


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine distance = 1 - cosine similarity."""
    a = a.detach().to(torch.float32)
    b = b.detach().to(torch.float32)
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float(1.0 - torch.dot(a, b).item())


def add_gumbel_noise(logits, temperature):
    """Add simple Gumbel-like noise controlled by `temperature` to encourage exploration."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Plan how many masked tokens to fill at each iteration, evenly over the remaining steps.

    Returns a [B, steps] tensor; we use column 0 per iteration and shift next iteration.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # [B,1]
    steps = max(int(steps), 1)
    base = mask_num // steps
    remainder = (mask_num % steps).squeeze(1)  # [B]
    num_transfer_tokens = base.expand(-1, steps).clone().to(torch.int64)  # [B,steps]
    if steps > 0:
        idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)  # [1,steps]
        bump = (idx < remainder.unsqueeze(1)).to(torch.int64)            # [B,steps]
        num_transfer_tokens += bump
    return num_transfer_tokens


SPECIAL_TOKEN_PATTERN = r"<mask:(\d+)>"


def _mask_token_str(tokenizer, mask_id: int) -> str:
    """Best-effort conversion of `mask_id` to string token; fallback to a generic placeholder."""
    try:
        t = tokenizer.convert_ids_to_tokens(mask_id)
        if isinstance(t, str) and len(t) > 0:
            return t
    except Exception:
        pass
    return "<|mask|>"


def expand_span_masks_like_mmdm(text: str,
                                mask_token: str,
                                mask_counts: int = 0,
                                add_tail_if_missing: bool = True) -> str:
    """Expand span masks such as `<mask:N>` to `mask_token * N`.

    Optionally append `mask_counts` mask tokens at the end if the text contains no mask and
    `add_tail_if_missing` is True.
    """
    def repl(m):
        n = max(int(m.group(1)), 0)
        return mask_token * n
    out = re.sub(SPECIAL_TOKEN_PATTERN, repl, str(text))
    if add_tail_if_missing and (mask_token not in out) and (mask_counts > 0):
        out = out + (mask_token * mask_counts)
    return out


# ===================== MMaDA Generation (Enhanced) =====================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,                        # token ids, shape [B, L]
    attention_mask=None,           # Use if provided; otherwise ignored
    steps=64,                      # Target to fill all masks in ~`steps` iterations
    gen_length=0,                  # MMaDA normally doesn't append a tail; kept for API compatibility
    block_length=0,                # MMaDA does not do block-wise fill; kept for API compatibility
    temperature=0.5,
    cfg_scale=0.0,
    remasking="low_confidence",    # ["low_confidence","random","rate","adaptive","adaptive_step"]
    mask_id=126336,
    random_rate: float = 0.0,
    injection_step: Optional[int] = None,
    alpha0: float = 0.3,
    sp_mode: str = "off",          # ["off","logit","hidden"]
    sp_threshold: float = 0.35,
    refinement_steps: int = 8,
    remask_ratio: float = 0.9,
    suppression_value: float = 1e6,
    correct_only_first_block: bool = True,   # Align with LLaDA API; MMaDA has a single "block"
    accept_cues: Optional[Iterable[str]] = None,
    refuse_cues: Optional[Iterable[str]] = None,
    baseline_hidden: Optional[torch.Tensor] = None,
    fill_all_masks: bool = True,
    debug_print: bool = False,
    attack_method: str = "none",                 # "none" | "pad"
    pad_anchors: Optional[Iterable[str]] = None, # e.g., ["Step 1:", "Step 2:", "Step 3:"]
    pad_positions: Optional[Iterable[int]] = None,
    pad_in_uncond: bool = True,
    protected_index: Optional[torch.Tensor] = None,  # NEW: protected positions (e.g., system/self-reminder)
):
    """
    Keep the same interface/strategies as the enhanced LLaDA script, adapted to typical MMaDA usage:
      - Fill in-place at `<mask>` positions within the input sequence; no extra tail by default.
      - No block segmentation; the entire span is treated as a single block for self-check & optional refinement.
    """
    device = next(model.parameters()).device

    # ---- Initialize x: MMaDA uses the original prompt directly; no tail by default ----
    x = prompt.clone().to(device)         # [B, L]
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    prompt_len = x.shape[1]

    # Optionally append a tail (kept for cross-API parity)
    if int(gen_length) > 0:
        tail_len = int(gen_length)
        tail = torch.full((x.size(0), tail_len), mask_id, dtype=torch.long, device=device)
        x = torch.cat([x, tail], dim=1)
        if attention_mask is not None:
            am_tail = torch.ones((attention_mask.size(0), tail_len),
                                dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, am_tail], dim=1)
        tail_start = prompt_len
        tail_end = prompt_len + tail_len
    else:
        tail_start = tail_end = x.shape[1]

    # Align with LLaDA semantics for `fill_all_masks`
    if (tail_end > tail_start) and (not fill_all_masks):
        # LLaDA semantics: restrict checking/writing to the tail region
        block_start, block_end = tail_start, tail_end
    else:
        # Fill across whole sequence; if no tail, restrict to min span covering all masks
        block_start, block_end = 0, x.shape[1]
        if (tail_end == tail_start):  # no tail: cover all mask positions
            mask_pos = (x == mask_id).nonzero(as_tuple=False)
            if mask_pos.numel() > 0:
                first = int(mask_pos[:, 1].min().item())
                last  = int(mask_pos[:, 1].max().item()) + 1
                block_start, block_end = first, last

    # Positions that are initially non-mask (used by CFG unconditional branch)
    prompt_index = (x != mask_id)

    # === LLaDA-identical PAD injection support ===
    uncond_prompt_index = prompt_index  # default: unconditional branch sees the same prompt region
    if attack_method.lower() == "pad":
        anchors = list(pad_anchors) if pad_anchors is not None else ["Step 1:", "Step 2:", "Step 3:"]

        # (0) Ensure there is a contiguous suffix after the prompt to write anchors into
        if x.shape[1] == prompt.shape[1]:
            gl = gen_length if int(gen_length) > 0 else 256
            pad = torch.full((x.size(0), gl), mask_id, dtype=torch.long, device=x.device)
            x = torch.cat([x, pad], dim=1)
            prompt_index = (x != mask_id)
            uncond_prompt_index = prompt_index
            # keep attention_mask in sync
            if attention_mask is not None:
                am_pad = torch.ones(attention_mask.size(0), gl, dtype=attention_mask.dtype, device=x.device)
                attention_mask = torch.cat([attention_mask, am_pad], dim=1)

        # (1) Suffix length after the (original) prompt
        after_prompt_len = x.shape[1] - prompt.shape[1]

        # (2) Compute injection positions (integer spacing; mirror LLaDA)
        if pad_positions is None:
            m = len(anchors)
            gap = max(after_prompt_len // (m + 1), 1)
            gap = gap // 1.5  # keep the same heuristic as LLaDA (note: integer division in original)
            offsets = [(i + 1) * gap for i in range(m)]
        else:
            offsets = list(pad_positions)

        # (3) Write anchors (no special tokens)
        for rel, text in zip(offsets, anchors):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            s = prompt.shape[1] + int(rel)
            e = s + len(ids)
            if 0 <= s < x.shape[1] and e <= x.shape[1]:
                x[:, s:e] = torch.tensor(ids, dtype=torch.long, device=x.device).unsqueeze(0)

        # (4) Whether the unconditional branch also "sees" anchors (same as LLaDA)
        if not pad_in_uncond:
            uncond_prompt_index = (x != mask_id)

    # >>> NEW: Align `protected_index` to the current `x` shape (applies to all attack methods)
    if protected_index is not None:
        pi = protected_index.to(device)
        if pi.dtype != torch.bool:
            pi = (pi != 0)
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)
        if pi.size(0) == 1 and x.size(0) > 1:
            pi = pi.expand(x.size(0), -1).clone()
        # Align sequence length: pad with False on the right; truncate if too long
        if pi.size(1) < x.size(1):
            pad_len = x.size(1) - pi.size(1)
            pad = torch.zeros((pi.size(0), pad_len), dtype=torch.bool, device=device)
            pi = torch.cat([pi, pad], dim=1)
        elif pi.size(1) > x.size(1):
            pi = pi[:, :x.size(1)]
        protected_index = pi  # [B, L]; newly appended tokens are unprotected by default

    # Single "block"
    block_start, block_end = 0, x.shape[1]
    steps_total = max(int(steps), 1)

    # Cue id sets (for logit-based self-check)
    accept_ids, refuse_ids = build_cue_id_sets(tokenizer, accept_cues, refuse_cues)

    first_step_block_hidden_mean = None
    use_hidden_detection = (sp_mode == "hidden") and (baseline_hidden is not None)
    warned_no_hidden = False

    last_logits = None

    # ========== Main loop: continue until there are no masks left ==========
    iter_id = 0
    while (x == mask_id).any():
        if iter_id >= 9999:  # hard stop to avoid pathological loops
            if debug_print:
                print("[MMaDA] Reached max safety iterations (9999). Breaking.", flush=True)
            break

        # Optional adversarial token injection at a specific iteration
        if injection_step is not None and iter_id == injection_step:
            if debug_print:
                print("[MMaDA] Injecting jailbreak tokens…", flush=True)
            injection_map = {0: "Sorry"}
            for relative_pos, text in injection_map.items():
                injection_ids = tokenizer(text, add_special_tokens=True).input_ids
                absolute_start_pos = min(max(0, relative_pos), x.shape[1] - len(injection_ids))
                absolute_end_pos = absolute_start_pos + len(injection_ids)
                x[:, absolute_start_pos:absolute_end_pos] = torch.tensor(
                    injection_ids, dtype=torch.long, device=x.device
                ).unsqueeze(0)

        mask_index = (x == mask_id)           # [B, L]
        if not mask_index.any():
            break

        # Plan how many tokens to fill this round: distribute remaining masks across remaining steps
        remaining_steps = max(steps_total - iter_id, 1)
        num_transfer_tokens_plan = get_num_transfer_tokens(mask_index, remaining_steps)  # [B, remaining_steps]
        num_transfer_tokens = num_transfer_tokens_plan[:, 0]  # take the first column for this iteration

        # ====== Forward pass ======
        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[uncond_prompt_index] = mask_id
            x_cat = torch.cat([x, un_x], dim=0)
            if attention_mask is not None:
                # duplicate attention mask for CFG
                am_cat = torch.cat([attention_mask, attention_mask], dim=0)
                out = model(x_cat, attention_mask=am_cat, output_hidden_states=use_hidden_detection, return_dict=True)
            else:
                out = model(x_cat, output_hidden_states=use_hidden_detection, return_dict=True)

            logits = out.logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

            hidden_states = None
            if use_hidden_detection and hasattr(out, "hidden_states") and out.hidden_states is not None:
                hidden_states = [hs[:1] for hs in out.hidden_states]
            elif use_hidden_detection and not warned_no_hidden:
                print("[Self-Detection] hidden_states unavailable; fallback to logits.", flush=True)
                warned_no_hidden = True
        else:
            if attention_mask is not None:
                out = model(x, attention_mask=attention_mask, output_hidden_states=use_hidden_detection, return_dict=True)
            else:
                out = model(x, output_hidden_states=use_hidden_detection, return_dict=True)

            logits = out.logits
            if use_hidden_detection and hasattr(out, "hidden_states") and out.hidden_states is not None:
                hidden_states = out.hidden_states
            else:
                hidden_states = None
                if use_hidden_detection and not warned_no_hidden:
                    print("[Self-Detection] hidden_states unavailable; fallback to logits.", flush=True)
                    warned_no_hidden = True

        last_logits = logits

        # Record the mean hidden state over the block at the first step (for hidden-mode detection)
        if (use_hidden_detection) and (iter_id == 0) and (hidden_states is not None) and (len(hidden_states) > 0):
            last_h = hidden_states[-1]
            h_block = last_h[:, block_start:block_end, :]
            first_step_block_hidden_mean = h_block.mean(dim=1).squeeze(0).detach()

        # Sampling & selection
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

        p = F.softmax(logits, dim=-1)
        model_confidence = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]
        R = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

        if remasking == "low_confidence":
            x0_p = model_confidence
        elif remasking == "random":
            x0_p = R
        elif remasking == "rate":
            x0_p = (1 - float(random_rate)) * model_confidence + float(random_rate) * R
        elif remasking == "adaptive":
            alpha = torch.clamp(torch.tensor(alpha0, device=x0.device, dtype=model_confidence.dtype), 0.0, 1.0)
            x0_p = (1 - alpha) * model_confidence + alpha * R
        elif remasking == "adaptive_step":
            frac = 1.0 - (iter_id / (steps_total - 1)) if steps_total > 1 else 1.0
            alpha = torch.clamp(torch.tensor(alpha0 * frac, device=x0.device, dtype=model_confidence.dtype), 0.0, 1.0)
            x0_p = (1 - alpha) * model_confidence + alpha * R
        else:
            raise NotImplementedError(remasking)

        # Only select top-k within currently masked positions; also exclude protected positions
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.tensor(-float("inf"), device=x0.device))
        if protected_index is not None:
            confidence = confidence.masked_fill(protected_index, -float("inf"))
        if not fill_all_masks:
            confidence[:, :block_start] = -float("inf")
            confidence[:, block_end:]   = -float("inf")

        transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for b in range(confidence.shape[0]):
            k_plan = int(num_transfer_tokens[b].item())
            if k_plan <= 0:
                continue
            eligible = (confidence[b] > -float("inf"))
            eligible_count = int(eligible.sum().item())
            if eligible_count == 0:
                continue
            k = min(k_plan, eligible_count)
            conf_b = torch.where(eligible, confidence[b], torch.tensor(-float("inf"), device=confidence.device))
            _, select_index = torch.topk(conf_b, k=k)
            transfer_index[b, select_index] = True
        x[transfer_index] = x0[transfer_index]

        iter_id += 1

    # ====== Self-check & optional refinement (treat entire span as one "block") ======
    unsafe_flag = False
    detection_score = 0.0

    if sp_mode != "off" and last_logits is not None:
        positions_mask = torch.zeros_like(x[:, :], dtype=torch.bool, device=x.device)
        positions_mask[:, block_start:block_end] = True
        sp_logit = safety_perplexity_from_logits(last_logits, positions_mask, accept_ids, refuse_ids)
        detection_score = sp_logit
        if sp_logit >= sp_threshold:
            unsafe_flag = True

        if use_hidden_detection and (first_step_block_hidden_mean is not None) and (baseline_hidden is not None):
            dist = cosine_distance(first_step_block_hidden_mean, baseline_hidden)
            detection_score = dist
            if dist >= sp_threshold:
                unsafe_flag = True

    if debug_print:
        print(f"[MMaDA] Safety score={detection_score:.3f}, unsafe={unsafe_flag}", flush=True)
        print(f"[MMaDA] block=({block_start}, {block_end}), tail=({tail_start}, {tail_end}), fill_all={fill_all_masks}", flush=True)

    if unsafe_flag:  # Single block, so `correct_only_first_block` is naturally satisfied
        if debug_print:
            print(f"[MMaDA] --> Refinement Phase (steps={refinement_steps}, remask_ratio={remask_ratio})", flush=True)

        # Allow re-masking only within [block_start, block_end) and not in protected positions (shared across batch)
        eligible_positions = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        eligible_positions[block_start:block_end] = True
        if protected_index is not None:
            prot_any = protected_index.any(dim=0)   # exclude if protected in any batch
            eligible_positions &= ~prot_any

        cand = torch.nonzero(eligible_positions, as_tuple=False).squeeze(1)
        num_to_remask = int(cand.numel() * float(remask_ratio))

        if num_to_remask > 0 and cand.numel() > 0:
            perm = cand[torch.randperm(cand.numel(), device=x.device)[:num_to_remask]]
            global_indices_to_remask = perm
            original_token_ids_at_remasked_pos = x[:, global_indices_to_remask].clone()
            x[:, global_indices_to_remask] = mask_id

            refinement_mask_index = (x[:, block_start:block_end] == mask_id)
            num_refine_transfer = get_num_transfer_tokens(refinement_mask_index, max(int(refinement_steps), 1))

            for r_step in range(max(int(refinement_steps), 1)):
                mask_index = (x == mask_id)

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[uncond_prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    if attention_mask is not None:
                        am_cat = torch.cat([attention_mask, attention_mask], dim=0)
                        out = model(x_, attention_mask=am_cat, return_dict=True)
                    else:
                        out = model(x_, return_dict=True)
                    logits = out.logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    if attention_mask is not None:
                        logits = model(x, attention_mask=attention_mask).logits
                    else:
                        logits = model(x).logits

                # Suppress original tokens at the re-masked positions to encourage replacement
                if torch.isfinite(torch.tensor(suppression_value)):
                    logits[
                        0,
                        global_indices_to_remask,
                        original_token_ids_at_remasked_pos[0]
                    ] -= suppression_value

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                # Restrict to the current block and exclude protected positions
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float("inf"), device=x0.device))
                if protected_index is not None:
                    confidence = confidence.masked_fill(protected_index, -float("inf"))

                refine_transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for b in range(confidence.shape[0]):
                    k_plan = int(num_refine_transfer[b, r_step].item())
                    if k_plan <= 0:
                        continue
                    eligible = (confidence[b] > -float("inf"))
                    eligible_count = int(eligible.sum().item())
                    if eligible_count == 0:
                        continue
                    k = min(k_plan, eligible_count)
                    conf_b = torch.where(eligible, confidence[b], torch.tensor(-float("inf"), device=confidence.device))
                    _, select_index = torch.topk(conf_b, k=k)
                    refine_transfer_index[b, select_index] = True

                x[refine_transfer_index] = x0[refine_transfer_index]

    return x


# ===================== Public Wrapper (API aligned with LLaDA) =====================

@torch.no_grad()
def generate_mmada(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    model,
    steps: int = 64,
    gen_length: int = 0,
    block_length: int = 0,
    temperature: float = 0.5,
    mask_id: int = 126336,
    *,
    tokenizer=None,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    random_rate: float = 0.0,
    injection_step: Optional[int] = None,
    alpha0: float = 0.3,
    sp_mode: str = "off",
    sp_threshold: float = 0.35,
    refinement_steps: int = 8,
    remask_ratio: float = 0.9,
    suppression_value: float = 1e6,
    correct_only_first_block: bool = True,
    fill_all_masks: bool = True,   # MMaDA default True (no tail by default)
    debug_print: bool = False,
    baseline_hidden: Optional[torch.Tensor] = None,
    attack_method: str = "none",
    pad_anchors: Optional[Iterable[str]] = None,
    pad_positions: Optional[Iterable[int]] = None,
    pad_in_uncond: bool = True,
    protected_index: Optional[torch.Tensor] = None,   # NEW: forwarded through
):
    """Thin wrapper matching LLaDA's signature while calling the MMaDA generator."""
    assert tokenizer is not None, "`generate_mmada` requires a tokenizer (for cues and optional injection/self-check)."
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=input_ids,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        random_rate=random_rate,
        injection_step=injection_step,
        alpha0=alpha0,
        sp_mode=sp_mode,
        sp_threshold=sp_threshold,
        refinement_steps=refinement_steps,
        remask_ratio=remask_ratio,
        suppression_value=suppression_value,
        correct_only_first_block=correct_only_first_block,
        baseline_hidden=baseline_hidden,
        fill_all_masks=fill_all_masks,
        debug_print=debug_print,
        attack_method=attack_method,
        pad_anchors=pad_anchors,
        pad_positions=pad_positions,
        pad_in_uncond=pad_in_uncond,
        protected_index=protected_index,
    )
