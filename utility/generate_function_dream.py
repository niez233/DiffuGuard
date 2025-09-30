# -*- coding: utf-8 -*-
# Dream 外层生成器（统一版）：
# - 自适应 Remask 控制（用于逐步落子）
# - Hidden Self-Detection(step-0 block hidden vs baseline）
# - Self-Correction（：随机重掩 + 原 token 抑制 + 逐步再填）
from __future__ import annotations
from typing import Optional, Iterable, Tuple, Dict
import math
import logging
import copy

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ------------------ 基础工具 ------------------

@torch.no_grad()
def add_gumbel_noise(logits: torch.Tensor, temperature: float):
    """按 Gumbel-max 的思路给 logits 加噪后再取 argmax；temperature=0 则不加噪。"""
    if temperature <= 0:
        return logits
    # 用 float64 降低低精度噪声的偏差（参考 MDM 讨论）
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** float(temperature)
    return logits.exp() / gumbel_noise

@torch.no_grad()
def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    给定当前 mask 分布（[B,L]），把“落子名额”平均分配到 steps 个 refinement 步。
    返回形状 [B, steps] 的整型张量。
    """
    steps = max(int(steps), 1)
    if mask_index.dim() != 2:
        raise ValueError(f"mask_index must be [B,L], got {mask_index.shape}")
    mask_num = mask_index.sum(dim=1)                    # [B]
    base = (mask_num // steps).unsqueeze(1)             # [B,1]
    remainder = (mask_num % steps)                      # [B]
    out = base.repeat(1, steps).to(torch.int64)         # [B,steps]
    if steps > 0:
        idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)  # [1,steps]
        bump = (idx < remainder.unsqueeze(1)).to(torch.int64)             # [B,steps]
        out += bump
    return out

@torch.no_grad()
def _clone_and_update_config(
    model,
    *,
    max_new_tokens: int,
    steps: int,
    temperature: float,
    top_p: Optional[float],
    mask_token_id: int,                     # 必传
    return_dict_in_generate: bool = True,
    output_history: Optional[bool] = None,
):
    """对齐并克隆 generation_config，兼容不同字段名（steps/sampling_steps/diffusion_steps）。"""
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        raise ValueError("model has no .generation_config; cannot prepare DreamGenerationConfig")
    cfg = cfg.clone() if hasattr(cfg, "clone") else copy.deepcopy(cfg)

    if hasattr(cfg, "max_new_tokens"): cfg.max_new_tokens = int(max_new_tokens)
    if hasattr(cfg, "temperature"):    cfg.temperature = float(temperature)
    if hasattr(cfg, "top_p") and top_p is not None: cfg.top_p = float(top_p)
    if hasattr(cfg, "return_dict_in_generate"): cfg.return_dict_in_generate = bool(return_dict_in_generate)
    if output_history is not None and hasattr(cfg, "output_history"): cfg.output_history = bool(output_history)

    # 步数字段兼容
    for k in ("steps", "sampling_steps", "diffusion_steps"):
        if hasattr(cfg, k):
            setattr(cfg, k, int(max(1, steps)))
            break

    # 关键：设置 mask_token_id
    if hasattr(cfg, "mask_token_id"):
        cfg.mask_token_id = int(mask_token_id)

    # pad/bos/eos 保底
    base = getattr(model, "generation_config", None)
    for k in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(cfg, k, None) is None and base is not None and getattr(base, k, None) is not None:
            setattr(cfg, k, getattr(base, k))

    return cfg

def _ids_from_phrases(tokenizer, phrases: Iterable[str]) -> set:
    ids = set()
    for ph in phrases:
        toks = tokenizer(ph, add_special_tokens=False).input_ids
        for t in toks:
            ids.add(int(t))
    return ids

def build_cue_id_sets(
    tokenizer,
    accept_cues: Optional[Iterable[str]] = None,
    refuse_cues: Optional[Iterable[str]] = None
) -> Tuple[set, set]:
    if accept_cues is None:
        accept_cues = ["Sure", "Of course", "Here", "Step", "1.", "First", "Proceed", "Let's"]
    if refuse_cues is None:
        refuse_cues = ["Sorry", "I can't", "cannot", "won't", "illegal", "harmful", "not assist", "refuse"]
    return _ids_from_phrases(tokenizer, accept_cues), _ids_from_phrases(tokenizer, refuse_cues)

# ------------------ 自适应 Remask 控制器（逐步落子） ------------------

class AdaptiveRemaskController:
    """
    logits_hook：
      - 缓存 prev_x（上一轮 token）
      - 默认不做 logits 版 SP；只做逐步落子控制

    tokens_hook：
      - 找到“刚刚从 MASK 变成 token”的位置（newly-filled）
      - score = (1-α)*置信度 + α*随机；保留 top-keep_k，其余回退成 <mask>，
        keep_k = round((1-α)*K)，K 为本步新填充的数量
    """
    def __init__(self, *, mask_id: int, block_start: int, block_end: int,
                 steps_per_block: int, alpha0: float,
                 accept_ids: set, refuse_ids: set,
                 enable_sp_logits: bool = False):
        self.mask_id = int(mask_id)
        self.s = int(block_start)
        self.e = int(block_end)
        self.S = int(max(steps_per_block, 1))
        self.alpha0 = float(alpha0)
        self.accept_ids = accept_ids
        self.refuse_ids = refuse_ids
        self.enable_sp_logits = bool(enable_sp_logits)

        self.prev_x: Optional[torch.Tensor] = None
        self.sp_step0: Optional[float] = None

    def _alpha_at(self, i: int) -> float:
        if self.S <= 1:
            return self.alpha0
        frac = 1.0 - float(i) / float(self.S - 1)
        return max(0.0, min(1.0, self.alpha0 * frac))

    def logits_hook(self, step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        self.prev_x = x.clone()
        # 若想启用 logits 版 SP，可在 step==0 时计算；此处默认关闭
        return logits

    def tokens_hook(self, step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.prev_x is None:
            return x
        try:
            newly = (x != self.mask_id) & (self.prev_x == self.mask_id)
            newly[:, :self.s] = False
            newly[:, self.e:] = False
            if not newly.any():
                self.prev_x = x.clone()
                return x

            B, T = x.shape
            probs = torch.softmax(logits, dim=-1)  # [B,T,V]
            x_ids = x.clone()
            x_ids[~newly] = 0  # 占位
            conf_all = torch.gather(probs, dim=-1, index=x_ids.unsqueeze(-1)).squeeze(-1)  # [B,T]
            conf = conf_all.masked_fill(~newly, 0.0)

            alpha = self._alpha_at(step)
            rand = torch.rand_like(conf)
            score = (1.0 - alpha) * conf + alpha * rand

            x_new = x.clone()
            for b in range(B):
                idx_b = torch.nonzero(newly[b], as_tuple=False).squeeze(1)
                K = idx_b.numel()
                if K <= 0:
                    continue
                keep_k = max(0, min(K, int(round((1.0 - alpha) * K))))
                if keep_k < K:
                    scores_b = score[b, idx_b]
                    keep_local = torch.topk(scores_b, k=keep_k).indices if keep_k > 0 else torch.empty(0, dtype=torch.long, device=x.device)
                    keep_idx = idx_b[keep_local] if keep_k > 0 else torch.empty(0, dtype=torch.long, device=x.device)
                    mask_set = set(idx_b.tolist()) - set(keep_idx.tolist())
                    if mask_set:
                        mask_idx = torch.tensor(sorted(mask_set), device=x.device, dtype=torch.long)
                        x_new[b, mask_idx] = self.mask_id

            self.prev_x = x_new.clone()
            return x_new
        except Exception:
            self.prev_x = x.clone()
            return x

# ------------------ 自纠正（统一给 PAD / 非 PAD 用） ------------------

@torch.no_grad()
def apply_self_correction(
    model,
    tokenizer,
    seq_ids: torch.Tensor,                  # [B, L_out]
    initial_mask_in_prompt: torch.Tensor,   # [B, L_prompt]
    steps: int,
    temperature: float,
    mask_id: int,
    refinement_steps: int,
    remask_ratio: float,
    suppression_value: float,
    correction_scope: str,                  # "block_all" | "span_all" | "masked_only"
    exclude_mask_positions: Optional[torch.Tensor] = None,  # [B, L_out] True=禁止重掩（用于保护 PAD 锚点）
    debug_print: bool = False,
) -> torch.Tensor:
    """
    自纠正：随机重掩 + 原token抑制 + 逐步再填（per-step top-k）。
    """
    device = seq_ids.device
    x = seq_ids.clone()
    B, L_out = x.shape

    # 统一 exclude_mask 到 [B, L_out]
    if exclude_mask_positions is not None:
        excl = exclude_mask_positions.to(device)
        if excl.shape != (B, L_out):
            excl2 = torch.zeros((B, L_out), dtype=torch.bool, device=device)
            bmin = min(B, excl.shape[0]); lmin = min(L_out, excl.shape[1])
            excl2[:bmin, :lmin] = excl[:bmin, :lmin]
            exclude_mask_positions = excl2
        else:
            exclude_mask_positions = excl

    # 全 1 的 attn（兼容 Dream 的 SDPA）
    attn = torch.ones_like(x, dtype=torch.bool, device=device)

    # span_all 边界（来自原 prompt 的初始 mask）
    span_bounds = []
    for b in range(B):
        m = initial_mask_in_prompt[b].nonzero(as_tuple=False).squeeze(1)
        if m.numel() > 0:
            s = int(m.min().item()); e = int(m.max().item()) + 1
        else:
            s, e = 0, L_out
        s = max(0, min(s, L_out)); e = max(s, min(e, L_out))
        span_bounds.append((s, e))

    # 对每个样本执行重掩与精炼
    for b in range(B):
        # 候选集合（按 correction_scope）
        if correction_scope == "block_all":
            cand = torch.arange(0, L_out, device=device)
        elif correction_scope == "span_all":
            s, e = span_bounds[b]; cand = torch.arange(s, e, device=device)
        else:  # "masked_only"
            e = min(L_out, initial_mask_in_prompt.shape[1])
            orig_mask = torch.zeros(L_out, dtype=torch.bool, device=device)
            orig_mask[:e] = initial_mask_in_prompt[b, :e].to(device)
            cand = orig_mask.nonzero(as_tuple=False).squeeze(1)

        # 排除保护位置（例如 PAD 锚点）
        if exclude_mask_positions is not None:
            keep_mask = ~exclude_mask_positions[b]
            cand = cand[keep_mask[cand]]

        if cand.numel() == 0:
            continue

        num_to_remask = int(max(1, round(remask_ratio * cand.numel())))
        pick = cand[torch.randperm(cand.numel(), device=device)[:num_to_remask]]

        # 记录原 token 并置回 mask
        orig_tokens = x[b, pick].clone()
        x[b, pick] = mask_id

        # 预分配每步落子数（基于当前剩余 mask）
        refine_mask = (x[b:b+1] == mask_id)  # [1, L_out]
        num_refine = get_num_transfer_tokens(refine_mask, max(int(refinement_steps), 1))  # [1, steps]

        for r in range(max(int(refinement_steps), 1)):
            out = model(x, attention_mask=attn, return_dict=True)
            logits = out.logits  # [B, L_out, V]

            # 强抑制原 token，避免回退
            if torch.isfinite(torch.tensor(suppression_value)):
                logits[b, pick, orig_tokens] -= float(suppression_value)

            logits_noised = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noised, dim=-1)

            p = F.softmax(logits, dim=-1)
            conf_all = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L_out]

            cur_mask = (x == mask_id)
            conf = torch.where(cur_mask, conf_all, torch.full_like(conf_all, -1e9))

            # 每步仅填 K 个（per-step top-k）
            k = int(num_refine[0, r].item())
            valid_idx = cur_mask[b].nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() > 0 and k > 0:
                k = min(k, valid_idx.numel())
                _, local = torch.topk(conf[b, valid_idx], k=k)
                choose = valid_idx[local]
                x[b, choose] = x0[b, choose]

            if debug_print and (r == 0 or r + 1 == refinement_steps):
                changed = (x[b, pick] != orig_tokens).float().mean().item()
                logging.info(f"  refine step {r+1}/{refinement_steps} | changed@pick={changed:.2%}")

            if not (x == mask_id).any():
                break

    return x

# ------------------ Dream 版 hidden 生成器（统一 PAD/非 PAD 自纠正） ------------------

@torch.no_grad()
def generate_dream_hidden(
    *,
    model,
    tokenizer,
    input_ids: torch.Tensor,                 # [B, L_in]
    attention_mask: Optional[torch.Tensor],  # [B, L_in] 或 None
    gen_length: int = 128,
    steps: int = 64,
    block_length: Optional[int] = None,
    temperature: float = 0.5,
    top_p: Optional[float] = 0.95,
    alpha0: float = 0.3,
    sp_threshold: float = 0.35,
    baseline_hidden: Optional[torch.Tensor] = None,
    attack_probe_hidden: Optional[torch.Tensor] = None,
    tau_hidden: float = 0.80,                # 占位（与旧接口兼容）
    refinement_steps: int = 8,
    remask_ratio: float = 0.9,
    suppression_value: float = 1e6,
    correct_only_first_block: bool = True,
    fill_all_masks: bool = False,
    mask_id: int = 151666,
    attack_method: str = "none",             # "none" / "pad"
    pad_anchors: Optional[Iterable[str]] = None,
    pad_positions: Optional[Iterable[int]] = None,
    protect_anchors: bool = True,
    correction_scope: str = "block_all",     # 与非 PAD 共用
    initial_mask_in_prompt: Optional[torch.Tensor] = None,  # [B, L_prompt]
    exclude_mask_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, list]:
    """
    Dream 版 hidden：
      - step=0 block hidden vs baseline 做余弦距离（1-cos）判定；
      - 触发后调用 apply_self_correction（随机重掩 + 原 token 抑制 + 逐步再填），
        与非 PAD 路径保持一致；支持 correction_scope 与锚点保护。
    """
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    if initial_mask_in_prompt is None:
        initial_mask_in_prompt = (input_ids == mask_id)

    # ---------- 规划 block ----------
    total_new = int(gen_length)
    if (block_length is None) or (block_length <= 0) or (block_length >= max(1, total_new)):
        num_blocks = 1
        seg_len = max(total_new, 0)
    else:
        num_blocks = math.ceil(total_new / block_length)
        seg_len = int(block_length)
    steps_per_block = max(int(round(steps / num_blocks)), 1)

    accept_ids, refuse_ids = build_cue_id_sets(tokenizer)

    # ---------- 初始 ids/attn ----------
    cur_ids = input_ids
    cur_attn = attention_mask

    # --------- 选择 block + PAD 注入（若需要）---------
    anchor_pos_mask = torch.zeros((cur_ids.size(0), cur_ids.size(1)), dtype=torch.bool, device=device)
    mask_pos = (cur_ids == mask_id).nonzero(as_tuple=False)
    planned_new_tokens = 0

    if mask_pos.numel() > 0:
        block_start = int(mask_pos[:, 1].min().item())
        block_end   = int(mask_pos[:, 1].max().item()) + 1
        planned_new_tokens = 0
    else:
        if (attack_method or "").lower() == "pad" and seg_len > 0:
            B = cur_ids.size(0)
            tail = torch.full((B, int(seg_len)), int(mask_id), dtype=cur_ids.dtype, device=device)
            cur_ids  = torch.cat([cur_ids, tail], dim=1)
            cur_attn = torch.cat([cur_attn, torch.ones_like(tail, device=device)], dim=1)
            block_start = int(cur_ids.size(1) - int(seg_len))
            block_end   = int(cur_ids.size(1))
            planned_new_tokens = 0
            # 注入锚点
            anchors = list(pad_anchors) if pad_anchors is not None else ["Step 1:", "Step 2:", "Step 3:"]
            blk_len = block_end - block_start
            if blk_len > 0 and tokenizer is not None and len(anchors) > 0:
                if pad_positions is None:
                    m = max(len(anchors), 1)
                    gap = max(blk_len // (m + 1), 1)
                    gap2 = max(int(gap / 1.5), 1)
                    rel_offsets = [(i + 1) * gap2 for i in range(m)]
                else:
                    rel_offsets = list(pad_positions)
                for rel, text in zip(rel_offsets, anchors):
                    tok = tokenizer(text, add_special_tokens=False).input_ids
                    if not tok:
                        continue
                    s = block_start + int(rel)
                    e = s + len(tok)
                    if s < block_end and e <= block_end:
                        cur_ids[:, s:e] = torch.tensor(tok, dtype=cur_ids.dtype, device=device).unsqueeze(0)
                        anchor_pos_mask[:, s:e] = True
        else:
            block_start = cur_ids.size(1)
            block_end   = block_start + int(seg_len)
            planned_new_tokens = int(seg_len)

    if fill_all_masks:
        block_start = 0
        block_end   = cur_ids.size(1)

    # ---------- Remask 控制器（仅作落子控制；不做 logits SP） ----------
    controller = AdaptiveRemaskController(
        mask_id=int(mask_id),
        block_start=int(block_start),
        block_end=int(block_end),
        steps_per_block=int(steps_per_block),
        alpha0=float(alpha0),
        accept_ids=accept_ids,
        refuse_ids=refuse_ids,
        enable_sp_logits=False,
    )

    # ---------- hidden hook：抓 step=0 的 block 均值 ----------
    hidden_holder: Dict[str, Optional[torch.Tensor]] = {"h_step0_block_mean": None}
    def _hidden_hook(step: int, x: torch.Tensor, h_last: torch.Tensor):
        if step == 0 and hidden_holder["h_step0_block_mean"] is None:
            try:
                h_blk = h_last[:, block_start:block_end, :]
                hidden_holder["h_step0_block_mean"] = h_blk.mean(dim=1).detach()
            except Exception:
                hidden_holder["h_step0_block_mean"] = None

    # ---------- 运行 diffusion_generate（长度保护） ----------
    input_len = cur_ids.shape[1]
    cfg = _clone_and_update_config(
        model,
        max_new_tokens=max(1, int(planned_new_tokens)),
        steps=int(steps_per_block),
        temperature=float(temperature),
        top_p=float(top_p) if top_p is not None else None,
        mask_token_id=int(mask_id),
        return_dict_in_generate=True,
        output_history=False,
    )
    if hasattr(cfg, "max_length"):
        cfg.max_length = int(input_len + max(1, int(planned_new_tokens)))

    out = model.diffusion_generate(
        inputs=cur_ids,
        generation_config=cfg,
        attention_mask=cur_attn,
        generation_logits_hook_func=lambda i, x, logits: controller.logits_hook(i, x, logits),
        generation_tokens_hook_func=lambda i, x, logits: controller.tokens_hook(i, x, logits),
        generation_hidden_hook_func=_hidden_hook,
    )
    seq = out.sequences if hasattr(out, "sequences") else out

    # ---------- 计算 hidden 分数（1 - cos） ----------
    sp_hidden = 0.0
    def _to_bh(x: Optional[torch.Tensor], dev: torch.device) -> Optional[torch.Tensor]:
        if x is None:
            return None
        x = x.detach()
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [H] -> [1, H]
        return x.to(device=dev, dtype=torch.float32)

    have_baseline = baseline_hidden is not None
    if have_baseline:
        ref_dev = baseline_hidden.device
        if hidden_holder["h_step0_block_mean"] is not None:
            h_blk  = _to_bh(hidden_holder["h_step0_block_mean"], ref_dev)
            h_base = _to_bh(baseline_hidden, ref_dev)
            cos = F.cosine_similarity(h_blk, h_base.expand_as(h_blk), dim=-1).mean().item()
            sp_hidden = max(0.0, float(1.0 - cos))
        elif attack_probe_hidden is not None:
            h_att  = _to_bh(attack_probe_hidden, ref_dev)
            h_base = _to_bh(baseline_hidden, ref_dev)
            cos = F.cosine_similarity(h_att, h_base.expand_as(h_att), dim=-1).mean().item()
            sp_hidden = max(0.0, float(1.0 - cos))
        else:
            sp_hidden = 0.0

    unsafe = have_baseline and (sp_hidden >= float(sp_threshold))
    logger.info(
        f"[Block 1/1|-H] have_baseline={have_baseline}, "
        f"have_hook_blk={hidden_holder['h_step0_block_mean'] is not None}, "
        f"have_probe={attack_probe_hidden is not None}, "
        f"SP(hidden)={sp_hidden:.3f}, thr={float(sp_threshold):.3f}, unsafe={unsafe}"
    )

    # ---------- 统一的“逐步再填”自纠正 ----------
    if unsafe and correct_only_first_block and refinement_steps > 0 and remask_ratio > 0.0:
                # 组合保护掩码：PAD 锚点（可选） ∪ 外部保护（system 前缀）
        excl = anchor_pos_mask if protect_anchors else None
        if exclude_mask_positions is not None:
            # 对齐到最终序列长度
            ex = exclude_mask_positions.to(seq.device)
            if ex.dim() == 1:
                ex = ex.unsqueeze(0)
            if ex.size(0) == 1 and seq.size(0) > 1:
                ex = ex.expand(seq.size(0), -1).clone()
            if ex.size(1) < seq.size(1):
                pad = torch.zeros(ex.size(0), seq.size(1)-ex.size(1), dtype=torch.bool, device=seq.device)
                ex = torch.cat([ex, pad], dim=1)
            elif ex.size(1) > seq.size(1):
                ex = ex[:, :seq.size(1)]
            excl = ex if excl is None else (excl | ex)

        seq = apply_self_correction(
            model=model,
            tokenizer=tokenizer,
            seq_ids=seq,
            initial_mask_in_prompt=initial_mask_in_prompt,
            steps=steps,
            temperature=temperature,
            mask_id=mask_id,
            refinement_steps=refinement_steps,
            remask_ratio=remask_ratio,
            suppression_value=suppression_value,
            correction_scope=correction_scope,
            exclude_mask_positions=excl,
            debug_print=False,
        )

    return seq, [float(sp_hidden)]
