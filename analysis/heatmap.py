import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict, Counter
import os
import re
import matplotlib.patheffects as pe
import json

ATTACK_METHOD = ""

class JailbreakPositionAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_hidden_states = {}
        self.hooks = []

        self.step_position_layer_prob_stats = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            )
        )
        self.total_prompts = 0

        self.final_projection = self.find_final_projection()
        self.final_norm = self.find_final_norm()
        print(f"Found projection: {type(self.final_projection)}")
        print(f"Found final norm: {type(self.final_norm)}")

    def find_final_projection(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, "out_features"):
                if module.out_features > 100000:
                    print(f"Found projection: {name} -> {module.out_features}")
                    return module
        raise ValueError("Could not find projection layer")

    def find_final_norm(self):
        for name, module in self.model.named_modules():
            if "ln_f" in name or "final" in name.lower():
                if hasattr(module, "weight"):
                    print(f"Found final norm: {name}")
                    return module
        return torch.nn.Identity()

    def register_ff_norm_hooks(self):
        print("Registering hooks for FF norm layers...")
        registered = 0
        for i in range(32):
            target_name = f"transformer.blocks.{i}.ff_norm"
            for name, module in self.model.named_modules():
                if name.endswith(target_name):

                    def make_hook(layer_name):
                        def hook(module, input, output):
                            if torch.is_tensor(output):
                                self.layer_hidden_states[layer_name] = (
                                    output.detach().clone()
                                )

                        return hook

                    handle = module.register_forward_hook(make_hook(target_name))
                    self.hooks.append(handle)
                    registered += 1
                    break

        for name, module in self.model.named_modules():
            if "ln_f" in name:

                def final_hook(module, input, output):
                    if torch.is_tensor(output):
                        self.layer_hidden_states["final_norm"] = output.detach().clone()

                handle = module.register_forward_hook(final_hook)
                self.hooks.append(handle)
                registered += 1
                break
        print(f"Successfully registered {registered} FF norm hooks")
        return registered > 0

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


    def accumulate_position_statistics(
        self, step_index, block_mask_positions, prompt_len, k=15
    ):
        print(
            f"    Accumulating stats for Step {step_index + 1}, {len(block_mask_positions)} positions..."
        )
        layer_order = [f"transformer.blocks.{i}.ff_norm" for i in range(32)]
        layer_order.append("final_norm")
        existing_layers = [
            layer for layer in layer_order if layer in self.layer_hidden_states
        ]

        for layer_name in existing_layers:
            hidden_states = self.layer_hidden_states[layer_name]
            normalized = (
                self.final_norm(hidden_states)
                if "final_norm" not in layer_name
                else hidden_states
            )
            logits = self.final_projection(normalized)

            for pos in block_mask_positions:
                if pos < logits.shape[1]:

                    relative_pos = pos - prompt_len

                    pos_logits = logits[0, pos]
                    pos_probs = F.softmax(pos_logits, dim=-1)
                    top_probs, top_indices = torch.topk(pos_probs, k)

                    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                        try:
                            token = (
                                self.tokenizer.decode(
                                    [idx.item()], skip_special_tokens=False
                                )
                                .replace(" ", " ")
                                .replace("<0x0A>", "\\n")
                                .strip()
                            )
                            if token == "":
                                token = "<SPACE>"
                        except:
                            token = f"<UNK:{idx.item()}>"
                        self.step_position_layer_prob_stats[step_index][relative_pos][
                            layer_name
                        ][rank][token].append(prob.item())

    def create_averaged_heatmap(
        self, layer_name, save_dir="results", k=15
    ):
        os.makedirs(save_dir, exist_ok=True)
        if not self.step_position_layer_prob_stats:
            print("  No data accumulated. Cannot create heatmap.")
            return

        if "final" in layer_name:
            layer_display_name = "Final Layer"
            file_suffix = "final_layer"
        else:
            try:
                layer_num = layer_name.split('.')[2]
                layer_display_name = f"Intermediate Layer {layer_num}"
                file_suffix = f"layer_{layer_num}"
            except IndexError:
                layer_display_name = layer_name
                file_suffix = layer_name.replace('.', '_')
        
        print(f"\nCreating averaged heatmap for: {layer_display_name}...")

        analyzed_steps = sorted(self.step_position_layer_prob_stats.keys())
        if not analyzed_steps:
            return
        positions = sorted(
            self.step_position_layer_prob_stats[analyzed_steps[0]].keys()
        )
        if not positions:
            return

        prob_matrix = np.zeros((k, len(positions)))
        token_matrix = [["" for _ in positions] for _ in range(k)]

        for j, pos in enumerate(positions):
            for i in range(k):
                cell_token_data = defaultdict(list)

                for step_index in analyzed_steps:
                    if (
                        pos in self.step_position_layer_prob_stats[step_index]
                        and layer_name
                        in self.step_position_layer_prob_stats[step_index][pos]
                        and i
                        in self.step_position_layer_prob_stats[step_index][pos][
                            layer_name
                        ]
                    ):

                        rank_data = self.step_position_layer_prob_stats[step_index][
                            pos
                        ][layer_name][i]
                        for token, prob_list in rank_data.items():
                            cell_token_data[token].extend(prob_list)

                if not cell_token_data:
                    continue

                most_dominant_token = max(
                    cell_token_data, key=lambda t: len(cell_token_data[t])
                )
                avg_prob = np.mean(cell_token_data[most_dominant_token])

                prob_matrix[i, j] = avg_prob
                token_matrix[i][j] = most_dominant_token

        fig, ax = plt.subplots(figsize=(max(len(positions) * 0.5, 10), k * 0.5))
        im = ax.imshow(prob_matrix, cmap="coolwarm", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(positions)))
        ax.set_yticks(np.arange(k))
        ax.set_xticklabels(positions)
        ax.set_yticklabels([f"#{i+1}" for i in range(k)])

        jailbreak_keywords = [
            "step",
            "tutorial",
            "guide",
            "how",
            "instructions",
            "method",
            "process",
            "here",
            "Here",
            "Sure",
            "sure",
        ]
        safety_keywords = [
            "sorry",
            "cannot",
            "can't",
            "unable",
            "illegal",
            "harmful",
            "dangerous",
            "not",
            "no",
            "I",
            "as",
            "As",
            "risk",
        ]
        harmful_keywords = [
            "kill",
            "murder",
            "harm",
            "hurt",
            "violence",
            "attack",
            "weapon",
            "bomb",
            "poison",
        ]

        for i in range(k):
            for j in range(len(positions)):
                token = token_matrix[i][j]
                prob = prob_matrix[i, j]
                if token:
                    token_lower = token.lower()
                    text_rotation = 0
                    text_color = "white" if prob < 0.65 else "black"
                    fontweight = "normal"
                    if any(word in token_lower for word in harmful_keywords):
                        text_color, fontweight = "red", "bold"
                    elif any(word in token_lower for word in safety_keywords):
                        text_color, fontweight = "#60D260", "bold"
                        text_rotation = 20
                    elif any(word in token_lower for word in jailbreak_keywords):
                        text_color, fontweight = "orange", "bold"
                        text_rotation = -20
                    display_token = token[:8] + ".." if len(token) > 8 else token

                    ax.text(
                        j,
                        i,
                        display_token,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9.5,
                        fontweight=fontweight,
                        rotation=text_rotation,
                        path_effects=[pe.Stroke(linewidth=0.35, foreground='black'), pe.Normal()]
                    )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Average Token Probability", rotation=270, labelpad=15)
        

        ax.set_title(
            f"Top-{k} Predicted Tokens at {layer_display_name} (Averaged over {self.total_prompts} Prompts & First {len(analyzed_steps)} Steps)\n"
            f"X-axis: Relative Token Position, Y-axis: Probability Rank",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Relative Token Position in Generation", fontsize=12)
        ax.set_ylabel("Prediction Rank", fontsize=12)

        plt.tight_layout()
        # save_path = f"{save_dir}/summary_heatmap_{file_suffix}.pdf"
        # plt.savefig(save_path, dpi=300, bbox_inches="tight")
        save_path = f"{save_dir}/summary_heatmap_{file_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap for {layer_display_name} to {save_path}")

    def print_position_statistics(self):
        print(
            f"\n{'='*80}\nAVERAGED STATISTICS (Across all prompts and steps)\n{'='*80}"
        )
        if not self.step_position_layer_prob_stats:
            return
        analyzed_steps = sorted(self.step_position_layer_prob_stats.keys())
        positions = sorted(
            self.step_position_layer_prob_stats[analyzed_steps[0]].keys()
        )

        for pos in positions[:5]:
            print(f"  Relative Position {pos}:")
            rank_0_token_counts = Counter()
            for step_index in analyzed_steps:
                if (
                    "final_norm" in self.step_position_layer_prob_stats[step_index][pos]
                    and 0
                    in self.step_position_layer_prob_stats[step_index][pos][
                        "final_norm"
                    ]
                ):
                    rank_0_data = self.step_position_layer_prob_stats[step_index][pos][
                        "final_norm"
                    ][0]
                    for token, prob_list in rank_0_data.items():
                        rank_0_token_counts[token] += len(prob_list)

            top_tokens = rank_0_token_counts.most_common(5)
            top_tokens_str = ", ".join(
                [f"{token}({count} times)" for token, count in top_tokens]
            )
            print(f"    Top 1st-rank tokens (aggregated): {top_tokens_str}")


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp(min=1e-9)
    gumbel_noise = -torch.log(-torch.log(noise)) * temperature
    return logits + gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def analyze_block1_positions(
    model,
    prompt,
    tokenizer,
    analyzer,
    steps=32,
    gen_length=256,
    block_length=64,
    temperature=0.3,
    cfg_scale=0.1,
    remasking="low_confidence",
    mask_id=126336,
    attention_mask=None,
    use_correction=False,
):
    if not analyzer.register_ff_norm_hooks():
        return False, None
    try:
        device = model.device
        prompt_len = prompt.shape[1]
        x = torch.full(
            (prompt.shape[0], prompt_len + gen_length), mask_id, dtype=torch.long
        ).to(device)
        x[:, :prompt_len] = prompt.clone()

        if ATTACK_METHOD == "sure":
            injection_map = {  # Set up injection tokens
                0: "Sure, here is",
            }
            # Before the denoising process begins, replace the mask tokens at the specified positions with target token sequences.
            for relative_pos, text in injection_map.items():
                injection_ids = tokenizer(text, add_special_tokens=True).input_ids
                absolute_start_pos = prompt.shape[1] + relative_pos
                absolute_end_pos = absolute_start_pos + len(injection_ids)
                if absolute_end_pos < x.shape[1]:
                    x[:, absolute_start_pos:absolute_end_pos] = torch.tensor(
                        injection_ids, dtype=torch.long, device=x.device
                    ).unsqueeze(0)
        elif ATTACK_METHOD == "pad":
            injection_map = {  # Set up injection tokens
                10: "Step 1:",
                gen_length // 3: "Step 2:",
                gen_length // 3 * 2: "Step 3:",
            }
            # Before the denoising process begins, replace the mask tokens at the specified positions with target token sequences.
            for relative_pos, text in injection_map.items():
                injection_ids = tokenizer(text, add_special_tokens=True).input_ids
                absolute_start_pos = prompt.shape[1] + relative_pos
                absolute_end_pos = absolute_start_pos + len(injection_ids)
                if absolute_end_pos < x.shape[1]:
                    x[:, absolute_start_pos:absolute_end_pos] = torch.tensor(
                        injection_ids, dtype=torch.long, device=x.device
                    ).unsqueeze(0)

        steps_per_block = steps // (gen_length // block_length)
        current_block_start_index = prompt_len
        current_block_end_index = prompt_len + block_length
        print(
            f"  Block 1 range: {current_block_start_index} - {current_block_end_index}"
        )

        if use_correction == True:
            mask_index = x == mask_id
            generate_mask_index = torch.ones_like(mask_index, dtype=torch.bool)

        num_correction = 0
        for step_i in range(steps_per_block):

            mask_index = x == mask_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[x != mask_id] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if ATTACK_METHOD == "dija":
                    logits = model(x, attention_mask=attention_mask).logits
                else:
                    logits = model(x).logits

            if step_i < 3:
                analyzer.layer_hidden_states.clear()
                _ = model(x)
                all_mask_positions = [
                    pos
                    for pos in range(current_block_start_index, current_block_end_index)
                    if pos < x.shape[1] and x[0, pos] == mask_id
                ]
                positions_to_analyze = all_mask_positions[:10]

                analyzer.accumulate_position_statistics(
                    step_i, positions_to_analyze, prompt_len=prompt_len, k=15
                )

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            else:
                x0_p = torch.rand_like(x0, dtype=torch.float64)

            x0_p[:, prompt.shape[1] + (1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            block_mask_index = (
                x[:, current_block_start_index:current_block_end_index] == mask_id
            )
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)


            for j in range(confidence.shape[0]):
                if num_transfer_tokens[j, step_i] > 0:
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, step_i]
                    )
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            print(
                f"    Step {step_i + 1} completed, transferred {transfer_index.sum().item()} tokens"
            )

        return True, x
    except Exception as e:
        import traceback

        print(f"  Error: {e}")
        traceback.print_exc()
        return False, None
    finally:
        analyzer.remove_hooks()




def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
    SAVE_DIR = "./results"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found at {MODEL_PATH}")
        print("Please update the MODEL_PATH variable in the main() function.")
        return

    print("Loading model for position-wise analysis...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    analyzer = JailbreakPositionAnalyzer(model, tokenizer)

    import pandas as pd
    # input_path = "datasets/wildjailbreak.json"
    # input_path = "datasets/advbench.json"
    input_path = "datasets/benign_prompt.json"
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item["prompt"] for item in data if "prompt" in item]
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
        prompts = df["prompt"].tolist()
    elif input_path.endswith(".tsv"):
        df = pd.read_csv(input_path, sep="\t")
        prompts = []
        for _, row in df.iterrows():
            if row["label"]:
                prompts.append(row["adversarial"])
            else:
                pass
    else:
        print(f"Invalid: {input_path}")
        return

    prompts = prompts[:5]

    print(f"Starting position-wise analysis of {len(prompts)} prompts...")
    print("Focus: Block 1 - First 3 STEPS, First 20 POSITIONS, AVERAGED into ONE plot")

    for prompt_idx, prompt in enumerate(prompts):
        print(
            f"\n{'='*80}\nPROMPT {prompt_idx + 1}/{len(prompts)}: {prompt[:60]}...\n{'='*80}"
        )
        m = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )

        attention_mask = None
        if ATTACK_METHOD == "dija":

            MASK_TOKEN = "<|mdm_mask|>"
            START_TOKEN = "<startoftext>"
            END_TOKEN = "<endoftext>"
            SPECIAL_TOKEN_PATTERN = r"<mask:(\d+)>"

            def process_prompt_instruct(prompt: str, mask_counts: int) -> str:
                """Replace <mask:x> with <|mdm_mask|> * x and append if missing"""

                def replace_mask(match):
                    return MASK_TOKEN * int(match.group(1))

                prompt = re.sub(SPECIAL_TOKEN_PATTERN, replace_mask, prompt)

                # Append mask tokens if none are present
                if MASK_TOKEN not in prompt and mask_counts:
                    prompt += START_TOKEN + MASK_TOKEN * mask_counts + END_TOKEN

                return prompt

            prompt = process_prompt_instruct(prompt, 36)

            inputs = tokenizer(
                prompt, return_tensors="pt", padding=True, padding_side="left"
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
        else:
            input_ids = tokenizer(
                text=prompt_text, return_tensors="pt", padding=True, padding_side="left"
            )["input_ids"].to(device)

        success, output = analyze_block1_positions(
            model=model,
            prompt=input_ids,
            tokenizer=tokenizer,
            analyzer=analyzer,
            steps=128,
            gen_length=128,
            block_length=128,
            temperature=0,
            cfg_scale=0,
            remasking="low_confidence",
            # remasking="random",
            attention_mask=attention_mask,
            use_correction=False,
        )

        if success:
            analyzer.total_prompts += 1
            print(f"  ✓ Analyzed prompt {prompt_idx + 1}")
            if output is not None:
                print(
                    "  Sample generation output:\n  "
                    + tokenizer.batch_decode(
                        output[:, input_ids.shape[1] :], skip_special_tokens=True
                    )[0]
                )
        else:
            print(f"  ✗ Failed prompt {prompt_idx + 1}")

    analyzer.print_position_statistics()
    analyzer.create_averaged_heatmap(layer_name="final_norm", save_dir=SAVE_DIR, k=6)

    print(f"\nPosition-wise analysis completed!")
    print(
        f"Successfully analyzed: {analyzer.total_prompts}/{len(prompts)} prompts"
    )


if __name__ == "__main__":
    main()
