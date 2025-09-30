import os
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
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
def generate(
    model,
    tokenizer,
    prompt,
    steps=64,
    gen_length=128,
    block_length=128,
    temperature=0.5,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    random_rate=0,
    injection_str=None,
    injection_step=None,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            if injection_step is not None and injection_str is not None:
                if i == injection_step:
                    injection_map = {
                        0: injection_str,
                    }
                    # print("Injecting jailbreak tokens...")
                    for relative_pos, text in injection_map.items():
                        injection_ids = tokenizer(text, add_special_tokens=True).input_ids
                        absolute_start_pos = prompt.shape[1] + relative_pos
                        absolute_end_pos = absolute_start_pos + len(injection_ids)
                        if absolute_end_pos < x.shape[1]:
                            x[:, absolute_start_pos:absolute_end_pos] = torch.tensor(
                                injection_ids, dtype=torch.long, device=x.device
                            ).unsqueeze(0)

            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence" or i != 0:
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == "rate":
                p = F.softmax(logits, dim=-1)
                model_confidence = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                random_confidence = torch.rand(
                    (x0.shape[0], x0.shape[1]), device=x0.device
                )
                x0_p = (
                    1 - random_rate
                ) * model_confidence + random_rate * random_confidence
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default="LLaDA-8B", help="nickname of the model"
    )
    parser.add_argument(
        "--custom_cache_dir",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="path to local model or model identifier from huggingface",
    )
    parser.add_argument("--steps", type=int, default=64, help="total diffusion steps")
    parser.add_argument("--gen_length", type=int, default=128, help="length of generated answer")
    parser.add_argument("--block_length", type=int, default=128, help="length of each blocks")
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="temperature for sampling"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=0.0,
        help="classifier-free guidance scale",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="the device to run the model on",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        default=None,
        help="path to input file (json/csv/tsv)",
    )
    parser.add_argument(
        "--num_of_test",
        type=int,
        default=None,
        help="number of test instances to run",
    )
    parser.add_argument(
        "--random_rate",
        type=float,
        default=0.0, 
        help="the random rate for remasking",
    )
    parser.add_argument(
        "--injection_str",
        type=str,
        default=None,
        help="which string to inject",
    )
    parser.add_argument(
        "--injection_step",
        type=int,
        default=None,
        help="which step to inject the string",
    )

    args = parser.parse_args()

    device = args.device
    custom_cache_dir = args.custom_cache_dir

    model = AutoModel.from_pretrained(
        f"{custom_cache_dir}",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=custom_cache_dir,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        f"{custom_cache_dir}",
        trust_remote_code=True,
        cache_dir=custom_cache_dir
    )

    tokenizer.pad_token = tokenizer.eos_token
    input_path = args.input_path

    output_json_path = f"results/result_{args.model_name}_random_{args.random_rate}_inject_{args.injection_str}_{args.injection_step}.json"
    print(f"Loading prompts from {input_path}")
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item["prompt"] for item in data]
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
        print(f"File type not available: {input_path}")
        return


    if args.num_of_test is not None and args.num_of_test < len(prompts):
        print(f"Use first {args.num_of_test} instances.")
        prompts = prompts[: args.num_of_test]


    results = []

    for idx, original_prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        m = [{"role": "user", "content": original_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )

        input_ids = tokenizer(
            text=prompt_text, return_tensors="pt", padding=True, padding_side="left"
        )["input_ids"].to(device)

        output_ids = generate(
            model,
            tokenizer,
            input_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking="rate",
            random_rate=args.random_rate,
            injection_str=args.injection_str,
            injection_step=args.injection_step
        )

        response_text = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        results.append(
            {
                "id": idx,
                "prompt": original_prompt,
                "response": response_text.strip(),
                "length": len(response_text.strip()),
            }
        )

        if len(results) % 5 == 0:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Completed all prompts.")


if __name__ == "__main__":
    main()
