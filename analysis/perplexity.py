import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import math


def calculate_perplexity(model_id, json_file_path, text_key="response", max_length=None, stride=512):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    
    if max_length is None:
        max_length = model.config.max_position_embeddings

    prompts = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if text_key in item and isinstance(item[text_key], str):
                    prompts.append(item[text_key])
                else:
                    print(f"Warning: Item did not contain a valid string for key '{text_key}': {item}")
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return None, []
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
        return None, []
    
    print(f"Successfully loaded {len(prompts)} texts from the JSON file.")
    
    perplexities = []
    
    for text in tqdm(prompts, desc="Calculating Perplexity"):

        encodings = tokenizer(text, return_tensors="pt")
        
        seq_len = encodings.input_ids.size(1)
        
        nlls = [] 
        prev_end_loc = 0
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        ppl = torch.exp(torch.stack(nlls).mean())
        ppl_value = ppl.item()
        if not math.isnan(ppl_value) and not math.isinf(ppl_value):
            perplexities.append(ppl_value)
        else:
            print(f"Warning: Skipped invalid perplexity value ({ppl_value}) for text: {text[:50]}...")

    if perplexities:
        average_perplexity = sum(perplexities) / len(perplexities)
        return average_perplexity, perplexities
    else:
        return None, []


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity for texts in a JSON file using a language model")
    
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="meta-llama/Llama-2-7b-hf",
        help="Model ID or path"
    )
    
    parser.add_argument(
        "--json_file", 
        type=str, 
        required=True,
        default=None,
        help="Path to the JSON file containing texts"
    )
    
    parser.add_argument(
        "--text_key", 
        type=str, 
        default="response",
        help="Key name for text field in JSON"
    )
    
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=None,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--stride", 
        type=int, 
        default=512,
        help="Stride for sliding window"
    )
    
    args = parser.parse_args()
    
    average_perplexity, perplexities = calculate_perplexity(
        model_id=args.model_id,
        json_file_path=args.json_file,
        text_key=args.text_key,
        max_length=args.max_length,
        stride=args.stride
    )
    

    if average_perplexity is not None:
        print("\n--- Results ---")
        print(f"Total number of texts evaluated: {len(perplexities)}")
        print(f"Average Perplexity: {average_perplexity:.4f}")
    else:
        print("No valid texts were found to calculate perplexity.")


if __name__ == "__main__":
    main()