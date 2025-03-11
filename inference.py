import torch
import json
from peft import PeftModel
from tqdm import tqdm
import yaml
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from datasets import Dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def data_preprocessing(data):
    formatted_data = []
    for item in data:
        formatted_data.append([                    
            {"role": "user", "content": f"Refine this ambiguous prompt for Python code generation to be more specific and unambiguous: {item['bad_prompt']}. 
             Please only refine the prompt, do not try to generate the actual code solution or any explanation to your output."}
        ])
    
    return formatted_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to inference data")
    args = parser.parse_args()

    config = load_config(args.config)

    model_path = config["model"]["path"]
    finetuned_model_path = config["training"]["output_dir"]

    # merge the base model with the fined-tuned one
    base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, finetuned_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
    tokenizer.pad_token = config["tokenizer"]["pad_token"]
    tokenizer.padding_side = config["tokenizer"]["padding_side"]

    inference_config = config["inference"]

    data = load_data(args.data_path)
    processed_data = data_preprocessing(data)
    outputs = []

    # Example:
    # generator([{"role": "user", "content": "What is the capital of France? Answer in one word."}], do_sample=False, max_new_tokens=2)
    # [{'generated_text': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': 'Paris'}]}]
    
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, **inference_config)
    
    for idx, item in enumerate(tqdm(processed_data)):
        result = generator(item)
        # Parse the generated text to get just the assistant's response
        generated_conversation = json.loads(result[0]['generated_text'])
        assistant_response = generated_conversation[1]['content']
        
        outputs.append({
            "index": idx,
            "original_prompt": data[idx]["bad_prompt"],
            "refined_prompt": assistant_response
        })

    with open("output.json", "w") as f:
        json.dump(outputs, f, indent=4)