import torch
import json
from tqdm import tqdm
import yaml
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import gc

def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

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
            # {"role": "user", "content": f"Complete the function, do not add a main method and do not return empty function body and do not pass: {item["refined_prompt"]}"}
            {"role": "user", "content": f"Complete the function, do not add a main method and do not return empty function body and do not pass: {item["prompt"]}"}
        ])
    
    return formatted_data

if __name__ == "__main__":
    # Clean up at the start
    cleanup_gpu_memory()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to inference data")
    args = parser.parse_args()

    config = load_config(args.config)

    model_path = config["model"]["path"]

    # Create offload directory if it doesn't exist
    os.makedirs("offload", exist_ok=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            offload_folder="offload",
            pad_token_id=tokenizer.eos_token_id
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        inference_config = config["inference"]

        data = load_data(args.data_path)
        data = data[:2]
        processed_data = data_preprocessing(data)
        outputs = []

        generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, **inference_config)
        
        for idx, item in enumerate(tqdm(processed_data)):
            result = generator(item)
            print(result)

            # Parse the generated text to get just the assistant's response
            # generated_conversation = json.loads(result[0]['generated_text'])
            # assistant_response = generated_conversation[1]['content']
        #     generated_text = result[0]['generated_text']
        #     print(generated_text)
        #     assistant_response = generated_text.split("Here's a refined version of the prompt:")[-1].strip()
            
        #     outputs.append({
        #         "index": idx,
        #         "original_prompt": data[idx]["bad_prompt"],
        #         "refined_prompt": assistant_response
        #     })

        # with open("codellama_output.json", "w") as f:
        #     json.dump(outputs, f, indent=4)

            
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        # Clean up at the end, regardless of success or failure
        cleanup_gpu_memory()