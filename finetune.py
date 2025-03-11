import sys
import logging
import torch
import datasets
import transformers
import argparse
import yaml
import json
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_training_args(config):
    base_training_config = {
    "do_eval": False,
    "log_level": "info",
    "logging_strategy": "steps",
    "max_steps": -1,              # -1 means use num_epochs instead
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 1,  # Evaluation batch size
    "remove_unused_columns": True, # Dataset processing
    "seed": 0,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }   
    # Update with model-specific config
    base_training_config.update(config["training"])
    return TrainingArguments(**base_training_config)

def setup_model_and_tokenizer(config):
    model_kwargs = config["model"]["kwargs"]
    if model_kwargs.get("torch_dtype") == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"],
        device_map="auto",
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"])

    # Apply tokenizer settings from config
    tokenizer_config = config["model"].get("tokenizer", {})
    
    # Set custom pad token if specified
    if "pad_token" in tokenizer_config:
        tokenizer.add_special_tokens({"pad_token": tokenizer_config["pad_token"]})
        # Model needs to resize embeddings as we added a new token
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer.pad_token=tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        # verify pad token and ID
        # print(f"Pad token: {tokenizer.pad_token}")
        # print(f"Pad token ID: {tokenizer.pad_token_id}")
    
    tokenizer.model_max_length = tokenizer_config.get("max_length", 2048)
    tokenizer.padding_side = tokenizer_config.get("padding_side", "right")

    # tokenizer.pad_token = tokenizer.unk_token  
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    
    return model, tokenizer

def data_preprocessing(data,tokenizer):
    formatted_data = []
    for item in data:
        formatted_data.append({
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that refines ambiguous prompts for Python code generation. Please make the following input prompt more specific and unambiguous. Please only refine the prompt, do not try to generate the actual code solution or any explanation to your output."},
                {"role": "user", "content": f"Your input prompt is: {item['bad_prompt']}"},
                {"role": "assistant", "content": item["prompt"]},
            ]
        })
    
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=1)
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    train_dataset = train_dataset.map(
        lambda x: tokenizer.apply_chat_template(
            x["messages"], tokenize=True, add_generation_prompt=False, padding=True, return_dict=True
        ),
        remove_columns=["messages"],
        desc="Applying chat template to train set",
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenizer.apply_chat_template(
            x["messages"], tokenize=True, add_generation_prompt=False, padding=True, return_dict=True
        ),
        remove_columns=["messages"],
        desc="Applying chat template to eval set",
    )

    
    return train_dataset, eval_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Setup training arguments and PEFT config
    train_conf = setup_training_args(config)
    peft_conf = LoraConfig(**config["peft"], task_type="CAUSAL_LM", bias="none")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_conf.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load and process data
    data = json.load(open(args.data_path))
    train_dataset, eval_dataset = data_preprocessing(data,tokenizer)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=train_dataset, 
        eval_dataset = eval_dataset, 
        tokenizer=tokenizer,
    )

    # Train and save
    train_result = trainer.train()
    trainer.save_model(train_conf.output_dir)
    
    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)