import sys
import logging
import torch
import datasets
import transformers
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
import json


logger = logging.getLogger(__name__)


training_config = {
    "bf16": True, 
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 2, 
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
}

peft_config = {
    "r": 4,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",  # may change
    "modules_to_save": None,
}

train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)


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

logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}, "
    f"distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")


checkpoint_path = "nvidia/Nemotron-Mini-4B-Instruct"

model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'


def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example




data = json.load(open("data/gpt4_generations/fixed_prompts.json"))
data = data[:10]
formatted_data = []
for item in data:
    formatted_data.append({
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant that refines ambiguous prompts."},
            {"role": "user", "content": f"Make this prompt detailed and unambiguous: {item['bad_prompt']}"},
            {"role": "assistant", "content": item["prompt"]},
        ]
    })

dataset = Dataset.from_list(formatted_data)
processed_dataset = dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=["messages"],
    desc="Applying chat template",
)

# some args not found anymore 
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=processed_dataset,
    # max_seq_length=2048,
    # dataset_text_field="text",
    tokenizer=tokenizer,
    # packing=True
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# # not tested 
# tokenizer.padding_side = 'left'
# metrics = trainer.evaluate()
# metrics["eval_samples"] = len(processed_dataset)
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)

trainer.save_model(train_conf.output_dir)
