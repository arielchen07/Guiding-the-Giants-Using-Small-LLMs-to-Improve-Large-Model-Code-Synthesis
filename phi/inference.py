
from transformers import pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
import json
from tqdm import tqdm 

# path for after fine tuning 
# model_path = "./checkpoint_dir/checkpoint-9254" 
# path for before finetuning
model_path = "microsoft/Phi-3.5-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer
)

def generate_detailed_prompt(bad_prompt):
    input_text = f"""<|system|>
You are a helpful AI assistant that refines ambiguous prompts <|end|>.
<|user|>
Make this prompt detailed and unambiguous: {bad_prompt} <|end|>
<|assistant|>"""
    
    generation_args = {
        "max_new_tokens": 150,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    
    output = pipe(input_text, **generation_args)
    return output[0]['generated_text']

# Example inference
data = json.load(open("data/human_eval_data_ambiguity_with_soln_new.json"))
outputs = []
for d in tqdm(data):
    output = generate_detailed_prompt(d["bad_prompt"])
    outputs.append(output)
    print(output)
responses_json = json.dumps(outputs, indent=4)  
with open(f"phi_outputs_without_finetune.json", "w") as file:
    file.write(responses_json)
    
