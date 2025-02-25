import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt4_api import make_requests as make_gpt4_requests

API_KEY = "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"


random.seed(42)


def encode_prompt(prompt_instructions, classification=False):
    """Encode multiple prompt instructions into a single string."""
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of tasks:\n"
    
    for idx, instruction_pair in enumerate(prompt_instructions):
        # Ensure each instruction is a dictionary with 'prompt' and 'bad_prompt' keys
        if isinstance(instruction_pair, dict) and 'prompt' in instruction_pair and 'bad_prompt' in instruction_pair:
            # Process the 'prompt' part
            prompt_text = re.sub(r"\s+", " ", instruction_pair['prompt']).strip().rstrip(":")
            # Process the 'bad_prompt' part
            bad_prompt_text = re.sub(r"\s+", " ", instruction_pair['bad_prompt']).strip().rstrip(":")
            # Add both to the prompt string
            prompt += f"{idx+1}. Prompt: {prompt_text}\n   Bad Prompt: {bad_prompt_text}\n"
        else:
            print(f"Invalid instruction format at index {idx}: {instruction_pair}")
    
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, similarities, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt4_response(response):
    if response is None:
        return []
    instructions = []
    for inst in response:
        prompt = inst["prompt"]
        bad_prompt = inst["bad_prompt"]

        prompt = re.sub(r"\s+", " ", prompt).strip()
        bad_prompt = re.sub(r"\s+", " ", bad_prompt).strip()

        prompt = prompt.strip().capitalize()
        bad_prompt = bad_prompt.strip().capitalize()

        if prompt == "" or bad_prompt == "":
            continue
        # filter out too short or too long instructions
        if len(prompt.split()) <= 3 or len(prompt.split()) > 600 or len(bad_prompt.split()) <= 3 or len(bad_prompt.split()) > 600:
            continue
        # # filter based on keywords that are not suitable for language models.
        # if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
        #     continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result. 
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        # if inst.startswith("Write a program"):
        #     continue
        # filter those starting with punctuation
        if prompt[0] in string.punctuation or bad_prompt[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not prompt[0].isascii() or not bad_prompt[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt4_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=1,
        help="the number of instructions to generate",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-4o-mini",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT4 at a time."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Read the entire JSON file as a single JSON object
    with open(args.seed_tasks_path, "r") as f:
        seed_tasks = json.load(f)
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [{"prompt": t["prompt"], "bad_prompt": t["bad_prompt"]} for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")
    
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "generated_prompts.jsonl")):
        with open(os.path.join(args.batch_dir, "generated_prompts.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    with open(os.path.join(args.batch_dir, "generated_prompts.json"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, 
                    similarities=None,
                    n=2)
                # sample human instructions from the pool
                prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
                batch_inputs.append(prompt)

            results = make_gpt4_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=True,
                n=1,
                # best_of=1,
                api_key=API_KEY,
                organization=args.organization,
            )

            instructions = post_process_gpt4_response(results)

            for inst in instructions:
                prompt_text = inst['prompt']
                bad_prompt_text = inst['bad_prompt']
                
                # Extract the 'bad_prompt' text from each instruction for scoring
                seed_prompts = [instr['bad_prompt'] for instr in seed_instructions if 'bad_prompt' in instr]
                machine_prompts = [instr['bad_prompt'] for instr in machine_instructions if 'bad_prompt' in instr]
                
                with Pool(4) as p:
                    # prompt_rouge_scores = p.map(partial(scorer.score, prompt_text), seed_prompts + machine_prompts)
                    bad_prompt_rouge_scores = p.map(partial(scorer.score, bad_prompt_text), seed_prompts + machine_prompts)
                
                # Calculate average scores or use them as needed
                # prompt_rouge_scores = [score["rougeL"].fmeasure for score in prompt_rouge_scores]
                bad_prompt_rouge_scores = [score["rougeL"].fmeasure for score in bad_prompt_rouge_scores]
                
                # Use the maximum score
                max_rouge_score = max(bad_prompt_rouge_scores)
                
                if max_rouge_score > 0.7:
                    continue
                
                all_instructions = seed_instructions + machine_instructions
                
                # # Calculate the most similar instructions using bad_prompt_rouge_scores
                # most_similar_instructions = {
                #         all_instructions[i] : bad_prompt_rouge_scores[i] for i in np.argsort(bad_prompt_rouge_scores)[-10:][::-1]
                #     }

                # fout.write(json.dumps({
                #     "instruction": inst,
                #     "most_similar": most_similar_instructions,
                #     "avg_similarity_score": float(np.mean(combined_rouge_scores)),
                #     "request_idx": request_idx
                # }) + "\n")

                # # Append the instruction and write to file
                machine_instructions.append(inst)
                inst_with_score = inst.copy()  # Create a copy of the original dictionary
                inst_with_score.update({"avg_similarity_score": float(np.mean(bad_prompt_rouge_scores))})
                fout.write(json.dumps(inst_with_score, indent=4) + "\n")
                progress_bar.update(1)
            request_idx += 1
