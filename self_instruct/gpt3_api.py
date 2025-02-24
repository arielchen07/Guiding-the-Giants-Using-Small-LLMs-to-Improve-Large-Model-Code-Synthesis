import json
import tqdm
import os
import random
import openai
from datetime import datetime
import argparse
import time

from openai import OpenAI
    

def make_requests(
        engine, prompts, max_tokens, temperature, top_p, 
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, api_key, retries=10, organization=None
    ):
    response = None
    target_length = max_tokens
    if organization is not None:
        openai.organization = organization
    
    client = OpenAI(api_key=api_key)

    retry_cnt = 0
    backoff_time = 30

    messages = [
                {
                    "role": "system",
                    "content": "You are an intelligent generator for coding problem descriptions."
                        "Please generate a pair of good code prompt and bad code prompt."
                        "The good code prompt should clearly describe the coding problem,"
                        "the bad code prompt is describing the same problem but should be more ambiguious with its problem descriptions hving multiple meanings."
                        "You can think of the bad code prompt as a question that some non-programmer may ask."
                        f"You can refer to these as examples: {prompts}"
                        "You shouldn't response empty and you should generate new pair of good code prompt and bad code prompt."
                        "Please use the format below: (the output should be json format)\n"
                        "{'prompt': 'the good code prompt', 'bad_prompt':'the bad code prompt'}"
                }
            ]

    while retry_cnt <= retries:
        try:
            completion = client.chat.completions.create(
                model=engine,
                response_format={ "type": "json_object" },
                messages=messages,
                max_tokens=target_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=n,
                stop=stop_sequences,
                logprobs=logprobs,
                # best_of=best_of,
            )
            response = completion.choices[0].message.content

            if response:
                # Parse the response string into a JSON object (dictionary)
                response_dict = json.loads(response)

                # all_response.append(response_dict)

                # break
                return [response_dict]

            else:
                # count_empty_response += 1
                print("No response or empty response from OpenAI API")
                assistant_message = {"role": "assistant", "content": response}
                user_message = {"role": "user", "content": "The response is empty, please try again:\n" + "please categorize the image into one of the given categories."}
                messages.append(assistant_message)
                messages.append(user_message)
                continue
            
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            print("Wait for a second and ask chatGPT4 again!")
            time.sleep(1)
            retry_cnt += 1
            continue

    # if isinstance(prompts, list):
    #     results = []
    #     for j, prompt in enumerate(prompts):
    #         data = {
    #             "prompt": prompt,
    #             "response": {"choices": choices[j * n: (j + 1) * n]} if choices else None,
    #             "created_at": str(datetime.now()),
    #         }
    #         results.append(data)
    #     return results
    # else:
    #     data = {
    #         "prompt": prompts,
    #         "response": response,
    #         "created_at": str(datetime.now()),
    #     }
        # return [data]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to GPT3.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from GPT3.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="The openai GPT3 engine to use.",
    )
    parser.add_argument(
        "--max_tokens",
        default=500,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of GPT3.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of GPT3.",
    )
    parser.add_argument(
        "--logprobs",
        default=5,
        type=int,
        help="The `logprobs` parameter of GPT3"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="The `n` parameter of GPT3. The number of responses to generate."
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists."
    )
    parser.add_argument(
        "--request_batch_size",
        default=20,
        type=int,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()

    
if __name__ == "__main__":
    random.seed(123)
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # read existing file if it exists
    existing_responses = {}
    if os.path.exists(args.output_file) and args.use_existing_responses:
        with open(args.output_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                existing_responses[data["prompt"]] = data

    # do new prompts
    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = [json.loads(line)["prompt"] for line in fin]
        else:
            all_prompt = [line.strip().replace("\\n", "\n") for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
            batch_prompts = all_prompts[i: i + args.request_batch_size]
            if all(p in existing_responses for p in batch_prompts):
                for p in batch_prompts:
                    fout.write(json.dumps(existing_responses[p]) + "\n")
            else:
                results = make_requests(
                    engine=args.engine,
                    prompts=batch_prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    stop_sequences=args.stop_sequences,
                    logprobs=args.logprobs,
                    n=args.n,
                )
                for data in results:
                    fout.write(json.dumps(data) + "\n")