import requests
import json
from tqdm import tqdm

model = "codellama/CodeLlama-34b-Instruct-hf"
API_URL = f"https://api-inference.huggingface.co/models/{model}"
# change for own access token
access_token = TODO
headers = {"Authorization": f"Bearer {access_token}"}

inputs = json.load(open("./output.json"))
responses = []

original = json.load(open("./data/human_eval_data_ambiguity_with_soln_new.json"))

prompt_prefix = "Complete the function, do not add a main method and do not return empty function body and do not pass"
for i in tqdm(range(len(inputs))):
	input = inputs[i]
	payload = {"inputs":prompt_prefix + " "+ input["refined_prompt"]}
	response = requests.post(API_URL, headers=headers, json=payload)
	response = response.json()[0]
	response["generated_text"] = response["generated_text"].replace(prompt_prefix, "").strip(" ")
	original[i]["solution"] = response["generated_text"]
	original[i]["llm_prompt"] = input["refined_prompt"]
	# print(response, "\n")
responses_json = json.dumps(original, indent=4)  

with open(f"codellama_output.json", "w") as file:
	file.write(responses_json)

