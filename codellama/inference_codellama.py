import requests
import json
from tqdm import tqdm

model = "codellama/CodeLlama-34b-Instruct-hf"
API_URL = f"https://api-inference.huggingface.co/models/{model}"
# change for own access token
access_token = TODO
headers = {"Authorization": f"Bearer {access_token}"}

with open("../data/human_eval_data_ambiguity_with_soln_new.json", "r") as file:
	inputs = json.load(file)

 # delete this to use all examples
# inputs = inputs[:2]
responses = []

prompt_prefix = "Complete the function, do not add a main method and do not return empty function body and do not pass"
for i in tqdm(range(len(inputs))):
	input = inputs[i]
	# print(f"{input["prompt"]=}\n")
	payload = {"inputs":prompt_prefix + " "+ input["prompt"]}
	response = requests.post(API_URL, headers=headers, json=payload)
	response = response.json()[0]
	response["generated_text"] = response["generated_text"].replace(prompt_prefix, "")
	response["index"] = i 
	responses.append(response)
	if i % 10 ==0:
		responses_json = json.dumps(responses, indent=4)
		with open(f"output_good.json", "w") as file:
			file.write(responses_json)
	
	# print(response, "\n")
	
responses_json = json.dumps(responses, indent=4)  # indent=4 for pretty formatting

with open(f"output_good.json", "w") as file:
    file.write(responses_json)

