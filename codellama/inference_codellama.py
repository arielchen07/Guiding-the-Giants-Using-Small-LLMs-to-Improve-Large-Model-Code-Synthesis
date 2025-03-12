import requests
import json
from tqdm import tqdm

model = "codellama/CodeLlama-34b-Instruct-hf"
API_URL = f"https://api-inference.huggingface.co/models/{model}"
# change for own access token
access_token = TODO
headers = {"Authorization": f"Bearer {access_token}"}

def append_entry_to_json(new_dict):
    file_path = "codellama_output.json"
    try:
        # Read the existing data from the file
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # If file doesn't exist, initialize with an empty list
        data = []

    # Append the new dict to the list
    data.append(new_dict)

    # Write the updated list back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

inputs = json.load(open("./output.json"))
responses = []

original = json.load(open("./data/human_eval_data_ambiguity_with_soln_new.json"))

prompt_prefix = "Complete the function, do not add a main method and do not return empty function body and do not pass"
for i in tqdm(range(len(inputs))):
	input = inputs[i]
        
	delim_substrs = ["\"\"\"\n", "'''\n"]
	filtered_refined_prompt = input["refined_prompt"]

	last_delim_index = -1
	for delim in delim_substrs:
		index = filtered_refined_prompt.rfind(delim)
		if index > last_delim_index:
			last_delim_index = index

	delim_substr_index = last_delim_index
	if delim_substr_index != -1:
		filtered_refined_prompt = filtered_refined_prompt[:delim_substr_index + len(delim_substrs[0])]

	payload = {"inputs": prompt_prefix + " " + filtered_refined_prompt}
	# payload = {"inputs": prompt_prefix + " " + input["refined_prompt"]}
	response = requests.post(API_URL, headers=headers, json=payload)
	# print(response)
	# print()
	# print(response.json())
	# print()
	response = response.json()[0]
	response["generated_text"] = response["generated_text"].replace(prompt_prefix, "").strip(" ")
	original[i]["solution"] = response["generated_text"]
	original[i]["llm_prompt"] = input["refined_prompt"]
	original[i]["llm_prompt_filtered"] = filtered_refined_prompt
	append_entry_to_json(original[i])
     