import json
a = json.load(open("../data/human_eval_data_ambiguity_with_soln_new.json"))
b = json.load(open("../data/code_llama_output_good.json"))
for i in range(len(a)):
    a[i]["solution"]= b[i]["generated_text"]
responses_json = json.dumps(a, indent=4)  # indent=4 for pretty formatting

with open(f"combined_code_llama_good.json", "w") as file:
    file.write(responses_json)