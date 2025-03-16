import json
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding as the text embedding
    return outputs.last_hidden_state[:, 0, :].numpy()

def compute_similarities():
    # Load the data
    with open("codellama_output.json", "r") as f:
        data = json.load(f)
    
    tokenizer, model = load_model()
    
    similarities = []
    
    # Process each pair of prompts
    for entry in tqdm(data):
        original_prompt = entry["prompt"]
        filtered_prompt = entry["llm_prompt_filtered"]
        
        # Get embeddings
        orig_embedding = get_embedding(original_prompt, tokenizer, model)
        filtered_embedding = get_embedding(filtered_prompt, tokenizer, model)
        
        # Compute cosine similarity
        similarity = cosine_similarity(orig_embedding, filtered_embedding)[0][0]
        
        similarities.append({
            "index": entry.get("index", -1),
            "similarity": float(similarity),
            "original_length": len(original_prompt),
            "filtered_length": len(filtered_prompt),
            "original_tokens": len(tokenizer.tokenize(original_prompt)),
            "filtered_tokens": len(tokenizer.tokenize(filtered_prompt))
        })
    
    # Save results
    with open("prompt_similarities.json", "w") as f:
        json.dump(similarities, f, indent=4)
    
    # summary statistics
    similarities_values = [s["similarity"] for s in similarities]
    print("\nSimilarity Statistics:")
    print(f"Mean: {np.mean(similarities_values):.4f}")
    print(f"Median: {np.median(similarities_values):.4f}")
    print(f"Min: {np.min(similarities_values):.4f}")
    print(f"Max: {np.max(similarities_values):.4f}")
    
    # highest and lowest similarity
    print("\nLowest Similarity Example:")
    lowest_idx = np.argmin(similarities_values)
    print(f"Index: {data[lowest_idx].get('index', lowest_idx)}")
    print(f"Similarity: {similarities_values[lowest_idx]:.4f}")
    print("Original:", data[lowest_idx]["prompt"])
    print("Filtered:", data[lowest_idx]["llm_prompt_filtered"])
    
    print("\nHighest Similarity Example:")
    highest_idx = np.argmax(similarities_values)
    print(f"Index: {data[highest_idx].get('index', highest_idx)}")
    print(f"Similarity: {similarities_values[highest_idx]:.4f}")
    print("Original:", data[highest_idx]["prompt"])
    print("Filtered:", data[highest_idx]["llm_prompt_filtered"])

if __name__ == "__main__":
    compute_similarities() 