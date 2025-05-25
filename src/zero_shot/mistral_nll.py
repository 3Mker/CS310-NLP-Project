import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from src.supervised.train_supervised import load_dataset

def preprocess_for_mistral(records, tokenizer, max_length=512):
    texts = []
    labels = []
    for record in records:
        record = records[record]
        prompt = record['prompt']
        result = record['result']
        label = record['label']
        
        # 拼接 prompt 和 result
        text = f"[PROMPT] {prompt} [RESULT] {result}"
        texts.append(text)
        labels.append(label)
    return texts, labels

def compute_nll(model, tokenizer, texts, labels, device):
    """Compute Negative Log-Likelihood (NLL) for a list of texts."""
    model.eval()
    nll_scores_1 = []
    nll_scores_0 = []
    for text, label in tqdm(zip(texts, labels), desc="Computing NLL"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            nll = outputs.loss.item()
            if label == 1:
                nll_scores_1.append(nll)
            else:
                nll_scores_0.append(nll)
    return nll_scores_1, nll_scores_0

def save_nll_scores(output_path, nll_scores):
    """Save NLL scores to a file."""
    with open(output_path, 'w') as f:
        for score in nll_scores:
            f.write(f"{score}\n")
    print(f"NLL scores saved to {output_path}")
    

def main():
    # Paths
    model_path = "local_model_mistral"
    data_path = "face2_zh_json"  # Replace with your dataset path
    output_path = "results/mistral_nll_scores.txt"

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    # tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    # tokenizer.padding_side = "left"  # Set padding side to left
    # tokenizer.truncation_side = "left"  # Set truncation side to left
    # tokenizer.truncation = True  # Enable truncation
    # tokenizer.max_length = 512  # Set max length for truncation

    # Load dataset
    print("Loading dataset...")
    records = load_dataset(data_path, data_type=['news'])
    texts, labels = preprocess_for_mistral(records, tokenizer)
    print(f"Loaded {len(records)} records.")
    print(f"Number of texts: {len(texts)}")
    print(f"Number of labels: {len(labels)}")
    # 
    # Compute NLL scores
    print("Computing NLL scores...")
    nll_scores_1, nll_scores_0 = compute_nll(model, tokenizer, texts, labels, device)

    # Save results
    print("Saving NLL scores...")
    output_path_1 = os.path.join(output_path, "nll_scores_1.txt")
    output_path_0 = os.path.join(output_path, "nll_scores_0.txt")
    save_nll_scores(output_path_1, nll_scores_1)
    save_nll_scores(output_path_0, nll_scores_0)
    print("NLL scores computation completed.")
    print("All done!")

if __name__ == "__main__":
    main()