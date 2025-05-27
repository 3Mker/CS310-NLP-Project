import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from src.supervised.train_supervised import load_dataset

# Set CUDA_VISIBLE_DEVICES to use GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    return texts[:2], labels[:2]

def compute_nll(model, tokenizer, texts, labels, device):
    """Compute Negative Log-Likelihood (NLL) for each token in a list of texts."""
    model.eval()
    nll_scores_1 = []
    nll_scores_0 = []
    for text, label in tqdm(zip(texts, labels), desc="Computing NLL"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.size())
            token_losses = token_losses.cpu().tolist()  # Convert to list for each token
            if label == 1:
                nll_scores_1.append(token_losses)
            else:
                nll_scores_0.append(token_losses)
    return nll_scores_1, nll_scores_0

def compute_nll_per_char(model, tokenizer, texts, labels, device):
    """Compute Negative Log-Likelihood (NLL) for each character in a list of texts."""
    model.eval()
    nll_scores_1 = []
    nll_scores_0 = []
    for text, label in tqdm(zip(texts, labels), desc="Computing NLL per character"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.size()).cpu().tolist()

        # Map token losses to characters
        tokenized_text = tokenizer.tokenize(text)
        char_losses = []
        token_idx = 0
        for char in text:
            if char.strip():  # Ignore spaces
                token_loss = token_losses[0][token_idx]
                char_losses.append(token_loss / len(tokenized_text[token_idx]))  # Average over token length
                token_idx += 1
        
        if label == 1:
            nll_scores_1.append(char_losses)
        else:
            nll_scores_0.append(char_losses)
    return nll_scores_1, nll_scores_0

def save_nll_scores(output_path, nll_scores):
    """Save NLL scores to a file, one sentence per line without brackets, separated by spaces."""
    # make sure the txt file exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if nll_scores:
        with open(output_path, 'w') as f:
            for sentence_scores in nll_scores:
                # Flatten the sentence_scores if it contains nested lists
                if isinstance(sentence_scores, list) and all(isinstance(token_score, list) for token_score in sentence_scores):
                    sentence_scores = [score for sublist in sentence_scores for score in sublist]
                f.write(" ".join(f"{score:.6f}" for score in sentence_scores) + "\n")
        print(f"NLL scores saved to {output_path}")
    else:
        print(f"No NLL scores to save for {output_path}.")
    

def main():
    # Paths
    model_path = "local_model_mistral"
    data_path = "face2_zh_json"  # Replace with your dataset path
    output_path = "results_nll"

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    data_types = ['news', 'webnovel', 'wiki']
    for data_type in data_types:
        # Load dataset
        print(f"Loading dataset for {data_type}...")
        records = load_dataset(data_path, data_type=[data_type])
        texts, labels = preprocess_for_mistral(records, tokenizer)
        print(f"Loaded {len(records)} records.")
        print(f"Number of texts: {len(texts)}")
        print(f"Number of labels: {len(labels)}")
        # Check if texts and labels are not empty before computing NLL scores
        if texts and labels:
            # Compute NLL scores
            print("Computing NLL scores...")
            nll_scores_1, nll_scores_0 = compute_nll(model, tokenizer, texts, labels, device)
        else:
            print("No texts or labels found, skipping NLL computation.")
            nll_scores_1, nll_scores_0 = [], []

        # Save results
        print("Saving NLL scores...")
        output_path_1 = os.path.join(output_path, data_type, "nll_scores_1.txt")
        output_path_0 = os.path.join(output_path, data_type, "nll_scores_0.txt")
        save_nll_scores(output_path_1, nll_scores_1)
        save_nll_scores(output_path_0, nll_scores_0)
        print("NLL scores computation completed.")
        print("All done!")

if __name__ == "__main__":
    main()