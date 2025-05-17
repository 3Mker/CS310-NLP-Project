from transformers import BertTokenizer
from transformers import AutoTokenizer, BertForSequenceClassification
import torch

def preprocess_for_bert(records, tokenizer, max_length=512):
    inputs = []
    labels = []
    for record in records:
        record = records[record]
        prompt = record['prompt']
        result = record['result']
        label = record['label']
        
        # 拼接 prompt 和 result
        text = f"[PROMPT] {prompt} [RESULT] {result}"
        
        # Tokenization
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        inputs.append({
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        })
        labels.append(label)
    
    return inputs, torch.tensor(labels)

if __name__ == "__main__":
    print("start")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

    print("tokenizer loaded")
    records = [
        {"prompt": "问题1", "result": "答案1", "label": 1},
        {"prompt": "问题2", "result": "答案2", "label": 0}
    ]
    
    inputs, labels = preprocess_for_bert(records, tokenizer)
    print(inputs)
    print(labels)
    print("end")