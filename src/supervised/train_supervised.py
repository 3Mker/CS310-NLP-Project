import argparse
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from src.utils.data_utils import load_chinese_qwen2_dataset, load_english_ghostbuster_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from src.utils.util import preprocess_for_bert
from src.utils.dataset import BertDataset

def load_dataset(data_path, data_type):
    """Load dataset from a given path and return texts and labels."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path {data_path} does not exist.")
    if 'face2_zh_json' in data_path:
        print(f"Loading Chinese Qwen-2 dataset from {data_path} for types {data_type}")
        return load_chinese_qwen2_dataset(data_path, data_type)
    elif 'ghostbuster' in data_path:
        print(f"Loading English Ghostbuster dataset from {data_path} for types {data_type}")
        return load_english_ghostbuster_dataset(data_path, data_type)
    else:
        print(f"Error: Unsupported dataset path {data_path}.")
        raise ValueError(f"Data type {data_type} not supported.")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='ghostbuster-data') # 'ghostbuster-data'
    parser.add_argument('--data_type', type=str, default=['essay','wp','reuter']) # ['essay','wp','reuter']  ['news','webnovel','wiki']
    parser.add_argument('--output_dir', type=str, default='results_train/supervised')
    parser.add_argument('--model_name', type=str, default='bert-base-chinese') # 'bert-base-uncased'
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--if_local', type=bool, default=True)
    args = parser.parse_args()

    string_data_type = ','.join(args.data_type)
    string_data_path_and_type = os.path.join(args.data_path, string_data_type)

    args.output_dir = os.path.join(args.output_dir, args.model_name, string_data_path_and_type)

    # print arguments
    print("Arguments:")
    print(f"Data path: {args.data_path}")
    print(f"Data type: {args.data_type}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    records = load_dataset(args.data_path, args.data_type)
    print(f"Loaded {len(records)} records.")
    print(records[0])
    print(records[7000])
    print(records[14000])
    exit()
    if args.model_name == 'bert-base-chinese':
        model_path = 'local_model_Bert_Chinese'
        if args.if_local:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
        
    elif args.model_name == 'bert-base-uncased':  
        model_path = 'local_model_Bert_English'
        if args.if_local:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    else:
        raise ValueError(f"Model {args.model_name} not supported.")

    inputs, labels = preprocess_for_bert(records, tokenizer)

    # Split dataset; placeholder split 80/20
    train_enc, eval_enc, train_labels, eval_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)
    
    # Convert to PyTorch Dataset
    train_dataset = BertDataset(train_enc, train_labels)
    eval_dataset = BertDataset(eval_enc, eval_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        load_best_model_at_end=True,
        save_strategy='steps',
        eval_strategy='steps',
    )
    def compute_metrics(pred):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)



if __name__ == '__main__':
    main()