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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn.parallel import DataParallel

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
    parser.add_argument('--data_type', nargs='+', type=str, default=['essay','wp','reuter']) # ['essay','wp','reuter']  ['news','webnovel','wiki']
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

    # Wrap the model with DataParallel for multi-GPU training
    model = DataParallel(model, device_ids=[0, 1, 2])

    # Ensure the model is moved to the appropriate device
    model = model.to('cuda')

    inputs, labels = preprocess_for_bert(records, tokenizer)

    # Split dataset; placeholder split 80/20
    train_enc, eval_enc, train_labels, eval_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)
    
    #count 0 in eval_labels
    count_0 = 0
    count_1 = 0
    for label in eval_labels:
        if label == 0:
            count_0 += 1
        else:
            count_1 += 1
    print(f"Count of 0 in eval_labels: {count_0}")
    print(f"Count of 1 in eval_labels: {count_1}")

    # Convert to PyTorch Dataset
    train_dataset = BertDataset(train_enc, train_labels)
    eval_dataset = BertDataset(eval_enc, eval_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        load_best_model_at_end=True,
        save_strategy='epoch',
        eval_strategy='epoch',
    )

    def plot_metrics(metrics, output_dir):
        epochs = range(1, len(metrics['loss']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics['loss'], label='Loss')
        plt.plot(epochs, metrics['accuracy'], label='Accuracy')
        plt.plot(epochs, metrics['precision'], label='Precision')
        plt.plot(epochs, metrics['recall'], label='Recall')
        plt.plot(epochs, metrics['f1'], label='F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.title('Training Metrics')
        plt.savefig(f"{output_dir}/metrics_curve.png")
        plt.close()

    def compute_metrics(pred):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)

        # Generate confusion matrix
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"{training_args.output_dir}/confusion_matrix_epoch_{trainer.state.epoch}.png")
        plt.close()

        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for epoch in range(args.epochs):
        trainer.train()
        eval_results = trainer.evaluate()

        metrics['loss'].append(eval_results['eval_loss'])
        metrics['accuracy'].append(eval_results['eval_accuracy'])
        metrics['precision'].append(eval_results['eval_precision'])
        metrics['recall'].append(eval_results['eval_recall'])
        metrics['f1'].append(eval_results['eval_f1'])

    plot_metrics(metrics, training_args.output_dir)
    trainer.train()
    trainer.save_model(args.output_dir)



if __name__ == '__main__':
    main()