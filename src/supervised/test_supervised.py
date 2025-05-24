import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from src.utils.data_utils import load_chinese_qwen2_dataset, load_english_ghostbuster_dataset
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
import torch
from src.utils.util import preprocess_for_bert
from src.utils.dataset import BertDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def evaluate_ood(trainer, eval_dataset, domain_name, output_dir):
    """Evaluate model on a specific domain and return metrics."""
    print(f"\nEvaluating on domain: {domain_name}")
    results = trainer.evaluate(eval_dataset=eval_dataset)
    
    # Generate confusion matrix
    predictions = trainer.predict(eval_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix - {domain_name}')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{domain_name}.png'))
    plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train_type', type=str, required=True, 
                      help='Domain type that the model was trained on, e.g., news, webnovel, wiki for Chinese, or essay, wp, reuter for English')
    parser.add_argument('--test_types', nargs='+', type=str, required=True,
                      help='Domain types for testing')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='results_test/supervised')
    args = parser.parse_args()
    # Print arguments
    print("Arguments:")
    print(f"Data path: {args.data_path}")
    print(f"Model name: {args.model_name}")
    print(f"Train type: {args.train_type}")
    print(f"Test types: {args.test_types}")
    print(f"Batch size: {args.batch_size}")
    # print(f"Output directory: {args.output_dir}")
    
    # Create output directory for this specific experiment
    experiment_name = f"{args.model_name}/{args.train_type}_to_{'_'.join(args.test_types)}"
    args.output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load trained model and tokenizer
    if args.model_name == 'bert-base-chinese':
        model_path = f"results_train/supervised/bert-base-chinese/face2_zh_json/{args.train_type}"
        tokenizer_path = 'local_model_Bert_Chinese'
    elif args.model_name == 'bert-base-uncased':
        model_path = f"results_train/supervised/bert-base-uncased/ghostbuster-data/{args.train_type}"
        tokenizer_path = 'local_model_Bert_English'
    else:
        raise ValueError(f"Model {args.model_name} not supported.")
    
    # Load tokenizer and model
    print(f"Loading model trained on {args.train_type} from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model = model.to('cuda')

    # Set up trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = (preds == labels).mean()
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # Test on each target domain
    all_results = {}
    for test_type in args.test_types:
        print(f"\n=== Testing on domain: {test_type} ===")
        test_records = load_dataset(args.data_path, [test_type])
        test_inputs, test_labels = preprocess_for_bert(test_records, tokenizer)
        test_dataset = BertDataset(test_inputs, test_labels)
        
        results = evaluate_ood(trainer, test_dataset, test_type, args.output_dir)
        all_results[test_type] = results
    
    # Save results
    with open(os.path.join(args.output_dir, 'ood_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n=== OOD Testing Results Summary ===")
    for domain, results in all_results.items():
        print(f"\nDomain: {domain}")
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"F1 Score: {results['eval_f1']:.4f}")
        print(f"Precision: {results['eval_precision']:.4f}")
        print(f"Recall: {results['eval_recall']:.4f}")

if __name__ == '__main__':
    main()