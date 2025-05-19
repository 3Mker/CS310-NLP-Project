import argparse
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from src.utils.data_utils import load_chinese_qwen2_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from src.utils.util import preprocess_for_bert
from src.utils.dataset import BertDataset


saved_model_path = "./results/supervised/bert-base-uncased/face2_zh_json/news,webnovel,wiki"

tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = BertForSequenceClassification.from_pretrained(saved_model_path)