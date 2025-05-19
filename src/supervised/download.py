from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification

# 替换为你需要的模型名称
model_name = "bert-base-chinese"

# 下载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

# 保存到本地
model.save_pretrained("./local_model")
tokenizer.save_pretrained("./local_model")