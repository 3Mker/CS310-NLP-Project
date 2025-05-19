from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification

# # Chinese BERT模型
# # 替换为你需要的模型名称
# model_name = "bert-base-chinese"

# # 下载模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

# # 保存到本地
# model.save_pretrained("./local_model_Bert_Chinese")
# tokenizer.save_pretrained("./local_model_Bert_Chinese")

# English BERT模型
model_name = "bert-base-uncased"
# 下载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
# 保存到本地
model.save_pretrained("./local_model_Bert_English")
tokenizer.save_pretrained("./local_model_Bert_English")