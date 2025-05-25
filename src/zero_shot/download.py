# Use a pipeline as a high-level helper
from transformers import pipeline

# pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# save to local directory
local_path = "local_model_mistral"
model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)

print("Model and tokenizer saved to local directory.")

# Load model from local directory
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained(local_path)
# model = AutoModelForCausalLM.from_pretrained(local_path)

# print("Model and tokenizer loaded successfully from local directory.")