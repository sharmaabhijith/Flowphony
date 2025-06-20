from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"  # small, fast to test

try:
    print(f"Trying to load model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print("✅ Model and tokenizer loaded successfully from Hugging Face.")
except Exception as e:
    print("❌ Failed to load model.")
    print("Error:", str(e))