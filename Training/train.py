import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("neifuisan/Neuro-sama-QnA")
print("Dataset loaded:", dataset)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Preprocess function using 'instruction' and 'output'
def preprocess_function(examples):
    inputs = [f"Q: {q} A: {a}" for q, a in zip(examples["instruction"], examples["output"])]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_dataset["train"]

# Training arguments
training_args = TrainingArguments(
    output_dir="./neuro-sama-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    evaluation_strategy="no",
    learning_rate=5e-5,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./neuro-sama-finetuned/final")
tokenizer.save_pretrained("./neuro-sama-finetuned/final")

# Test function with device fix
def generate_response(prompt):
    # Move inputs to the same device as the model (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure model is on the right device
    inputs = tokenizer(f"Q: {prompt} A:", return_tensors="pt").to(device)  # Move inputs to device
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test it
print(generate_response("What’s your favorite game?"))