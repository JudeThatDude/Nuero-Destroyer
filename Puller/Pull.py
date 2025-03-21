import requests
from datasets import load_dataset

# Function to save text to a file
def save_to_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

# Get the README (instructions) from the Hugging Face repository
readme_url = "https://huggingface.co/datasets/neifuisan/Neuro-sama-QnA/raw/main/README.md"
response = requests.get(readme_url)
readme_content = response.text

# Save README to file
save_to_file("neuro_sama_instructions.txt", readme_content)

# Load the dataset
dataset = load_dataset("neifuisan/Neuro-sama-QnA")

# Inspect the dataset structure
print("Dataset structure:", dataset['train'].features)

# Convert the dataset to string format based on actual keys
qna_content = ""
for item in dataset['train']:
    # Since 'Question' and 'Answer' aren't the keys, let's try common alternatives
    # After checking the dataset, it might use different field names
    # For now, we'll dump all available fields
    qna_content += "Entry:\n"
    for key, value in item.items():
        qna_content += f"{key}: {value}\n"
    qna_content += "\n"

# Save QnA data to file
save_to_file("neuro_sama_qna.txt", qna_content)

print("Files saved successfully!")
print("Instructions saved to: neuro_sama_instructions.txt")
print("QnA data saved to: neuro_sama_qna.txt")