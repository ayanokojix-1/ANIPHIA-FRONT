import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import kagglehub
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Set device and model name
model_name = "distilgpt2"
device = "cpu"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Download the dataset from Kaggle using kagglehub
# Example dataset: 'zynicide/wine-reviews' - change it to your dataset name
dataset_path = kagglehub.dataset_download('')

# Load dataset into pandas (assuming it's a CSV file, adjust if different)
df = pd.read_csv(dataset_path)
df = df[:1000]  # Limit to 1000 samples if necessary

# Convert pandas dataframe to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Preprocess the dataset
def map_dataset(batch):
    return tokenizer(
        batch["Plot"],  # Adjust the column name as per your dataset
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True
    )

# Apply the preprocessing to the dataset
dataset = dataset.map(
    map_dataset,
    batched=True,
    batch_size=8,
    remove_columns=list(df.columns)  # Remove the original columns after processing
)

# Remove overflow_to_sample_mapping if present
dataset = dataset.remove_columns("overflow_to_sample_mapping")

# Split dataset into training and testing
dataset = dataset.train_test_split(test_size=0.2)

# Prepare DataCollator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Masked Language Modeling (MLM) is False for GPT-like models
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=10
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    data_collator=data_collator
)

# Train the model
trainer.train()