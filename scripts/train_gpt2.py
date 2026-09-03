import json
import zipfile
from pathlib import Path
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Paths
ZIP_PATH = Path("shakespear.zip")
DATA_PATH = Path("shakespeare_data.txt")
MODEL_PATH = Path("src/lib/shakespeare-gpt2")

# Extract Shakespeare data
def extract_shakespeare_data():
    print("Extracting Shakespeare data...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())
        
        if "alllines.txt" in names:
            raw = archive.read("alllines.txt").decode("utf-8", errors="ignore")
            lines = raw.splitlines()
        elif "Shakespeare_data.csv" in names:
            import csv
            raw = archive.read("Shakespeare_data.csv").decode("utf-8", errors="ignore").splitlines()
            reader = csv.DictReader(raw)
            lines = [row.get("PlayerLine", "") for row in reader]
        else:
            raise FileNotFoundError("Need alllines.txt or Shakespeare_data.csv")
    
    # Clean and write
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
    DATA_PATH.write_text("\n".join(cleaned_lines), encoding="utf-8")
    print(f"Extracted {len(cleaned_lines)} lines to {DATA_PATH}")

# Train model
def train_model():
    print("Loading pre-trained GPT-2...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=str(DATA_PATH),
        block_size=128,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_PATH),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Train
    print("Training...")
    trainer.train()
    
    # Save
    print(f"Saving model to {MODEL_PATH}...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    if not DATA_PATH.exists():
        extract_shakespeare_data()
    train_model()
