import os
import torch
import evaluate
import numpy as np
from datasets import load_dataset


class Utils:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.accuracy = evaluate.load("accuracy")
    
    @staticmethod
    def get_device():
        return 'mps' if torch.backends.mps.is_built() else \
            'cuda' if torch.cuda.is_available() else \
            'cpu'
    
    def tokenize_text(self, batch):
        return self.tokenizer(batch["text"], truncation=True, padding=True)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)
    
    def get_tokenized_data(self, ):
        
        imdb_dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join("test", "test_input", "train.csv"),
                "validation": os.path.join("test", "test_input", "val.csv"),
                "test": os.path.join("test", "test_input", "test.csv"),
            },
        )

        print("Tokenizer input max length:", self.tokenizer.model_max_length)
        print("Tokenizer vocabulary size:", self.tokenizer.vocab_size)

        imdb_tokenized = imdb_dataset.map(self.tokenize_text, batched=True)
        del imdb_dataset
        
        return imdb_tokenized
