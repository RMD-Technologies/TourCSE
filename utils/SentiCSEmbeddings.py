import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

class SentiCSEmbeddings():
    def __init__(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        
    def encode(self, texts: list[str], **kwargs):
        features = []
        with torch.no_grad():
            for text in tqdm(texts):
                inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                # Extract the pooler output and move it to CPU
                output = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
                features.append(output.cpu().numpy())
        
        # Stack the features correctly
        return np.vstack(features)  # Stack features into a single NumPy array