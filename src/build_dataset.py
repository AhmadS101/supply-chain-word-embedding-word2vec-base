import pandas as pd
import ast
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


# Read the target and context word indices from CSV files
target_indices = pd.read_csv("../data/processed/target_indices.csv")[
    "target_index"
].tolist()
context_indices = (
    pd.read_csv("../data/processed/context_indices.csv")["context_indices"]
    .apply(ast.literal_eval)
    .tolist()
)

# Convert indices to PyTorch tensors
target_tensor = torch.tensor(target_indices, dtype=torch.long)

# convert context indices to tensor list then apply padding to context tensors
context_tensor = [
    torch.tensor(context_word, dtype=torch.long) for context_word in context_indices
]
context_tensor = pad_sequence(context_tensor, batch_first=True, padding_value=0)


# Define the dataset class
class EmbeddingDataset(Dataset):
    def __init__(self, target_tensor, context_tensor):
        self.target_tensor = target_tensor
        self.context_tensor = context_tensor

    def __len__(self):
        return len(self.target_tensor)

    def __getitem__(self, idx):
        return self.target_tensor[idx], self.context_tensor[idx]


# Create the dataset
dataset = EmbeddingDataset(target_tensor, context_tensor)


# Define the DataLoader
dataset_loader = DataLoader(dataset, batch_size=32, shuffle=True)
