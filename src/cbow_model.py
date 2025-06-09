import sys
import os
import torch
import torch.nn as nn

# Get absolute path to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)


from src.build_dataset import target_indices

embedding_dim = 300
vocab_size = len(target_indices)


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        context = self.embeddings(context)
        context = context.mean(dim=1)
        output = self.output(context)
        return output


torch.manual_seed(42)

# Initialize the model, loss function, and optimizer
model = CBOWModel(vocab_size, embedding_dim)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
