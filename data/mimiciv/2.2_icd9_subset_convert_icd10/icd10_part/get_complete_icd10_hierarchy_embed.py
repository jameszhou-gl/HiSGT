import seaborn as sns
from sklearn.manifold import TSNE
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv
import json
import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
import matplotlib.pyplot as plt


# Load the hierarchical graph
with open("data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/icd10_hierarchical_graph.json") as f:
    graph = json.load(f)

# Create node-to-ID mapping
node_to_id = {node: idx for idx, node in enumerate(graph.keys())}

# Create edges based on 'children' and 'parent' relationships
edges = []
for node, attributes in graph.items():
    for child in attributes.get("children", []):
        # Directed edge: parent -> child
        edges.append((node_to_id[node], node_to_id[child]))

# Convert edges to tensor format
# Transpose to shape [2, num_edges]
edge_index = torch.tensor(edges, dtype=torch.long).t()

# Initialize node features as identity (one-hot encoded IDs)
num_nodes = len(graph)
# Shape: [num_nodes, num_nodes] (Identity matrix)
node_features = torch.eye(num_nodes)

# Convert to PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index)


class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = BatchNorm(hidden_dim)  # Add normalization
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.norm2 = BatchNorm(embedding_dim)  # Add normalization

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.norm1(x)  # Apply normalization
        x = self.conv2(x, edge_index)
        x = self.norm2(x)  # Apply normalization
        return x


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, encoder):
        super(GraphAutoencoder, self).__init__()
        self.encoder = encoder

    def forward(self, x, edge_index):
        embeddings = self.encoder(x, edge_index)
        reconstructed_adj = torch.matmul(
            embeddings, embeddings.t())  # Adjacency reconstruction
        return embeddings, reconstructed_adj


# Convert edge_index to dense adjacency matrix
true_adj = to_dense_adj(data.edge_index)[0]  # Shape: [num_nodes, num_nodes]

# Initialize model and optimizer
input_dim = num_nodes  # Same as number of nodes (identity features)
print(f'input_dim: {input_dim}')
hidden_dim = 768
embedding_dim = 384
encoder = GNNEncoder(input_dim, hidden_dim, embedding_dim)
model = GraphAutoencoder(encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    embeddings, reconstructed_adj = model(data.x, data.edge_index)
    loss = F.mse_loss(torch.sigmoid(reconstructed_adj), true_adj)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Shape: [num_nodes, embedding_dim]
    embeddings = model.encoder(data.x, data.edge_index)
    node_to_id_serializable = list(node_to_id.items())
    # Save embeddings and node_to_id to an npz file
    np.savez("data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/complete_icd10_hierarchy_embed.npz",
            embeddings=embeddings.detach().numpy(),
            node_to_id=node_to_id_serializable)
