import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class BaseLinkRegressor(nn.Module):
    """Parent class for all baseline models, defines common decoder."""
    def __init__(self, hidden_dim):
        super(BaseLinkRegressor, self).__init__()
        # Common edge regressor: [Node_A_Emb, Node_B_Emb] -> Score
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def decode(self, z, edge_label_index):
        # z: embeddings of all nodes
        src, dst = edge_label_index[0], edge_label_index[1]
        z_src = z[src]
        z_dst = z[dst]
        # Concatenate features
        cat_feat = torch.cat([z_src, z_dst], dim=-1)
        # MLP prediction
        h = F.relu(self.lin1(cat_feat))
        out = self.lin2(h)
        return torch.sigmoid(out).view(-1)

# ==================== 1. GCN Model ====================
class GCN_Predictor(BaseLinkRegressor):
    def __init__(self, num_nodes, embedding_dim, hidden_dim):
        super(GCN_Predictor, self).__init__(hidden_dim)
        self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def encode(self, edge_index, edge_weight):
        x = self.node_emb.weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def forward(self, edge_index, edge_weight, predict_edges):
        z = self.encode(edge_index, edge_weight)
        return self.decode(z, predict_edges)

# ==================== 2. GraphSAGE Model ====================
class SAGE_Predictor(BaseLinkRegressor):
    def __init__(self, num_nodes, embedding_dim, hidden_dim):
        super(SAGE_Predictor, self).__init__(hidden_dim)
        self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        # SAGE usually does not use edge weights, or handles them differently; simplified to mean aggregation here
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def encode(self, edge_index):
        x = self.node_emb.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, edge_index, edge_weight, predict_edges):
        # SAGE ignores edge_weight (or you can check documentation on how to support weights)
        z = self.encode(edge_index)
        return self.decode(z, predict_edges)

# ==================== 3. Pure MLP Model (ignores graph structure) ====================
class MLP_Predictor(BaseLinkRegressor):
    """Only uses node ID to learn embeddings, completely ignores graph topology, used to test whether graph structure is really useful."""
    def __init__(self, num_nodes, embedding_dim, hidden_dim):
        super(MLP_Predictor, self).__init__(hidden_dim)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim) # Map directly to hidden

    def forward(self, edge_index, edge_weight, predict_edges):
        # Ignore edge_index, predict directly with embeddings
        z = self.node_emb.weight
        return self.decode(z, predict_edges)