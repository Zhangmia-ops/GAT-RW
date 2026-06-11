# feature_builder.py (Spectral Edition)

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from collections import defaultdict

# Global adjacency matrix (sparse representation)
adj_matrix = defaultdict(dict)

def load_graph_from_file(filename):
    """
    Read graph file and build adjacency matrix.
    """
    global adj_matrix
    adj_matrix.clear()
    max_node_idx = -1

    print(f"  -> Reading graph file: {filename} ...")
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split(',')
                if len(parts) != 3: continue
                try:
                    u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                except ValueError: continue

                max_node_idx = max(max_node_idx, u, v)
                adj_matrix[u][v] = w
                adj_matrix[v][u] = w
        
        num_nodes = max_node_idx + 1
        print(f"  -> Graph loaded. Nodes: {num_nodes}, Edges: {sum(len(v) for v in adj_matrix.values())//2}")
        return num_nodes
    except Exception as e:
        print(f"File read error: {e}")
        raise

def compute_local_features(num_nodes):
    """Compute local features: degree, weighted degree, clustering coefficient"""
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    weighted_degree = torch.zeros(num_nodes, dtype=torch.float32)
    clustering = torch.zeros(num_nodes, dtype=torch.float32)

    for u in range(num_nodes):
        if u in adj_matrix:
            neighbors = adj_matrix[u]
            k = len(neighbors)
            degree[u] = k
            weighted_degree[u] = sum(neighbors.values())
            
            # Compute clustering coefficient
            if k >= 2:
                triangle_count = 0
                neighbor_list = list(neighbors.keys())
                for i in range(k):
                    for j in range(i + 1, k):
                        ni, nj = neighbor_list[i], neighbor_list[j]
                        if nj in adj_matrix[ni]:
                            triangle_count += 1
                clustering[u] = 2 * triangle_count / (k * (k - 1))
    
    return torch.stack([degree, weighted_degree, clustering], dim=1)

def compute_laplacian_positional_encoding(num_nodes, k=8):
    """
    [Core Improvement] Compute the smallest k non-trivial eigenvectors of the normalized Laplacian matrix.
    This gives each node a "geometric coordinate" in the graph.
    """
    print(f"  -> Computing Laplacian spectral features (k={k})... This provides global structural coordinate information")
    
    # 1. Build sparse adjacency matrix
    row_ind, col_ind, data = [], [], []
    for u, neighbors in adj_matrix.items():
        for v, w in neighbors.items():
            row_ind.append(u)
            col_ind.append(v)
            data.append(w) # Or use 1.0 to focus only on topology
            
    adj_csr = sp.csr_matrix((data, (row_ind, col_ind)), shape=(num_nodes, num_nodes))
    
    # 2. Build normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    degrees = np.array(adj_csr.sum(axis=1)).flatten()
    # Handle isolated nodes (degree 0)
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees!=0)
    d_inv_sqrt[degrees == 0] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    laplacian = sp.eye(num_nodes) - d_mat_inv_sqrt @ adj_csr @ d_mat_inv_sqrt
    
    # 3. Eigendecomposition
    # We need the smallest k+1 eigenvalues and corresponding eigenvectors (the first is usually constant, representing connectivity)
    # 'SM' = Smallest Magnitude
    try:
        # For stability, compute a bit more, then drop the first one (corresponding to eigenvalue close to 0)
        eigenvalues, eigenvectors = eigsh(laplacian, k=k+1, which='SM', tol=1e-2)
        
        # Sort eigenvalues (small to large)
        idx = eigenvalues.argsort()
        eigenvectors = eigenvectors[:, idx]
        
        # Discard the first eigenvector (usually near constant or all-zero, low information), keep the next k
        pe = torch.from_numpy(eigenvectors[:, 1:k+1]).float()
        
    except Exception as e:
        print(f"    [Warning] Eigendecomposition failed (graph may be too small or disconnected): {e}. Using random features instead.")
        pe = torch.randn(num_nodes, k)
        
    return pe

def build_node_features_with_clustering(num_nodes):
    """
    Feature engineering V3.0: local features + spectral positional encoding
    """
    if not adj_matrix and num_nodes > 0:
        print("Warning: adjacency matrix is empty.")

    # 1. Local features (3 dimensions)
    print("  -> Computing local features...")
    local_feats = compute_local_features(num_nodes)
    
    # 2. Global spectral features (8 dimensions)
    k_spectral = 8
    spectral_feats = compute_laplacian_positional_encoding(num_nodes, k=k_spectral)
    
    # 3. Concatenate
    features = torch.cat([local_feats, spectral_feats], dim=1)
    print(f"  -> Raw features built, total dimension: {features.shape[1]} (3 local + {k_spectral} spectral)")

    # 4. Z-score normalization (critical!)
    print("  -> Performing Z-score normalization...")
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std < 1e-6] = 1.0 # Avoid division by zero
    normalized_features = (features - mean) / std

    return normalized_features