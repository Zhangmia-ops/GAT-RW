# UST_minus.py

import argparse
import os
import random
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def adjacency_to_edges(adj_matrix):
    """Convert sparse adjacency matrix (storing only non-zero edges) to edge list"""
    edges = []
    adj_matrix = adj_matrix.tocoo()
    for i, j, w in zip(adj_matrix.row, adj_matrix.col, adj_matrix.data):
        if i < j and w != 0:
            edges.append((w, i, j))
    return edges

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x
            return True
        return False

def kruskal_UST(n, edges):
    """Kruskal algorithm to generate random spanning tree (not sorted by weight, edges chosen randomly)"""
    edges_shuffled = edges.copy()
    random.shuffle(edges_shuffled)
    uf = UnionFind(n)
    UST_edges, discarded_edges = [], []
    processed_count = 0
    for edge in edges_shuffled:
        processed_count += 1
        w, u, v = edge
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            UST_edges.append((u, v, w))
        else:
            discarded_edges.append((u, v, w))
        if len(UST_edges) == n - 1:
            break
    if processed_count < len(edges_shuffled):
        remaining_edges_raw = edges_shuffled[processed_count:]
        reformatted_remaining = [(u, v, w) for w, u, v in remaining_edges_raw]
        discarded_edges.extend(reformatted_remaining)
    return UST_edges, discarded_edges

def generate_UST_from_adjacency(adj_matrix):
    n = adj_matrix.shape[0]
    edges = adjacency_to_edges(adj_matrix)
    UST_edges, discarded_edges = kruskal_UST(n, edges)
    return UST_edges, discarded_edges

def load_graph_from_file(filename, n):
    """Load undirected graph from edge list file, build sparse adjacency matrix"""
    adj = lil_matrix((n, n), dtype=np.float32)
    with open(filename, "r") as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) != 3: continue
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
            if u >= n or v >= n: raise ValueError(f"Node index out of range: u={u}, v={v} (n={n})")
            if u == v: continue
            adj[u, v], adj[v, u] = w, w
    return adj.tocsr()

def create_node_mapping(file_path):
    """Create mapping from original nodes to consecutive new indices (starting from 0)"""
    nodes = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',')
            if len(parts) < 3: continue
            u, v, _ = parts[:3]
            nodes.add(int(u)); nodes.add(int(v))
    sorted_nodes = sorted(nodes)
    return {node: idx for idx, node in enumerate(sorted_nodes)}

def remap_edgelist(input_path, output_path, node_map):
    """Generate new edge list file, node IDs starting from 0 consecutively"""
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        f_out.write("# Remapped Edge List (nodes start from 0)\n")
        for line in f_in:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',')
            if len(parts) < 3: continue
            u, v, weight = int(parts[0].strip()), int(parts[1].strip()), parts[2].strip()
            new_u, new_v = node_map.get(u), node_map.get(v)
            if new_u is not None and new_v is not None:
                f_out.write(f"{new_u},{new_v},{weight}\n")

def save_edges_to_file(edges_list, output_path, header="# u,v,weight\n"):
    """Process an edge list into u<v standard format, deduplicate, sort, and save to file."""
    processed_edges, seen = [], set()
    for item in edges_list:
        if len(item) != 3: continue
        u, v, w = item
        if isinstance(w, str): w = float(w)
        if u > v: u, v = v, u
        if (u, v) not in seen:
            processed_edges.append((u, v, w))
            seen.add((u, v))
    edges_sorted = sorted(processed_edges, key=lambda x: (x[0], x[1]))
    with open(output_path, "w") as f_out:
        if header: f_out.write(header)
        for u, v, w in edges_sorted:
            f_out.write(f"{u},{v},{w}\n")

def sort_edgelist(input_path, output_path):
    """Read edge list from file, then process and save using the generic function."""
    edges = []
    with open(input_path, "r") as f_in:
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split(",")
            if len(parts) < 3: continue
            try:
                u, v, w = int(parts[0]), int(parts[1]), parts[2].strip()
                edges.append((u, v, w))
            except ValueError: continue
    save_edges_to_file(edges, output_path)

def generate_graph_with_removed_edges(UST_edges, discarded_edges, num_to_remove, graph_output_path, removed_output_path):
    """Remove a specified number of non-Spanning-Tree_Backbone edges from the original graph, generate a base graph, and save the removed edges."""
    if num_to_remove < 0: raise ValueError("Number of edges to remove cannot be negative.")
    num_discarded_available = len(discarded_edges)
    if num_to_remove > num_discarded_available:
        print(f"Warning: number of edges to remove ({num_to_remove}) > available ({num_discarded_available}). Will remove all non-Spanning-Tree_Backbone edges.")
        num_to_remove = num_discarded_available
    
    shuffled_discarded = discarded_edges.copy()
    random.shuffle(shuffled_discarded)

    edges_that_were_removed = shuffled_discarded[:num_to_remove]
    edges_that_were_kept = shuffled_discarded[num_to_remove:]
    
    final_edges_for_graph = UST_edges + edges_that_were_kept
    
    save_edges_to_file(final_edges_for_graph, graph_output_path, header=f"# Training graph with {num_to_remove} edges removed.\n")
    save_edges_to_file(edges_that_were_removed, removed_output_path, header=f"# Test set: The {len(edges_that_were_removed)} edges that were removed.\n")

    print("-" * 20)
    print(f"Successfully generated training graph: {graph_output_path}")
    print(f"and saved test set: {removed_output_path}")
    print(f"Total edges in new graph: {len(final_edges_for_graph)}")
    print("-" * 20)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data preprocessing script for node remapping, Spanning-Tree_Backbone generation, and creating training/test sets after edge removal.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', required=True, help="Path to the original dataset, e.g., 'amazon_weight.txt'")
    parser.add_argument('--output-dir', required=True, help="Output directory for all preprocessed files, e.g., './preprocessed_data/amazon/'")
    parser.add_argument('--remove-list', type=int, nargs='+', default=[20, 50, 100, 200], 
                        help="List of edge removal counts, e.g., --remove-list 20 50")
    args = parser.parse_args()

    # --- Automated workflow ---

    # Step 1: Create all required subdirectories
    base_dir = args.output_dir
    base_files_dir = os.path.join(base_dir, "base_files")
    training_graphs_dir = os.path.join(base_dir, "training_graphs")
    test_sets_dir = os.path.join(base_dir, "test_sets")
    
    os.makedirs(base_files_dir, exist_ok=True)
    os.makedirs(training_graphs_dir, exist_ok=True)
    os.makedirs(test_sets_dir, exist_ok=True)

    # Step 2: Dynamically define all file paths
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # Temporary file, can be placed in main output directory
    remapped_file = os.path.join(base_dir, f"{base_name}_remapped_temp.txt")
    
    # Base file paths
    sorted_file = os.path.join(base_files_dir, f"{base_name}_sorted.txt")
    mapping_file = os.path.join(base_files_dir, f"{base_name}_mapping.txt")
    UST_file = os.path.join(base_files_dir, f"{base_name}_UST.txt")
    discarded_file = os.path.join(base_files_dir, f"{base_name}_discarded.txt")

    # Step 3: Execute preprocessing
    print(f"--- Processing: {args.input} ---")
    print("\nStep 3.1: Creating node mapping...")
    node_mapping = create_node_mapping(args.input)
    with open(mapping_file, "w") as f:
        f.write("original_id,new_id\n")
        for original_id, new_id in sorted(node_mapping.items()):
            f.write(f"{original_id},{new_id}\n")
    print(f"  -> Node mapping saved to: {mapping_file}")

    print("\nStep 3.2: Remapping and sorting edge list...")
    remap_edgelist(args.input, remapped_file, node_mapping)
    sort_edgelist(remapped_file, sorted_file)
    os.remove(remapped_file) # Delete temporary remapped file
    print(f"  -> Standardized full graph saved to: {sorted_file}")
    
    n = len(node_mapping)
    print(f"  -> Number of nodes: {n}")
    
    print("\nStep 3.3: Loading graph and generating Spanning-Tree_Backbone/non-Spanning-Tree_Backbone edges...")
    adj = load_graph_from_file(sorted_file, n)
    UST_edges, discarded_edges = generate_UST_from_adjacency(adj)
    print(f"  -> Spanning-Tree_Backbone edge count: {len(UST_edges)}")
    print(f"  -> Non-Spanning-Tree_Backbone edge (pool) count: {len(discarded_edges)}")

    # Step 4: Save base files to base_files/
    print("\nStep 4: Saving base files...")
    save_edges_to_file(UST_edges, UST_file, header="# u,v,weight (Spanning-Tree_Backbone edges)\n")
    print(f"  -> Spanning-Tree_Backbone edges saved to: {UST_file}")
    
    save_edges_to_file(discarded_edges, discarded_file, header="# u,v,weight (Non-Spanning-Tree_Backbone edges, the 'pool')\n")
    print(f"  -> Non-Spanning-Tree_Backbone edges (pool) saved to: {discarded_file}")

    # Step 5: Batch generate training graphs and test sets
    print(f"\nStep 5: Generating training graphs and test sets for removal counts {args.remove_list}...")
    for num_to_remove in args.remove_list:
        # Build full paths with subdirectories for training graph and test set
        graph_output_path = os.path.join(training_graphs_dir, f"graph_minus_{num_to_remove}_edges.txt")
        removed_output_path = os.path.join(test_sets_dir, f"graph_minus_{num_to_remove}_REMOVED.txt")
        
        generate_graph_with_removed_edges(
            UST_edges=UST_edges,
            discarded_edges=discarded_edges,
            num_to_remove=num_to_remove,
            graph_output_path=graph_output_path,
            removed_output_path=removed_output_path
        )

    print(f"\n✅ All preprocessing tasks for {args.input} completed!")