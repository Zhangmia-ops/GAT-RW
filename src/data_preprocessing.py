import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import os
import json

def build_node_mapping(graph_file):
    """
    Stage 1.1: Scan the original graph file, extract all unique nodes, and build bidirectional mapping dictionary M_node
    """
    unique_nodes = set()
    with open(graph_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                unique_nodes.add(int(parts[0]))
                unique_nodes.add(int(parts[1]))
                
    sorted_nodes = sorted(list(unique_nodes))
    # Forward mapping: original_id -> standardized_id [0, |V|-1]
    node_map = {orig_id: std_id for std_id, orig_id in enumerate(sorted_nodes)}
    # Reverse mapping: standardized_id -> original_id
    reverse_node_map = {std_id: orig_id for std_id, orig_id in enumerate(sorted_nodes)}
    
    return node_map, reverse_node_map, len(sorted_nodes)

def process_raw_graph(graph_file, node_map):
    """
    Stage 1.2: Convert the edges of the original graph to standardized index space E_std
    """
    std_edges = []
    with open(graph_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                std_edges.append((node_map[u], node_map[v], w))
    return std_edges

def process_query_pairs(query_file, node_map):
    """
    Stage 1.3: Convert target query node pairs and filter out OOV (Out-Of-Vocabulary) nodes not in the graph
    """
    std_queries = []
    filtered_out_count = 0
    with open(query_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                # Strict OOV check
                if u in node_map and v in node_map:
                    std_queries.append((node_map[u], node_map[v]))
                else:
                    filtered_out_count += 1
                    
    print(f"Query pairs loaded. Successfully mapped: {len(std_queries)} pairs, filtered OOV node pairs: {filtered_out_count} pairs.")
    return std_queries

def decompose_and_distribute(std_edges, std_queries, num_nodes, output_dir="inference_data"):
    """
    Stage 2: Connectivity Decomposition and Query Distribution
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Use scipy to quickly extract connected components
    row = np.array([e[0] for e in std_edges])
    col = np.array([e[1] for e in std_edges])
    data = np.array([e[2] for e in std_edges])
    
    # Build sparse matrix
    adj_matrix = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # Find connected components of the undirected graph
    num_components, component_labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    print(f"Graph connectivity check completed: found {num_components} independent connected components (Subgraphs).")

    # 2. Group nodes by connected component
    component_to_nodes = {k: set() for k in range(num_components)}
    for node_idx, comp_label in enumerate(component_labels):
        component_to_nodes[comp_label].add(node_idx)

    # 3. Distribute edges to corresponding subgraphs
    subgraph_edges = {k: [] for k in range(num_components)}
    for u, v, w in std_edges:
        label_u = component_labels[u]
        # Only assign to the same component if they belong together
        if label_u == component_labels[v]:
            subgraph_edges[label_u].append((u, v, w))

    # 4. Distribute query pairs to corresponding subgraphs (cross-component unreachable pairs are automatically filtered)
    subgraph_queries = {k: [] for k in range(num_components)}
    cross_component_queries = 0
    for u, v in std_queries:
        label_u = component_labels[u]
        if label_u == component_labels[v]:
            subgraph_queries[label_u].append((u, v))
        else:
            cross_component_queries += 1
            
    print(f"Query distribution completed. Unpredictable pairs across disconnected regions: {cross_component_queries} pairs (safely filtered).")

    # 5. Save the split subgraphs and query pairs
    valid_subgraph_count = 0
    for k in range(num_components):
        # Optimization: ignore components with no edges and no queries (isolated nodes)
        if len(subgraph_edges[k]) == 0 and len(subgraph_queries[k]) == 0:
            continue
            
        valid_subgraph_count += 1
        edges_path = os.path.join(output_dir, f"subgraph_{valid_subgraph_count}_edges.txt")
        queries_path = os.path.join(output_dir, f"subgraph_{valid_subgraph_count}_queries.txt")
        
        with open(edges_path, 'w') as f:
            f.write(f"# Subgraph {valid_subgraph_count} Edges\n")
            for u, v, w in subgraph_edges[k]:
                f.write(f"{u},{v},{w}\n")
                
        with open(queries_path, 'w') as f:
            f.write(f"# Subgraph {valid_subgraph_count} Query Pairs\n")
            for u, v in subgraph_queries[k]:
                f.write(f"{u},{v}\n")

    print(f"Data processing completed. Saved {valid_subgraph_count} valid processing batches to directory '{output_dir}'.")
    # New addition
    component_index = {
    "node_to_component": {},
    "component_to_nodes": {}
    }
    for node_idx, comp_label in enumerate(component_labels):
        component_index["node_to_component"][str(node_idx)] = int(comp_label)

    for k, nodes in component_to_nodes.items():
        component_index["component_to_nodes"][str(k)] = list(nodes)
    index_path = os.path.join(output_dir, "component_index.json")
    with open(index_path, "w") as f:
        json.dump(component_index, f, indent=2)
    print(f"Component index saved to: {index_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw graph and query files for inference.")
    parser.add_argument("--raw-graph", required=True, help="Path to the raw graph file.")
    parser.add_argument("--raw-query", default="raw_queries_target.txt", help="Path to the raw query file.")
    parser.add_argument("--output-root", default="processed_inference_data", help="Root directory for processed output.")
    parser.add_argument("--mapping-file", default="M_node_dictionary.txt", help="Name of the node mapping file.")

    args = parser.parse_args()

    RAW_GRAPH_FILE = args.raw_graph
    RAW_QUERY_FILE = args.raw_query

    dataset_name = os.path.splitext(os.path.basename(RAW_GRAPH_FILE))[0]
    OUTPUT_DIRECTORY = os.path.join(args.output_root, dataset_name)
    MAPPING_FILE = os.path.join(OUTPUT_DIRECTORY, args.mapping_file)

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print("=== Start preprocessing data for inference ===")

    # Stage 1: topology standardization
    print("Stage 1: topology and query standardization...")
    node_map, reverse_node_map, num_unique_nodes = build_node_mapping(RAW_GRAPH_FILE)
    print(f"Extracted {num_unique_nodes} unique nodes.")

    # Save M_node dictionary for restoring prediction results
    with open(MAPPING_FILE, 'w') as f:
        f.write("# original_id, standardized_id\n")
        for orig, std in node_map.items():
            f.write(f"{orig},{std}\n")

    std_edges = process_raw_graph(RAW_GRAPH_FILE, node_map)
    std_queries = process_query_pairs(RAW_QUERY_FILE, node_map)

    # Stage 2: connectivity decomposition and query distribution
    print("\nStage 2: connectivity decomposition and query distribution...")
    decompose_and_distribute(std_edges, std_queries, num_unique_nodes, output_dir=OUTPUT_DIRECTORY)

    print("\n=== Preprocessing pipeline finished ===")