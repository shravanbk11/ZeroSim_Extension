import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_device_name(device_name, pin_type):
    # Remove trailing underscore and digits, e.g., "pmos_0" becomes "pmos"
    base = re.sub(r'_\d+$', '', device_name)
    # Append the pin type so that the final token reflects the pin.
    return f"{base}_{pin_type}"


def normalize_param_list(param_list, device):
    """
    Normalize a parameter list according to the expected parameter type.
    
    For a 3-element list, assume [W, L, M] with ranges:
      - W: 0.2 to 10
      - L: 0.13 to 5.0
      - M: 1 to 3200

    For a 1-element list, determine which parameter it is from the device:
      - If device == "capacitor_idx": C, range 1e-12 to 100e-12
      - If device == "resistor_idx": R, range 100 to 1e6
      - If device == "Ib": Ib, range 1e-6 to 4e-5
      
    If the device is not recognized, return the raw value.
    """
    if len(param_list) == 3:
        # [W, L, M]
        W, L, M = param_list
        norm_W = (W - 0.2) / (10 - 0.2)
        norm_L = (L - 0.13) / (5.0 - 0.13)
        norm_M = (M - 1) / (3200 - 1)
        return [norm_W, norm_L, norm_M]
    elif len(param_list) == 1:
        if device.startswith("capacitor"):
            # Parameter is C.
            C = param_list[0]
            norm_C = (C - 1e-12) / ((100e-12) - 1e-12)
            return [norm_C]
        elif device.startswith("resistor"):
            # Parameter is R.
            R = param_list[0]
            norm_R = (R - 100) / (1e6 - 100)
            return [norm_R]
        elif device == "Ib":
            # Parameter is Ib.
            Ib = param_list[0]
            norm_Ib = (Ib - 1e-6) / (4e-5 - 1e-6)
            return [norm_Ib]
        else:
            return param_list
    else:
        return param_list


class CircuitDataset(Dataset):
    def __init__(self, topology_file, sample_files, max_nodes=100, pad_device_token="<PAD>", device2idx=None,
        performance_mean=None, performance_std=None):
        """
        Args:
            topology_file (str): Path to the JSON file containing topology data.
            sample_files (list or str): A list of paths (or a single path) to the JSON files with samples.
            max_nodes (int): The fixed number of nodes per graph. Graphs with fewer nodes are padded;
                             graphs with more nodes are truncated.
            pad_device_token (str): Special token used for padding node device tokens.
        """
        # Load topology data. Support both a list or a dict with a key "topologies".
        with open(topology_file, 'r') as f:
            topology_data = json.load(f)
        if isinstance(topology_data, list):
            topologies_list = topology_data
        else:
            topologies_list = topology_data.get("topologies", [])
        
        # Build mapping from topology_id to topology information.
        self.topologies = {topo["topology_id"]: topo for topo in topologies_list}
        
        # Load samples data.
        if isinstance(sample_files, str):
            sample_files = [sample_files]
        self.samples = []
        for file in sample_files:
            with open(file, 'r') as f:
                sample_data = json.load(f)
            if isinstance(sample_data, list):
                self.samples.extend(sample_data)
            else:
                self.samples.extend(sample_data.get("samples", []))
        
        self.max_nodes = max_nodes
        
        # If no vocabulary is provided, build it from the topology data.
        if device2idx is None:
            device_set = set()
            for topo in self.topologies.values():
                nodes = topo["netlist"]["nodes"]
                for node in nodes:
                    normalized = normalize_device_name(node["device"], node["pin_type"])
                    device_set.add(normalized)
            
            # Reserve index 0 for padding.
            self.device2idx = {pad_device_token: 0}
            for device in sorted(device_set):
                self.device2idx[device] = len(self.device2idx)
        else:
            self.device2idx = device2idx
        
        # Determine the order of performance metrics from the first sample.
        if len(self.samples) > 0:
            self.metric_keys = sorted(list(self.samples[0]["performance"].keys()))
        else:
            self.metric_keys = []

        # ----- Compute normalization statistics for performance metrics -----
        # Compute or load normalization statistics for performance metrics.
        if performance_mean is not None and performance_std is not None:
            self.performance_mean = np.array(performance_mean)
            self.performance_std = np.array(performance_std)
        else:
            # Build a list of performance vectors.
            performance_list = []
            for sample in self.samples:
                performance_list.append([sample["performance"][key] for key in self.metric_keys])
            performance_array = np.array(performance_list)  # shape: [num_samples, num_metrics]
            # Compute mean and std over the samples for each metric.
            self.performance_mean = performance_array.mean(axis=0)
            self.performance_std = performance_array.std(axis=0)
            self.performance_median = np.median(performance_array, axis=0)
            # Avoid division by zero.
            self.performance_std[self.performance_std == 0] = 1e-12

            print("Performance Metrics Normalization Stats:")
            for i, key in enumerate(self.metric_keys):
                print(f" {key}: mean = {self.performance_mean[i]:.6g}, std = {self.performance_std[i]:.6g}, median = {self.performance_median[i]:.6g}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        topology_id = sample["topology_id"]
        topology = self.topologies[topology_id]
        netlist = topology["netlist"]
        nodes_info = netlist["nodes"]
        actual_num_nodes = len(nodes_info)
        
        # If there are more nodes than max_nodes, truncate.
        if actual_num_nodes > self.max_nodes:
            nodes_info = nodes_info[:self.max_nodes]
            actual_num_nodes = self.max_nodes
        
        # Convert node device strings to integer tokens.
        node_tokens = [self.device2idx.get(normalize_device_name(node["device"], node["pin_type"]), self.device2idx["<PAD>"]) for node in nodes_info]
        # Pad the node tokens list if necessary.
        if len(node_tokens) < self.max_nodes:
            pad_length = self.max_nodes - len(node_tokens)
            node_tokens.extend([self.device2idx["<PAD>"]] * pad_length)
        
        # --- Process parameter data ---
        params_dict = sample["parameters"]
        max_param_len = 3
        # Build the parameter matrix for the nodes.
        params_matrix = []
        for i in range(actual_num_nodes):
            device = nodes_info[i]["device"]    # device type for node i
            if i in params_dict:
                param_list = params_dict[i]
            else:
                # If the sample did not include parameters for this node, add it first.
                param_list = [0.0] * 1
            # Normalize the parameter list.
            norm_params = normalize_param_list(param_list, device)
            # Pad parameter list to have length max_param_len.
            if len(norm_params) < max_param_len:
                norm_params = norm_params + [0.0] * (max_param_len - len(norm_params))
            params_matrix.append(norm_params)
        # If the number of nodes is less than max_nodes, pad with rows of zeros.
        if actual_num_nodes < self.max_nodes:
            for _ in range(self.max_nodes - actual_num_nodes):
                params_matrix.append([0.0] * max_param_len)
        
        # --- Process the adjacency matrix ---
        adj_matrix = netlist["adjacency_matrix"]
        # Truncate (if needed) to match the number of nodes actually used.
        adj_matrix = [row[:actual_num_nodes] for row in adj_matrix[:actual_num_nodes]]
        # Pad the adjacency matrix to size (max_nodes x max_nodes) with zeros.
        adj_padded = np.zeros((self.max_nodes, self.max_nodes), dtype=float)
        adj_padded[:actual_num_nodes, :actual_num_nodes] = np.array(adj_matrix)
        
        # --- Process performance metrics ---
        # Create a vector in the order of self.metric_keys.
        performance = np.array([sample["performance"][key] for key in self.metric_keys])
        # Normalize using the provided or computed statistics.
        performance_norm = (performance - self.performance_mean) / self.performance_std
        performance_tensor = torch.tensor(performance_norm, dtype=torch.float)  # shape: [num_metrics]
        
        # Convert everything to torch tensors.
        nodes_tensor = torch.tensor(node_tokens, dtype=torch.long)                       # shape: [max_nodes]
        params_tensor = torch.tensor(params_matrix, dtype=torch.float)                   # shape: [max_nodes, max_param_len]
        attn_mask_tensor = torch.tensor(adj_padded, dtype=torch.float)                   # shape: [max_nodes, max_nodes]
        
        return {
            "nodes": nodes_tensor,
            "params": params_tensor,
            "attn_mask": attn_mask_tensor,
            "performance": performance_tensor
            # "topology_id": topology_id,
            # "sample_id": sample["sample_id"]
        }
