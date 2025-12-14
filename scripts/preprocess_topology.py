import json
import argparse
from collections import defaultdict
"""
topologies = [
    {
        "topology_name": "topology name"
        "topology_id": 1,
        "netlist": {
            "nodes": [{'id': 6, 'device': 'pmos_0', 'pin_type': 'd'}],
            "edges": [[0, 1], [1, 2], ...],
            "adjacency_matrix": adj_matrix,
            "device_to_pin_ids": device_to_pin_ids
        }
    },
    {
        "topology_id": 2,
        ...
    }
]
"""


# --- Helper: Create a node dictionary for a pin ---
def create_pin_node(device, pin_type, net, node_id_counter):
    node = {
        "id": node_id_counter,
        "device": device,
        "pin_type": pin_type,
        "net": net
    }
    node_id_counter += 1
    return node, node_id_counter


def create_adjacency_matrix(n_nodes, edges):
    # Initialize adjacency matrix with zeros
    adj_matrix = [[0] * n_nodes for _ in range(n_nodes)]
    # Fill in the adjacency matrix
    for i, j in edges:
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1
    return adj_matrix


# --- Process the netlist line by line ---
def process_netlist(netlist_str, transistor_pin_types, twopin_pin_types):
    # --- Initialize data structures ---
    nodes = []   # List of all pin nodes (each will have a unique id)
    edges = []   # List of edges (connections between pin nodes)
    device_to_pin_ids = {}  # Map from devices to list of pin node ids
    node_id_counter = 0  # Counter for generating unique node ids
    nmos_count, pmos_count, capacitor_count, resistor_count = 0, 0, 0, 0
    voltage_count = 0

    # process the .subckt line to get the ports
    subckt_ports = []
    for line in netlist_str.splitlines():
        line = line.strip()
        if line.startswith(".subckt"):
            tokens = line.split()
            ports = tokens[2:]
            subckt_ports = ports
            break
    subckt_ports = [port.lower() for port in subckt_ports]
    
    for port in subckt_ports:
        node = {
            "id": node_id_counter,
            "device": port,
            "pin_type": "port"
        }
        nodes.append(node)
        device_to_pin_ids[port] = [node_id_counter]
        node_id_counter += 1

    # process the device lines
    for line in netlist_str.splitlines():
        # Skip empty or control lines (.subckt, .ends, etc.)
        if not line or line.startswith(".") or line.startswith("*"):
            continue
        tokens = line.split()
        if not tokens:
            continue
        if 'I' in tokens[0]:
            subckt_ports.append("ib")
            node = {
                "id": node_id_counter,
                "device": "ib",
                "pin_type": "port"
            }
            nodes.append(node)
            device_to_pin_ids["ib"] = [node_id_counter]
            node_id_counter += 1
            continue

    # process the device lines
    for line in netlist_str.splitlines():
        # Skip empty or control lines (.subckt, .ends, etc.)
        if not line or line.startswith(".") or line.startswith("*"):
            continue
        tokens = line.split()
        if not tokens:
            continue

        tokens = [t.lower() for t in tokens]
        # Determine device type
        if len(tokens) == 9:
            if tokens[5] == "sky130_fd_pr__pfet_01v8":
                device = "pmos_" + str(pmos_count)
                pmos_count += 1
            else:
                device = "nmos_" + str(nmos_count)
                nmos_count += 1
        else:
            if 'c' in tokens[0]:
                device = "capacitor_" + str(capacitor_count)
                capacitor_count += 1
            elif 'r' in tokens[0]:
                device = "resistor_" + str(resistor_count)
                resistor_count += 1
            elif 'v' in tokens[0]:
                device = "voltage_" + str(voltage_count)
                voltage_count += 1

        # For transistors: 4 pins; for others (e.g., R*, C*): 2 pins.
        if device[:4] in {"pmos", "nmos"}:
            num_pins = 4
            pin_types = transistor_pin_types
        else:
            num_pins = 2
            pin_types = twopin_pin_types

        # Extract the nets for the pins.
        tokens = ["ib" if "ib" in s else s for s in tokens]
        nets = tokens[1:1+num_pins]
        # Create a list to hold the pin node IDs for this device.
        pin_ids = []
        for pt, net in zip(pin_types, nets):
            node, node_id_counter = create_pin_node(device, pt, net, node_id_counter)
            nodes.append(node)
            pin_ids.append(node["id"])
        device_to_pin_ids[device] = pin_ids

    # --- Create internal edges: fully connect all pins of the same device ---
    for device, pin_ids in device_to_pin_ids.items():
        for i in range(len(pin_ids)):
            for j in range(i+1, len(pin_ids)):
                edges.append([pin_ids[i], pin_ids[j]])

    # --- Create external edges: connect pins (from different devices) that share the same net ---
    net_to_node_ids = defaultdict(list)
    for node in nodes:
        if "net" in node:
            net_to_node_ids[node["net"]].append(node["id"])
            del node['net']

    id_to_node = {node["id"]: node for node in nodes}

    for net, node_ids in net_to_node_ids.items():
        # If the net is a port node, connect the port node to all other nodes on the net.
        if net in subckt_ports:
            for node_id in node_ids:
                edges.append([device_to_pin_ids[net][0], node_id])
        else:
            # Connect each pair of pins with the same net if they belong to different devices.
            for i in range(len(node_ids)):
                for j in range(i+1, len(node_ids)):
                    # To check if they are from different devices, look up the instance names.
                    node_i = id_to_node[node_ids[i]]
                    node_j = id_to_node[node_ids[j]]
                    if node_i["device"] != node_j["device"]:
                        # To avoid duplicates, we check if an edge already exists.
                        if not any((e[0] == node_ids[i] and e[1] == node_ids[j]) or
                                (e[0] == node_ids[j] and e[1] == node_ids[i])
                                for e in edges):
                            edges.append([node_ids[i], node_ids[j]])

    # Sort edges based on the first element of each pair
    sorted_edges = sorted(edges, key=lambda pair: pair[0])
    return nodes, sorted_edges, device_to_pin_ids


def main():
    # Set up argparse to get the netlist filename from the command line.
    parser = argparse.ArgumentParser(description="Process a netlist file to build a topology JSON.")
    parser.add_argument("netlist_file", type=str, help="A file contains all netlist file paths")
    parser.add_argument("output_filename", type=str, help="Output file for the JSON topology (e.g., topology.json)")
    args = parser.parse_args()

    # --- Settings for device types ---
    transistor_pin_types = ["d", "g", "s", "b"]
    twopin_pin_types = ["p", "n"]

    # We'll accumulate each topology in a list
    topologies = []

    with open(args.netlist_file, "r") as f:
        netlist_filenames = [line.strip() for line in f]

    # Process each netlist file
    for idx, netlist_filename in enumerate(netlist_filenames, start=1):
        with open(netlist_filename, "r") as f:
            netlist_str = f.read()

        # Process the netlist
        nodes, edges, device_to_pin_ids = process_netlist(netlist_str, transistor_pin_types, twopin_pin_types)
        adj_matrix = create_adjacency_matrix(len(nodes), edges)

        # _Pin_3.txt
        topology_name = netlist_filename[netlist_filename.rfind('/') + 1:-10]

        topology = {
            "topology_name": topology_name,
            "topology_id": idx,
            "netlist": {
                "nodes": nodes,    # List of all pin nodes {'id': 6, 'device': 'pmos_0', 'pin_type': 'd'}
                "edges": edges,    # List of edges between pin nodes [[0, 1], [1, 2], ...]
                "adjacency_matrix": adj_matrix,  # Adjacency matrix of the graph
                "device_to_pin_ids": device_to_pin_ids  # Map from devices to list of pin node ids
            }
        }
        topologies.append(topology)
    
    # Write all topologies to the output JSON file
    output_data = {
        "topologies": topologies
    }
    # Save the resulting topology JSON to the specified file.
    with open(args.output_filename, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Topology saved to {args.output_filename}")


if __name__ == "__main__":
    main()


