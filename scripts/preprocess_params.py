import json
import re
import os
import argparse
"""
samples = [
    {
        "sample_id": "000000",
        "topology_id": 1,
        "parameters": {
            "0 (node_idx)": [L, W, M]
            "1": [...]
        },
        "performance": {"metric1": 0.95, "metric2": 0.80}
    },
    {
        "sample_id": "000001",
        "topology_id": 2,
        "parameters": {
            "0": [L, W, M]
            "1": [...]
        },
        "performance": {"metric1": 0.90, "metric2": 0.85}
    }
]
"""

def read_params(file_name):
    # Read entire file
    with open(file_name, 'r') as f:
        text = f.read()
    
    text = text.replace('\n', ' ').replace('+', ' ')
    # Split by whitespace into tokens
    tokens = re.split(r'\s+', text.strip())

    params_dict = {}
    
    for token in tokens:        
        if '=' in token:
            name, value_str = token.split('=', 1)
            name = name.lower()
            value = float(value_str)
            # Store in dictionary
            params_dict[name] = value

    return params_dict


# --- Build instance parameters from netlist ---
def process_netlist(netlist_str, device_to_pin_ids):
    # only consider the devices 
    sample_parameters = {}  # mapping from node idx order to parameter list
    pins = list(device_to_pin_ids.values())
    
    sample_parameters[pins[4][0]] = ["PARAM_CLOAD".lower()]
    sample_parameters[pins[5][0]] = ["CURRENT_0_BIAS".lower()]

    device_idx = 6
    for line in netlist_str.splitlines():
        line = line.strip()
        # Skip empty or control lines
        if not line or line.startswith(".") or line.startswith("*"):
            continue
        tokens = line.split()

        cur_params = []
        if len(tokens) == 9:
            for token in tokens:        
                if '=' in token:
                    name, value_str = token.split('=', 1)
                    if value_str.startswith("'"):
                        value_str = value_str[1:-1]
                    cur_params.append(value_str.lower())
        else:
            if '=' in tokens[-1]:
                name, value_str = tokens[-1].split('=', 1)
                cur_params.append(value_str.lower())
            else:
                cur_params.append(tokens[-1].lower())
        
        # assign cur_params to corresponding nodes
        for node_idx in pins[device_idx]:
            sample_parameters[node_idx] = cur_params
        device_idx += 1
        
    return sample_parameters


def extract_param_value(file_path, param_name):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regex to find the parameter assignment (e.g., .PARAM PARAM_CLOAD =100.00p)
    match = re.search(rf"\.PARAM\s+{param_name}\s*=\s*([\d.]+)([pnumkMG]*)", content, re.IGNORECASE)
    
    if match:
        value, unit = match.groups()
        value = float(value)
        
        # Convert unit to scientific notation
        unit_multipliers = {
            'p': 1e-12,  # pico
            'n': 1e-9,   # nano
            'u': 1e-6,   # micro
            'm': 1e-3,   # milli
            'k': 1e3,    # kilo
            'M': 1e6,    # mega
            'G': 1e9     # giga
        }
        multiplier = unit_multipliers.get(unit, 1)
        numerical_value = value * multiplier

        return numerical_value
    else:
        return None


# --- Match the current params to each pin
def transform_params(param_struc, cur_params_dict, param_cload):
    """
    For each entry in dict1, look up the corresponding numeric value
    in dict2. Handle optional multipliers like '*4' or '4*' around 
    the MOSFET string, and return a new dictionary with numeric values.
    """
    sample_params = {}
    
    for key, param_list in param_struc.items():
        numeric_params = []
        for param in param_list:
            if '*' in param:
                parts = param.split('*')
                if parts[0].isdigit():
                    numeric_params.append(float(parts[0]) * cur_params_dict[parts[1]])
                else:
                    numeric_params.append(float(parts[1]) * cur_params_dict[parts[0]])
            else:
                if param in cur_params_dict:
                    numeric_params.append(cur_params_dict[param])
                else:
                    numeric_params.append(param_cload)

        sample_params[key] = numeric_params
    return sample_params


def main():
    # Set up argparse to get the netlist filename from the command line.
    parser = argparse.ArgumentParser(description="Process a netlist file to build a topology JSON.")
    parser.add_argument("--topology_filename", type=str, help="Topology json file")
    parser.add_argument("--n_sample", type=int, help="The number of samples for each topology to use")
    parser.add_argument("--netlist_path", type=str, help="The root of netlists and cirs")
    parser.add_argument("--dataset_path", type=str, help="The root of all samples stored")
    parser.add_argument("--output_path", type=str, help="The root of output files for the JSON samples")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    performance_file = "performance.json"
    params_file = "_vars.spice"

    # get the device_to_pin_id from the topology.json
    with open(args.topology_filename, "r") as f:
        data = json.load(f)
        # The top-level key we created is "topologies" (a list)
        topologies = data["topologies"]

    for i in range(len(topologies)):
        samples = []
        topology = topologies[i]
        topology_name = topology["topology_name"]
        topo_idx = topology["topology_id"]
        device_to_pin_ids = topology["netlist"]["device_to_pin_ids"]
        output_filename = os.path.join(args.output_path, topology_name + "_samples.json")

        # get the netlist_str
        with open(os.path.join(args.netlist_path, topology_name + "_Pin_3.txt"), "r") as f:
            netlist_str = f.read()

        # preprocess the netlist_str:
        param_struc = process_netlist(netlist_str, device_to_pin_ids)
        # get the 'param_cload' form the .cir file
        param_cload = extract_param_value(os.path.join(args.netlist_path, topology_name + "_ACDC.cir"), 'PARAM_CLOAD')

        # read the params and performance
        for j in range(args.n_sample):
            cur_params_file = os.path.join(args.dataset_path, topology_name, "sample_" + str(j), topology_name + params_file)
            cur_performance_file = os.path.join(args.dataset_path, topology_name, "sample_" + str(j), performance_file)

            # read the params from the cur_params_file
            cur_params_dict = read_params(cur_params_file)
            sample_params = transform_params(param_struc, cur_params_dict, param_cload)

            with open(cur_performance_file, "r") as f:
                sample_performance = json.load(f)

            # read the performance metrcis
            sample = {
                "sample_id": j,
                "topology_id": topo_idx,
                "parameters": sample_params,
                "performance": sample_performance
            }
            samples.append(sample)

        # Write all topologies to the output JSON file
        output_data = {
            "samples": samples
        }

        # Save the resulting topology JSON to the specified file.
        with open(output_filename, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"{topology_name} samples saved to {output_filename}")


if __name__ == "__main__":
    main()
