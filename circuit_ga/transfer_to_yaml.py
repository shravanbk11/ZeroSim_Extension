#!/usr/bin/env python3
 
import re
import sys
import yaml
from yaml.representer import SafeRepresenter
from collections import OrderedDict

###############################################################################
# 1) A custom class for items we want in flow style
###############################################################################
class FlowItem(list):
    """A list subclass that we'll force to render in flow (inline) style."""
    pass

###############################################################################
# 2) Register a representer that forces "FlowItem" objects into flow style
###############################################################################
def represent_flow_item(dumper, data):
    """
    data is your FlowItem object (a list). We force it to serialize as [ ... ].
    """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# Register the representer with PyYAML
yaml.add_representer(FlowItem, represent_flow_item)

###############################################################################
# 3) Parse netlist
###############################################################################
def parse_spice_netlist(netlist_path, ib_entry_needed=True):
    xm_regex = re.compile(
        r"^\s*xm(\d+)\s+[\w\(\)\[\]$\.]+\s+[\w\(\)\[\]$\.]+\s+[\w\(\)\[\]$\.]+\s+[\w\(\)\[\]$\.]+\s+sky130_fd_pr__(p|n)fet_01v8",
        re.IGNORECASE
    )
    type_map = {'p': 'pfet_01v8', 'n': 'nfet_01v8'}

    c_regex = re.compile(r"^\s*C(\d+)\s", re.IGNORECASE)
    v_regex = re.compile(r"^\s*V(\d+)\s", re.IGNORECASE)
    i_regex = re.compile(r"^\s*I(\d+)\s", re.IGNORECASE)

    ckt_hierarchy = []
    op_info = {}

    with open(netlist_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty or comment lines
            if not line or line.startswith('*'):
                continue

            # 1) Transistor
            xm_match = xm_regex.match(line)
            if xm_match:
                index_str = xm_match.group(1)
                pfet_nfet = xm_match.group(2)
                device_type = type_map[pfet_nfet]

                device_name = f"M{index_str}"
                instance_name = f"x1.XM{index_str}"

                ckt_hierarchy.append([device_name, instance_name, device_type, "m"])
                op_info[device_name] = {}
                continue

            # 2) Capacitor
            c_match = c_regex.match(line)
            if c_match:
                c_index_str = c_match.group(1)
                cap_name = f"C{c_index_str}"
                ckt_hierarchy.append([cap_name, "x1", cap_name, "c"])
                op_info[cap_name] = {}
                continue

            # 2) Voltage source
            v_match = v_regex.match(line)
            if v_match:
                v_index_str = v_match.group(1)
                vol_name = f"V{v_index_str}"
                ckt_hierarchy.append([vol_name, "x1" , vol_name, "v"])
                op_info[vol_name] = {}
                continue

            # 2) Capacitor
            i_match = i_regex.match(line)
            if i_match:
                i_index_str = i_match.group(1)
                current_name = f"I{i_index_str}"
                ckt_hierarchy.append([current_name, "x1", current_name, "i"])
                op_info[current_name] = {}
                continue

    if ib_entry_needed:
        ckt_hierarchy.append(["Ib", "", "Ib", "i"])
        op_info["Ib"] = {}

    return ckt_hierarchy, op_info

###############################################################################
# 4) Sort ckt_hierarchy: M0 < M1 < ... < C0 < C1 < ...
###############################################################################
def sort_hierarchy(ckt_hierarchy):
    def key_func(item):
        name = item[0]  # e.g. "M0", "C0", "Ib"
        if name.startswith("M"):
            return (0, int(name[1:]))
        elif name.startswith("C"):
            return (1, int(name[1:]))
        elif name.startswith("V"):
            return (2, int(name[1:]))
        elif name.startswith("I"):
            return (3, int(name[1:]))
        else:
            # Others (like "Ib")
            return (4, 999999)

    ckt_hierarchy.sort(key=key_func)

###############################################################################
# 5) Reorder op_info in same order
###############################################################################
def reorder_op_info(ckt_hierarchy, op_info):
    # Make an OrderedDict in the same order as ckt_hierarchy
    ordered = OrderedDict()
    for item in ckt_hierarchy:
        dev_name = item[0]
        if dev_name in op_info:
            ordered[dev_name] = op_info[dev_name]
    return ordered

###############################################################################
# 6) Convert OrderedDict -> normal dict (preserves insertion order in Python 3.7+)
###############################################################################
def to_regular_dict(ordered):
    # In Python 3.7+, dict keeps insertion order
    # So this conversion preserves the sequence of keys
    return dict(ordered)

###############################################################################
# 7) New function to parse the .PARAM section and generate parameter bounds
###############################################################################
def parse_parameter_bounds(netlist_file):
    """
    Reads the netlist file and parses the .PARAM section.
    Based on the parameter names found, returns a dictionary of parameter bounds.
    """
    p_indices = set()
    n_indices = set()
    capacitor_indices = set()
    resistor_indices = set()
    vb_indices = set()
    ib_indices = set()

    # Read the netlist file lines
    with open(netlist_file, 'r') as f:
        lines = f.readlines()

    # Identify the .PARAM section (and continuation lines starting with "+")
    param_lines = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith('.PARAM'):
            # Start of .PARAM section; remove the ".PARAM" keyword
            param_lines.append(stripped[len('.PARAM'):].strip())
            collecting = True
        elif collecting and stripped.startswith('+'):
            # Continuation of the .PARAM section
            param_lines.append(stripped[1:].strip())
        elif collecting and not stripped.startswith('+'):
            # End of .PARAM section once a non-continuation line is reached
            break

    # Join all parameter tokens and split by whitespace.
    param_content = " ".join(param_lines)
    tokens = param_content.split()

    # Process each token like "P1_L=1" or "CAPACITOR_0=1p"
    for token in tokens:
        if '=' not in token:
            continue
        key, _ = token.split('=', 1)
        key = key.strip()

        # Check for P parameters: e.g., P1_L, P2_W, P3_M, etc.
        m = re.match(r'^P(\d+)_(L|W|M)$', key, re.IGNORECASE)
        if m:
            p_indices.add(m.group(1))
            continue

        # Check for N parameters: e.g., N1_L, N2_W, etc.
        m = re.match(r'^N(\d+)_(L|W|M)$', key, re.IGNORECASE)
        if m:
            n_indices.add(m.group(1))
            continue

        # Check for CAPACITOR parameters: e.g., CAPACITOR_0
        m = re.match(r'^CAPACITOR_(\d+)$', key, re.IGNORECASE)
        if m:
            capacitor_indices.add(m.group(1))
            continue

        # Check for RESISTOR parameters: e.g., RESISTOR_3
        m = re.match(r'^RESISTOR_(\d+)$', key, re.IGNORECASE)
        if m:
            resistor_indices.add(m.group(1))
            continue

        # Check for VB parameters: e.g., VB1, VB2, etc.
        m = re.match(r'^VB(\d+)$', key, re.IGNORECASE)
        if m:
            vb_indices.add(m.group(1))
            continue

        # Check for IB parameters: e.g., IB1, IB2, etc.
        m = re.match(r'^IB(\d+)$', key, re.IGNORECASE)
        if m:
            ib_indices.add(m.group(1))
            continue

    # Build the parameter_bounds dictionary based on the found parameters
    parameter_bounds = {}

    # For P parameters: add three entries per index.
    for idx in sorted(p_indices, key=lambda x: int(x)):
        parameter_bounds[f"L_P{idx}"] = {"min": 0.13, "max": 1.0, "type": "real"}
        parameter_bounds[f"W_P{idx}"] = {"min": 0.2, "max": 10.0, "type": "real"}
        parameter_bounds[f"M_P{idx}"] = {"min": 1, "max": 100, "type": "int"}

    # For N parameters.
    for idx in sorted(n_indices, key=lambda x: int(x)):
        parameter_bounds[f"L_N{idx}"] = {"min": 0.13, "max": 1.0, "type": "real"}
        parameter_bounds[f"W_N{idx}"] = {"min": 0.2, "max": 10.0, "type": "real"}
        parameter_bounds[f"M_N{idx}"] = {"min": 1, "max": 100, "type": "int"}

    # For CAPACITOR parameters.
    for idx in sorted(capacitor_indices, key=lambda x: int(x)):
        parameter_bounds[f"C{idx}"] = {"min": 1.0, "max": 100.0, "type": "real"}

    # For RESISTOR parameters.
    for idx in sorted(resistor_indices, key=lambda x: int(x)):
        parameter_bounds[f"R{idx}"] = {"min": 0.1, "max": 1000.0, "type": "real"}

    # For VB and IB parameters: use the index in the key.
    for idx in sorted(vb_indices, key=lambda x: int(x)):
        parameter_bounds[f"V{idx}"] = {"min": 0.0, "max": 1.8, "type": "real"}

    for idx in sorted(ib_indices, key=lambda x: int(x)):
        parameter_bounds[f"IB{idx}"] = {"min": 1e-6, "max": 4e-5, "type": "real"}

    return parameter_bounds

###############################################################################
# 8) create_yaml
###############################################################################
def create_yaml(ckt_hierarchy, op_info, parameter_bounds, output_path):
    """
    Produces a YAML file with block-style top-level but inline sub-lists in ckt_hierarchy.
    """
    # Mark each sublist to be rendered in flow style
    ckt_hierarchy_in_flow = [FlowItem(item) for item in ckt_hierarchy]

    # Convert op_info to normal dict
    # so we don't get "!!python/object/apply:collections.OrderedDict"
    op_info_regular = to_regular_dict(op_info)

    data = {
        "parameter_bounds": parameter_bounds,
        "ckt_hierarchy": ckt_hierarchy_in_flow,
        "op": op_info_regular
    }

    with open(output_path, 'w') as f:
        yaml.dump(
            data,
            f,
            sort_keys=False,          # do NOT reorder dict keys alphabetically
            default_flow_style=False, # block style for top-level
            width=120
        )

###############################################################################
# 9) Main
###############################################################################
def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_spice.py <input_spice_file> <input_param_file> <output_yaml_file>")
        sys.exit(1)

    netlist_file = sys.argv[1]
    param_file  = sys.argv[2]
    output_yaml  = sys.argv[3]

    # First, parse the parameter bounds from the .PARAM section.
    parameter_bounds = parse_parameter_bounds(param_file)

    # Parse netlist
    ckt_hierarchy, op_info = parse_spice_netlist(netlist_file, ib_entry_needed=False)

    # Sort
    sort_hierarchy(ckt_hierarchy)

    # Reorder op in same order
    op_info_ordered = reorder_op_info(ckt_hierarchy, op_info)

    # Dump
    create_yaml(ckt_hierarchy, op_info_ordered, parameter_bounds, output_yaml)

    print(f"YAML file written to {output_yaml}")

###############################################################################
# 10) If __name__ == "__main__"
###############################################################################
if __name__ == "__main__":
    main()
