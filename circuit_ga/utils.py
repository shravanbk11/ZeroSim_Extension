import os
import yaml
import numpy as np
 

def get_dc_score(cur, target):
    score = np.min([(target - cur) / (target + cur), 0])
    return score


def get_ac_score(cur, target):
    if cur > 0:
        score = -1
    else:
        score = np.min([(cur - target) / (cur + target), 0])
        if cur < target:
            score = 0
    return score


def find_op_files(directory="."):
    """
    Returns a list of full file paths for files containing '_op' 
    within the specified directory.
    """
    op_files = []
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        # Check if '_op' is part of the filename and that it's a file (not a directory)
        if "_op" in filename and os.path.isfile(full_path):
            op_files.append(full_path)
    return op_files


class ParseParams:
    def __init__(self, config_file):
        self.params_mos = (
            'gmbs',
            'gm',
            'gds',
            'vdsat',
            'vth',
            'id',
            'ibd',
            'ibs',
            'gbd',
            'gbs',
            'isub',
            'igidl',
            'igisl',
            'igs',
            'igd',
            'igb',
            'igcs',
            'vbs',
            'vgs',
            'vds',
            'cgg',
            'cgs',
            'cgd',
            'cbg',
            'cbd',
            'cbs',
            'cdg',
            'cdd',
            'cds',
            'csg',
            'csd',
            'css',
            'cgb',
            'cdb',
            'csb',
            'cbb',
            'capbd',
            'capbs',
            #'qg',
            #'qb',
            #'qs',
            #'qinv',
            #'qdef',
            #'gcrg',
            #gtau'
        )
        
        self.params_r = (
            'r',
            'ac',
            'temp',
            'dtemp',
            'l',
            'w',
            'm',
            'tc',
            'tc1',
            'tc2',
            'scale',
            'noise',
            'i',
            'p',
            'sens_dc',
            'sens_real',
            'sens_imag',
            'sens_mag',
            'sens_ph',
            'sens_cplx'
        )
        
        self.params_c = (
            'capacitance',
            'cap',
            'c',
            'ic',
            'temp',
            'dtemp',
            'w',
            'l',
            'm',
            'scale',
            'i',
            'p',
            'sens_dc',
            'sens_real',
            'sens_imag',
            'sens_mag',
            'sens_ph',
            'sens_cplx'
        )

        self.params_i = (
            'dc',
            'acmag',
            'acphase',
            'acreal',
            'acimag',
            'v',
            'p',
            'current'
        )

        self.params_v = (
            'dc',
            'acmag',
            'acphase',
            'acreal',
            'acimag',
            'i',
            'p',
        )

        self.ckt_hierarchy = []
        self.op = {}

        # Read YAML data
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)  # safe_load is recommended over load

        # Assign the data to your class attributes
        self.ckt_hierarchy = data.get('ckt_hierarchy', [])
        self.op = data.get('op', {})

    def parse_dcop(self, file_names):
        """
        Parse one or more _op files and store the combined results in self.op.
        file_names can be a single string or a list of strings.
        """
        if isinstance(file_names, str):
            file_names = [file_names]

        # Will hold all parsed float values from all _op files
        all_values = []
        for f_name in file_names:
            with open(f_name, 'r') as f:
                lines_op = f.readlines()
            
            # Find the line "Values:\n"
            start_idx = None
            for index, line in enumerate(lines_op):
                if line == "Values:\n":
                    start_idx = index
                    break

            # If the file is formatted as expected, parse the relevant lines
            if start_idx is not None:
                # Usually we skip the next line or two after "Values:\n" 
                # (as in your example, start_idx + 2)
                data_lines = lines_op[start_idx+2:-1]
                for line in data_lines:
                    all_values.append(float(line.split('\n')[0].split('\t')[1]))

        # Now we have all the values, we can assign them to the correct parameters
        num_dev = len(self.ckt_hierarchy)
        num_dev_params_mos = len(self.params_mos)
        num_dev_params_r = len(self.params_r)
        num_dev_params_c = len(self.params_c)
        num_dev_params_i = len(self.params_i)
        num_dev_params_v = len(self.params_v)
        
        idx = 0       
        for i in range(num_dev):        
            dev_type = self.ckt_hierarchy[i][3]
            if dev_type == 'm' or dev_type == 'M':   
                for j in range(num_dev_params_mos):
                    param = self.params_mos[j]
                    self.op[list(self.op)[i]][param] = all_values[idx+j]
                idx = idx + num_dev_params_mos
            elif dev_type == 'r' or dev_type == 'R':  
                for j in range(num_dev_params_r):
                    param = self.params_r[j]
                    self.op[list(self.op)[i]][param] = all_values[idx+j]
                idx = idx + num_dev_params_r
            elif dev_type == 'c' or dev_type == 'C':
                for j in range(num_dev_params_c):
                    param = self.params_c[j]
                    self.op[list(self.op)[i]][param] = all_values[idx+j]
                idx = idx + num_dev_params_c
            elif dev_type == 'i' or dev_type == 'I':
                for j in range(num_dev_params_i):
                    param = self.params_i[j]
                    self.op[list(self.op)[i]][param] = all_values[idx+j]
                idx = idx + num_dev_params_i
            elif dev_type == 'v' or dev_type == 'V':
                for j in range(num_dev_params_v):
                    param = self.params_v[j]
                    self.op[list(self.op)[i]][param] = all_values[idx+j]
                idx = idx + num_dev_params_v
            else:
                None
        
        return self.op

