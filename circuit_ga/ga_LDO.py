import os
import argparse
import yaml
import json
import numpy as np
 
from geneticalgorithm import geneticalgorithm as ga

from LDO_TB import LDOtestbenchEnv
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="GA for circuit simulation")
    parser.add_argument('--config_file', type=str, default='AMP_NMCF.yaml',
                        help="path to the corresponding .yaml file")
    parser.add_argument('--simulation_path', type=str, default='simulations',
                        help="path to the simulation folder")
    parser.add_argument('--log_file', type=str,default='ga_simulation.log',
                        help="path to the log file")
    parser.add_argument('--output_dir', type=str, required=True,
                         help="Directory to save the results.")
    parser.add_argument('--n_iter', type=int, default=10000)

    args = parser.parse_args()
    return args


def load_parameter_bounds(config_file):
    """
    Loads parameter bounds from a YAML configuration file.
    Returns: dict: Dictionary containing parameter bounds.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract the parameter bounds
    param_bounds = config.get('parameter_bounds', {})
    
    # Convert to a list of tuples for ga
    varbound = []
    param_names = []
    variable_type_mixed = []
    for param, bounds in param_bounds.items():
        if bounds['type'] == 'real':
            varbound.append([float(bounds['min']), float(bounds['max'])])
        else:
            varbound.append([int(bounds['min']), int(bounds['max'])])

        variable_type_mixed.append(bounds['type'])
        param_names.append(param)
    
    return np.array(varbound), param_names, np.array(variable_type_mixed)


def main(args):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )

    # Load parameter bounds from YAML
    varbound, param_names, variable_type_mixed = load_parameter_bounds(args.config_file)
    os.makedirs(args.output_dir, exist_ok=True)
    M_CL_low, L_CL, W_CL = 1, 30, 30
    CL_low = M_CL_low * (L_CL * W_CL * 2e-15 + (L_CL + W_CL) * 0.38e-15)
    env = LDOtestbenchEnv(args.output_dir, args.simulation_path, CL_low, args.config_file)

    def target_function(parameters):
        try:
            # Convert parameters to list and enforce integer constraints
            parameters = list(parameters)
            for idx, key in enumerate(param_names):
                if 'M_' in key or 'M_C' in key:
                    parameters[idx] = int(round(parameters[idx]))
                else:
                    parameters[idx] = float(parameters[idx])
            
            # Run simulation and get fitness score
            reward = env.step(parameters)
            
            return -reward  # Assuming smaller is better
        
        except Exception as e:
            logging.error(f"Error in fitness_function with parameters {parameters}: {e}")

    # Define GA parameters
    algorithm_param = {
        'max_num_iteration': args.n_iter,     # Maximum number of generations
        'population_size': 50,                # Population size
        'mutation_probability': 0.1,          # Mutation probability
        'elit_ratio': 0.01,                   # Elitism ratio
        'crossover_probability': 0.5,         # Crossover probability
        'parents_portion': 0.3,               # Parents portion
        'crossover_type': 'uniform',           # Crossover type
        'max_iteration_without_improv': None  # Termination condition
    }

    # Initialize the genetic algorithm
    try:
        model = ga(
            function=target_function,
            dimension=len(varbound),
            variable_type_mixed=variable_type_mixed,
            variable_boundaries=varbound,
            algorithm_parameters=algorithm_param
        )
        logging.info("Initialized Genetic Algorithm.")
    except Exception as e:
        logging.error(f"Failed to initialize Genetic Algorithm: {e}")
        env.close()
        return
    
    # Run the genetic algorithm
    try:
        logging.info("Starting Genetic Algorithm optimization.")
        model.run()
        logging.info("Genetic Algorithm optimization completed.")
    except Exception as e:
        logging.error(f"Error during Genetic Algorithm execution: {e}")
        env.close()
        return
    
    # Extract the best solution
    solution = model.output_dict
    
    # Save the best solution to a JSON file
    try:
        with open(os.path.join(args.output_dir, "best_solution.json"), "w") as f:
            json.dump(solution, f, indent=4)
        logging.info("Saved best solution to 'best_solution.json'.")
    except Exception as e:
        logging.error(f"Failed to save best solution: {e}")
    
    # Close the environment after GA completes
    try:
        env.close()
        logging.info("Closed simulation environment.")
    except Exception as e:
        logging.error(f"Error closing simulation environment: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

