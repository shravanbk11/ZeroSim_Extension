import os
import argparse
import yaml
import json
import numpy as np

from geneticalgorithm import geneticalgorithm as ga
 
from AMP import AMPEnv
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="GA for circuit simulation")
    # parser.add_argument('--config_file', type=str, default='AMP_NMCF.yaml',
    #                     help="path to the corresponding .yaml file")
    # parser.add_argument('--simulation_path', type=str, default='simulations',
    #                     help="path to the simulation folder")
    # parser.add_argument('--log_file', type=str, default='ga_simulation.log',
    #                     help="path to the log file")
    parser.add_argument('--output_dir', type=str, default='/volume/xyang/dataset/Opamp_dataset',
                         help="Directory to save the results.")
    parser.add_argument('--name', type=str, required=True,
                        help="Name of the simulation")
    parser.add_argument('--n_iter', type=int, default=5000)
    parser.add_argument('--time_out', type=int, default=600)

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
    log_file = os.path.join('logs/AMP_new', args.name+'_ga_simulation.log')
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    simulation_path = os.path.join('AMP_new', args.name)
    if args.name[-6:-1] == '_copy':
        config_file = os.path.join(simulation_path, args.name[:-6]+'.yaml')
    elif '_copy' in args.name:
        config_file = os.path.join(simulation_path, args.name[:-5]+'.yaml')
    else:
        config_file = os.path.join(simulation_path, args.name+'.yaml')

    # Load parameter bounds from YAML
    varbound, param_names, variable_type_mixed = load_parameter_bounds(config_file)

    output_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(output_dir, exist_ok=True)
    env = AMPEnv(output_dir, simulation_path, config_file, args.name)

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
            
            return -reward  # Assuming small is better
        
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
        'max_iteration_without_improv': None,  # Termination condition
        'stop_when_reached': 1e-10
    }

    # Initialize the genetic algorithm
    try:
        model = ga(
            function=target_function,
            dimension=len(varbound),
            variable_type_mixed=variable_type_mixed,
            variable_boundaries=varbound,
            algorithm_parameters=algorithm_param,
            function_timeout=args.time_out
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
        with open(os.path.join(simulation_path, "best_solution.json"), "w") as f:
            json.dump(solution.tolist(), f, indent=4)
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

