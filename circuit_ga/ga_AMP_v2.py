import os
import argparse
import yaml
import json
import numpy as np
import pygad

from AMP import AMPEnv
import logging
 
def parse_args():
    parser = argparse.ArgumentParser(description="GA for circuit simulation")
    parser.add_argument('--output_dir', type=str, default='/volume/xyang/dataset/AMP',
                         help="Directory to save the results.")
    parser.add_argument('--name', type=str, required=True,
                        help="Name of the simulation")
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
    
    varbound = []
    param_names = []
    variable_type = []
    for param, bounds in param_bounds.items():
        if bounds['type'] == 'real':
            varbound.append([float(bounds['min']), float(bounds['max'])])
        else:
            varbound.append([int(bounds['min']), int(bounds['max'])])

        variable_type.append(bounds['type'])
        param_names.append(param)
    
    return np.array(varbound), param_names, variable_type

def main(args):
    log_file = os.path.join('logs', args.name+'_ga_simulation_new.log')
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    simulation_path = os.path.join('AMP', args.name)
    config_file = os.path.join(simulation_path, args.name+'.yaml')

    # Load parameter bounds from YAML
    varbound, param_names, variable_type = load_parameter_bounds(config_file)

    output_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(output_dir, exist_ok=True)
    env = AMPEnv(output_dir, simulation_path, config_file, args.name)

    def fitness_function(solution, solution_idx):
        try:
            # Enforce integer constraints for specific parameters
            for idx, key in enumerate(param_names):
                if 'M_' in key or 'M_C' in key:
                    solution[idx] = int(round(solution[idx]))
                else:
                    solution[idx] = float(solution[idx])
            
            # Run simulation and get fitness score
            reward = env.step(solution)
            return reward  # Assuming higher is better
        
        except Exception as e:
            logging.error(f"Error in fitness_function with parameters {solution}: {e}")
            return -np.inf  # Penalize invalid solutions

    num_generations = args.n_iter
    num_parents_mating = 20
    sol_per_pop = 50
    num_genes = len(varbound)

    initial_population = np.random.uniform(
        low=varbound[:, 0], high=varbound[:, 1], size=(sol_per_pop, num_genes)
    )

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=varbound[:, 0],
        init_range_high=varbound[:, 1],
        mutation_percent_genes=10,
        parent_selection_type="sss",
        crossover_type="uniform",
        mutation_type="random",
        initial_population=initial_population
    )

    # Run the genetic algorithm
    try:
        logging.info("Starting Genetic Algorithm optimization.")
        ga_instance.run()
        logging.info("Genetic Algorithm optimization completed.")
    except Exception as e:
        logging.error(f"Error during Genetic Algorithm execution: {e}")
        env.close()
        return

    # Extract the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Save the best solution to a JSON file
    try:
        best_solution = {
            "solution": solution.tolist(),
            "fitness": solution_fitness
        }
        with open("best_solution.json", "w") as f:
            json.dump(best_solution, f, indent=4)
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
