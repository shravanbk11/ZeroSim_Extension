# -*- coding: utf-8 -*-
import os
import argparse
import yaml
import json
import time
import random
import logging
import numpy as np
 
#from AMP import AMPEnv
from AMP_all_sim import AMPEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Random sampling for circuit simulation")
    parser.add_argument('--output_dir', type=str, default='/volume/xyang/dataset/Opamp_dataset',
                        help="Directory to save the results.")
    parser.add_argument('--name', type=str, required=True,
                        help="Name of the simulation")
    parser.add_argument('--n_samples', type=int, default=60000,
                        help="Number of random samples to evaluate.")
    parser.add_argument('--time_budget_sec', type=int, default=0,
                        help="Optional wall-clock time budget (seconds). 0 = no limit.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument('--stop_when_reached', type=float, default=None,
                        help="Optional early-stop when reward >= this value.")
    parser.add_argument('--log_every', type=int, default=50,
                        help="Log progress every N evaluations.")
    parser.add_argument('--time_out', type=int, default=600,
                        help="(kept for interface compatibility; per-eval timeout if used in AMPEnv).")
    return parser.parse_args()


def load_parameter_bounds(config_file):
    """
    Loads parameter bounds from a YAML configuration file.
    Returns:
        varbound: np.ndarray of shape (D, 2) with [min, max] for each variable
        param_names: list[str] of parameter names
        variable_type_mixed: np.ndarray of 'real' or 'int'
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    param_bounds = config.get('parameter_bounds', {})

    varbound = []
    param_names = []
    variable_type_mixed = []
    for param, bounds in param_bounds.items():
        ptype = bounds.get('type', 'real').lower()
        pmin, pmax = bounds['min'], bounds['max']
        # ensure numeric
        if ptype == 'real':
            varbound.append([float(pmin), float(pmax)])
            variable_type_mixed.append('real')
        else:
            # inclusive integer bounds
            varbound.append([int(pmin), int(pmax)])
            variable_type_mixed.append('int')
        param_names.append(param)

    return np.array(varbound, dtype=float), param_names, np.array(variable_type_mixed, dtype=object)


def build_config_path(simulation_path, name):
    if name[-6:-1] == '_copy':
        return os.path.join(simulation_path, name[:-6] + '.yaml')
    elif '_copy' in name:
        return os.path.join(simulation_path, name[:-5] + '.yaml')
    else:
        return os.path.join(simulation_path, name + '.yaml')


def sample_one(varbound, variable_type_mixed, param_names):
    """
    Draw one random sample within bounds.
    Returns a list aligned with param_names.
    """
    sample = []
    for (low, high), ptype, pname in zip(varbound, variable_type_mixed, param_names):
        if ptype == 'real':
            val = random.random() * (high - low) + low
        else:
            # integer inclusive range
            val = random.randint(int(np.ceil(low)), int(np.floor(high)))
        sample.append(val)
    # enforce integer constraints for device counts if needed (matches your GA code logic)
    for idx, key in enumerate(param_names):
        if 'M_' in key or 'M_C' in key:
            sample[idx] = int(round(sample[idx]))
    return sample


def main(args):
    # --- Logging setup ---
    os.makedirs('logs/AMP_new', exist_ok=True)
    log_file = os.path.join('logs/AMP_new', args.name + '_random_simulation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info("Starting random sampling pipeline.")

    # --- Paths & config ---
    simulation_path = os.path.join('AMP_new', args.name)
    config_file = build_config_path(simulation_path, args.name)

    # --- Bounds ---
    varbound, param_names, variable_type_mixed = load_parameter_bounds(config_file)
    dim = len(param_names)
    logging.info(f"Loaded bounds for {dim} parameters from: {config_file}")

    # --- Output & environment ---
    output_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(output_dir, exist_ok=True)

    env = AMPEnv(output_dir, simulation_path, config_file, args.name)

    # --- Random seeds ---
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- CSV for all trials ---
    trials_csv = os.path.join(output_dir, "random_trials.csv")
    if not os.path.exists(trials_csv):
        with open(trials_csv, "w") as f:
            header = ",".join(param_names + ["reward"])
            f.write(header + "\n")

    # --- Evaluation loop ---
    best_reward = -float('inf')
    best_params = None
    start = time.time()

    def write_trial(params_list, reward_value):
        with open(trials_csv, "a") as f:
            row = ",".join(str(x) for x in params_list + [reward_value])
            f.write(row + "\n")

    num_evals = 0
    try:
        for i in range(1, args.n_samples + 1):
            # time budget check
            if args.time_budget_sec and (time.time() - start) >= args.time_budget_sec:
                logging.info(f"Time budget ({args.time_budget_sec}s) reached at {i-1} evaluations.")
                break

            params = sample_one(varbound, variable_type_mixed, param_names)

            try:
                reward = env.step(params)  # Higher reward is better
            except Exception as e:
                logging.error(f"Error in env.step with parameters {params}: {e}")
                continue

            write_trial(params, reward)
            num_evals += 1

            # track best
            if reward > best_reward:
                best_reward = reward
                best_params = list(params)

            # early stop
            if args.stop_when_reached is not None and reward >= args.stop_when_reached:
                logging.info(f"Early stop: reward {reward} >= {args.stop_when_reached} at eval {i}.")
                break

            if i % args.log_every == 0:
                logging.info(f"Evaluated {i} / {args.n_samples} samples. "
                             f"Best so far: {best_reward:.6g}")

        # Save best solution
        best_path = os.path.join(simulation_path, "best_solution.json")
        best_payload = {
            "best_reward": best_reward if best_params is not None else None,
            "best_parameters": dict(zip(param_names, best_params)) if best_params is not None else None,
            "num_evaluations": num_evals,
            "stopped_early": (args.stop_when_reached is not None and
                              best_reward is not None and best_reward >= args.stop_when_reached),
            "time_elapsed_sec": round(time.time() - start, 3),
        }
        with open(best_path, "w") as f:
            json.dump(best_payload, f, indent=2)
        logging.info(f"Saved best solution to '{best_path}'.")
    except Exception as e:
        logging.error(f"Unexpected error during random sampling: {e}")
    finally:
        try:
            env.close()
            logging.info("Closed simulation environment.")
        except Exception as e:
            logging.error(f"Error closing simulation environment: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
