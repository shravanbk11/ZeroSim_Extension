import os
import json
import glob
import random
import argparse


def load_samples(file_path):
    """Load samples from a JSON file.
    
    The file may contain a list of samples directly or a dict with a "samples" key.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "samples" in data:
        return data["samples"]
    else:
        raise ValueError(f"Unknown data format in file {file_path}")


def split_samples(samples, train_ratio, val_ratio, test_ratio):
    """Shuffle and split the sample list into train, validation, and test subsets."""
    random.shuffle(samples)
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    return train_samples, val_samples, test_samples


def main():
    parser = argparse.ArgumentParser(
        description="Read JSON files from a dir, split each file's samples into train/val/test subsets, and combine them."
    )
    parser.add_argument("--file_dir", type=str, required=True,
                        help="Path to the directory containing JSON files.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Proportion of samples for training (default: 0.8).")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Proportion of samples for validation (default: 0.1).")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Proportion of samples for testing (default: 0.1).")
    parser.add_argument("--train_output", type=str, default="train_samples.json",
                        help="Output file for training samples (default: train_samples.json).")
    parser.add_argument("--val_output", type=str, default="val_samples.json",
                        help="Output file for validation samples (default: val_samples.json).")
    parser.add_argument("--test_output", type=str, default="test_samples.json",
                        help="Output file for test samples (default: test_samples.json).")
    args = parser.parse_args()

    # Validate that the ratios add up to 1.0.
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must equal 1.0")

    # Read file paths from the input text file.
    # with open(args.file_list, "r") as f:
    #     input_files = [line.strip() for line in f if line.strip()]

    # Replace 'path/to/dir' with your target directory.
    input_files = glob.glob(os.path.join(args.file_dir + "*.json"))
    
    all_train_samples = []
    all_val_samples = []
    all_test_samples = []

    for file_path in input_files:
        samples = load_samples(file_path)
        train_samples, val_samples, test_samples = split_samples(
            samples, args.train_ratio, args.val_ratio, args.test_ratio
        )
        all_train_samples.extend(train_samples)
        all_val_samples.extend(val_samples)
        all_test_samples.extend(test_samples)
        print(f"File {file_path}: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples.")

    # Save the combined splits to separate JSON files.
    with open(args.train_output, "w") as f:
        json.dump(all_train_samples, f, indent=4)
    print(f"Saved {len(all_train_samples)} train samples to {args.train_output}")

    with open(args.val_output, "w") as f:
        json.dump(all_val_samples, f, indent=4)
    print(f"Saved {len(all_val_samples)} validation samples to {args.val_output}")

    with open(args.test_output, "w") as f:
        json.dump(all_test_samples, f, indent=4)
    print(f"Saved {len(all_test_samples)} test samples to {args.test_output}")


if __name__ == "__main__":
    main()
