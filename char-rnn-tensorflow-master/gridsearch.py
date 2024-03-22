import itertools
import subprocess
import os

# Define the hyperparameters and their respective sets of values to test
hyperparameters = {
    'rnn_size': [128, 256, 512],
    'num_layers': [1, 2, 3],
    'seq_length': [50, 100, 200]
}

# Generate all combinations of hyperparameters
combinations = list(itertools.product(*hyperparameters.values()))

# Base directory for saving the experiments
base_save_dir = "experiments"

if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)

# Loop over each combination and run the experiment
for combination in combinations:
    # Create a directory name based on the hyperparameter combination
    dir_name_components = []
    for key, value in zip(hyperparameters.keys(), combination):
        dir_name_components.append("{}{}".format(key, value))
    dir_name = "_".join(dir_name_components)
    save_dir = os.path.join(base_save_dir, dir_name)

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create the command to run `train.py` with the current set of hyperparameters
    args = ["python", "train.py", "--num_epochs", "10", "--save_dir", save_dir, "--logname", dir_name, "--model", "gru"]
    for key, value in zip(hyperparameters.keys(), combination):
        args.append("--{}".format(key))
        args.append(str(value))
    
    print("Running experiment with parameters: {}".format(dir_name))
    # Call the training script with the current set of hyperparameters
    subprocess.run(args)
    
    # Note: Consider adding exception handling and logging for robustness

print("Grid search is complete!")
