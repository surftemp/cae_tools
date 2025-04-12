import optuna
import subprocess
import json
import re

def objective(trial):
    # Define the hyperparameter search space
    params = {
        'nr_epochs': trial.suggest_int('nr_epochs', 3000, 9000, step=1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'lr_step_size': trial.suggest_int('lr_step_size', 1000, 1000),
        'lr_gamma': trial.suggest_uniform('lr_gamma', 1.0, 1.0),
        'lambda_mse': trial.suggest_uniform('lambda_mse', 1, 1),
        'lambda_ssim': trial.suggest_uniform('lambda_ssim', 1, 1),
        'lambda_pearson': trial.suggest_loguniform('lambda_pearson', 1e-5, 1e-1),
        'lambda_l1': trial.suggest_uniform('lambda_l1', 1, 1),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
        'dropout_rate': trial.suggest_uniform('dropout_rate', 0.0, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'fc_size': trial.suggest_int('fc_size', 64, 8192, step=64),
        'latent_size': trial.suggest_int('latent_size', 64, 2048, step=64),
    }
 
    max_latent_size = params['fc_size']  # `latent_size` should not exceed `fc_size`
    params['latent_size'] = trial.suggest_int('latent_size', 64, max_latent_size, step=64)

    print(f"Starting trial {trial.number} with params: {params}")

    # Define additional static parameters
    input_variables = ["land_cover", "albedo_monthly_climatology_means", "elevation", "era5_skt", "sin_doy", "cos_doy", "slope_magnitude", "slope_direction", "era5_u10", "era5_v10", "era5_ssrd", "urban_area", "suburban_area"]
    output_variable = "ST_slices"
    train_folder = "/lustre_scratch/shaerdan/data_folder/train_v3/train/processed_train/"
    test_folder = "/lustre_scratch/shaerdan/data_folder/test_v3/test/processed_test/"
    layer_definitions_path = "spec_august_ver.json"
    database_path = "database_v23_wo_ssrd.db"
    method = "unet"
    scheduler_type = "None"
    
    # Generate randomized training and testing paths
    train_paths = subprocess.check_output(f"ls {train_folder}*.nc | shuf -n 100", shell=True).decode().split()
    test_paths = subprocess.check_output(f"ls {test_folder}*.nc | shuf -n 100", shell=True).decode().split()

    # Randomize the output folder for each trial
    output_folder = f"/lustre_scratch/shaerdan/models/model_{trial.number}"

    # Build the command to execute train_cae.py
    cmd = [
        "python", "/home/jovyan/softwares/cae_tools/src/cae_tools/cli/train_cae.py",
        "--train-inputs", *train_paths,
        "--test-inputs", *test_paths,
        "--model-folder", output_folder,
        "--input-variables", *input_variables,
        "--output-variable", output_variable,
        "--nr-epochs", str(params['nr_epochs']),
        "--learning-rate", str(params['learning_rate']),
        "--lr-step-size", str(params['lr_step_size']),
        "--lr-gamma", str(params['lr_gamma']),
        "--lambda-mse", str(params['lambda_mse']),
        "--lambda-ssim", str(params['lambda_ssim']),
        "--lambda-pearson", str(params['lambda_pearson']),
        "--lambda-l1", str(params['lambda_l1']),
        "--weight-decay", str(params['weight_decay']),
        "--dropout-rate", str(params['dropout_rate']),
        "--batch-size", str(params['batch_size']),
        "--fc-size", str(params['fc_size']),
        "--latent-size", str(params['latent_size']),
        "--method", method,
        "--scheduler-type", scheduler_type,
        "--layer-definitions-path", layer_definitions_path,
        "--database-path", database_path
    ]

    try:
        # Run the training process and capture the output
        print(f"Running command for trial {trial.number}: {' '.join(cmd)}")
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()

        # Display command output for monitoring
        print(f"Command output for trial {trial.number}:\n{output}")

        # Use a regular expression to find all occurrences of "test_mse: <value>" and take the last one
        test_mse_values = re.findall(r"test_mse:\s([0-9]*\.[0-9]+)", output)
        if test_mse_values:
            final_test_mse = float(test_mse_values[-1])  # Get the last test_mse value
            print(f"Final test_mse for trial {trial.number}: {final_test_mse}")
        else:
            # If no test_mse was found, set a high value to indicate a failed trial
            print(f"No test_mse found for trial {trial.number}. Setting a high test_mse.")
            final_test_mse = float('inf')

    except subprocess.CalledProcessError as e:
        print(f"Error in trial {trial.number}: {e.output.decode()}")
        raise optuna.TrialPruned()

    # Log trial results after each trial
    trial_log = {
        "trial_number": trial.number,
        "params": params,
        "test_mse": final_test_mse
    }
    with open("trial_logs.json", "a") as f:
        f.write(json.dumps(trial_log) + "\n")

    return final_test_mse  # Minimize test_mse

# Set up the study with persistent SQLite storage
study = optuna.create_study(
    study_name="unet_hyperparameter_optimization",
    storage="sqlite:///optuna_study.db",  # Saves the study in a local SQLite file
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler()
)

# Enqueue a specific starting trial
starting_params = {
    'nr_epochs': 3000,
    'learning_rate': 1e-3,
    'lr_step_size': 1000,
    'lr_gamma': 1.0,
    'lambda_mse': 1,
    'lambda_ssim': 1,
    'lambda_pearson': 0.0005,
    'lambda_l1': 1,
    'weight_decay': 1e-5,
    'dropout_rate': 0.1,
    'batch_size': 256,
    'fc_size': 3200,
    'latent_size': 800
}
study.enqueue_trial(starting_params)

# Optimize the study
study.optimize(objective, n_trials=50, show_progress_bar=True)  # Number of trials for optimization

# Print best hyperparameters and test loss
print("Best hyperparameters:", study.best_params)
print("Best test_mse:", study.best_value)

# Optionally save final results
with open("optuna_results.json", "w") as f:
    json.dump({"best_params": study.best_params, "best_loss": study.best_value}, f)
