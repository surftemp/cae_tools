import argparse
import json
import os

import xarray as xr
import numpy as np

from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.models.unet import UNET
from cae_tools.models.resunet import RESUNET
from cae_tools.models.resunet_gan import RESUNET_GAN
from cae_tools.models.var_ae_model import VAEUNET
from cae_tools.models.vae_train import VAE_Train
from cae_tools.models.unet_res_train import UNET_RES_Train
from cae_tools.models.srcnn_res_train import SRCNN_RES_Train
from cae_tools.models.linear_model import LinearModel
from cae_tools.models.model_sizer import ModelSpec

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-inputs", nargs="+", help="path(s) to netcdf4 file containing training data", required=True)
    parser.add_argument("--test-inputs", nargs="+", help="path(s) to netcdf4 file containing test data", required=True)
    parser.add_argument("--model-folder", help="folder to save the trained model to",required=True)
    parser.add_argument("--continue-training", action="store_true", help="continue training model")
    parser.add_argument("--input-variables", nargs="+", help="name of the input variable(s) in training/test data", required=True)
    parser.add_argument("--output-variable", help="name of the output variable in training/test data", required=True)
    parser.add_argument("--nr-epochs", type=int, help="number of training epochs", default=500)
    parser.add_argument("--latent-size", type=int, help="size of the latent space", default=4)
    parser.add_argument("--fc-size", type=int, help="size of the fully-connected layers", default=16)
    parser.add_argument("--batch-size", type=int, help="number of images to process in one batch", default=10)
    parser.add_argument("--learning-rate", type=float, help="the learning rate", default=0.001)
    parser.add_argument("--lr-step-size", type=int, help="the schedular steps for the learning rate", default=500)    
    parser.add_argument("--lr-gamma", type=float, help="decay factor of the scheduled learning rate", default=0.5)  
    parser.add_argument("--lambda-mse", type=float, help="controls the strength of the mse loss in vae", default=1)       
    parser.add_argument("--lambda-kl", type=float, help="controls the strength of the kl loss in vae", default=1)   
    parser.add_argument("--lambda-l1", type=float, help="controls the strength of l1 regularization", default=0.001)   
    parser.add_argument("--lambda-pearson", type=float, help="controls the strength of the pearson loss", default=1)
    parser.add_argument("--lambda-ssim", type=float, help="controls the strength of the ssim loss", default=1)    
    parser.add_argument("--lambda-additional", type=float, help="controls the strength of additional regularization", default=1) 
    parser.add_argument("--weight-decay", type=float, help="weight decay coefficient", default=1e-5)
    parser.add_argument("--dropout-rate", type=float, help="dropout rate", default=1e-1)
    parser.add_argument("--additional-loss", type=str, help="additional loss types ('contrastive', 'histogram', 'perceptual')", default=None)    
    parser.add_argument("--scheduler-type", type=str, help="scheduler type ('StepLR', 'ReduceLROnPlateau', 'ExponentialLR','CosineAnnealingLR')", default=None)        
    parser.add_argument("--method", choices=["conv", "unet","unet_res", "srcnn_res", "resunet_gan","var","vae", "linear"], default="var", help="methods")
    parser.add_argument("--layer-definitions-path", help="specify path of a JSON file with layer definitions", default=None)
    parser.add_argument("--stride", type=int, help="stride to use in convolutional layers", default=2)
    parser.add_argument("--kernel-size", type=int, help="kernel size to use in convolutional layers", default=3)
    parser.add_argument("--input-layer-count", type=int, help="number of input convolutional layers", default=None)
    parser.add_argument("--output-layer-count", type=int, help="number of output convolutional layers", default=None)
    parser.add_argument("--model-id", type=str, help="specify the model id when creating a model", default=None)
    parser.add_argument("--database-path", type=str, help="path to a database to store evaluation results",
                        default=None)
    parser.add_argument("--chunk-size", type=int, help="chunk size for xarray", default=1000)
    parser.add_argument("--include-coasts", help="folder to save the trained model to",default=False)
    

    args = parser.parse_args()

    train_ds = [xr.open_dataset(train_input) for train_input in args.train_inputs] #,chunks={'box': args.chunk_size}
    test_ds = [xr.open_dataset(test_input) for test_input in args.test_inputs] #,chunks={'box': args.chunk_size}
    case_dimension = train_ds[0][args.output_variable].dims[0]

    train_ds = train_ds[0] if len(train_ds) == 1 else xr.concat(train_ds, dim=case_dimension)
    test_ds = test_ds[0] if len(test_ds) == 1 else xr.concat(test_ds, dim=case_dimension)
    
#     # deal with coastal areas or not
#     if not args.include_coasts:
#         train_ds = train_ds.where(train_ds.edge == False, drop=True)
#         test_ds = test_ds.where(test_ds.edge == False, drop=True)
        


    print("Training cases: %d, Test cases: %d"%(train_ds[case_dimension].shape[0],test_ds[case_dimension].shape[0]))

    training_paths = ";".join(args.train_inputs)
    test_paths = ";".join(args.test_inputs)

    # for scaler inputs, broadcast them to have the same dimension as output variable:
    for var in args.input_variables:
        dims = train_ds[var].dims
        if dims == (case_dimension,):
            if 'y' in train_ds.dims and 'x' in train_ds.dims:
                y_dim, x_dim = train_ds.dims['y'], train_ds.dims['x']
            else:
                raise ValueError("'y' and 'x' dimensions not found in the dataset.")

            # Get the original values
            original_values = train_ds[var].values

            # Extend dimensions: create an array filled with the scalar value
            expanded_values = np.broadcast_to(original_values[:, np.newaxis, np.newaxis, np.newaxis], (original_values.shape[0], 1, y_dim, x_dim))
            expanded_var = xr.DataArray(expanded_values, coords={case_dimension: train_ds[case_dimension], 'channel': [0], 'y': np.arange(y_dim), 'x': np.arange(x_dim)}, dims=(case_dimension, 'channel', 'y', 'x'))

            train_ds[var] = expanded_var    
            
    for var in args.input_variables:
        dims = test_ds[var].dims
        if dims == (case_dimension,):
            if 'y' in test_ds.dims and 'x' in test_ds.dims:
                y_dim, x_dim = test_ds.dims['y'], test_ds.dims['x']
            else:
                raise ValueError("'y' and 'x' dimensions not found in the dataset.")

            # Get the original values
            original_values = test_ds[var].values

            # Extend dimensions: create an array filled with the scalar value
            expanded_values = np.broadcast_to(original_values[:, np.newaxis, np.newaxis, np.newaxis], (original_values.shape[0], 1, y_dim, x_dim))
            expanded_var = xr.DataArray(expanded_values, coords={case_dimension: test_ds[case_dimension], 'channel': [0], 'y': np.arange(y_dim), 'x': np.arange(x_dim)}, dims=(case_dimension, 'channel', 'y', 'x'))

            test_ds[var] = expanded_var             
 
    if args.continue_training:
        parameters_path = os.path.join(args.model_folder, "parameters.json")
        with open(parameters_path) as f:
            parameters = json.loads(f.read())

        if parameters["type"] == "ConvAEModel":
            mt = ConvAEModel()
        elif parameters["type"] == "UNET":
            mt = UNET()
        elif parameters["type"] == "UNET_RES":
            mt = UNET_RES_Train()            
        elif parameters["type"] == "SRCNN_RES":
            mt = SRCNN_RES_Train()     
        elif parameters["type"] == "RESUNET_GAN":
            mt = RESUNET_GAN()             
        elif parameters["type"] == "VAEUNET":
            mt = VAEUNET()
        elif parameters["type"] == "VAE":
            mt = VAE_Train()            
        elif parameters["type"] == "LinearModel":
            mt = LinearModel()
        mt.load(args.model_folder)
        # update selected parameters from the command line args
        mt.nr_epochs = args.nr_epochs
        mt.lr = args.learning_rate
        mt.batch_size = args.batch_size
    else:
        if args.method == "conv":
            mt = ConvAEModel(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                             batch_size=args.batch_size, lr=args.learning_rate)
        elif args.method == "var":
            mt = VAEUNET(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs, lambda_mse=args.lambda_mse, lambda_l1=args.lambda_l1, lambda_pearson=args.lambda_pearson, lambda_ssim=args.lambda_ssim, lambda_additional=args.lambda_additional, lambda_kl=args.lambda_kl,
                        batch_size=args.batch_size, lr=args.learning_rate,lr_step_size=args.lr_step_size,lr_gamma=args.lr_gamma, scheduler_type=args.scheduler_type, weight_decay=args.weight_decay, database_path=args.database_path, additional_loss_type=args.additional_loss)
        elif args.method == "vae":
            mt = VAE_Train(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs, lambda_mse=args.lambda_mse, lambda_l1=args.lambda_l1, lambda_pearson=args.lambda_pearson, lambda_ssim=args.lambda_ssim, lambda_additional=args.lambda_additional, lambda_kl=args.lambda_kl,
                        batch_size=args.batch_size, lr=args.learning_rate,lr_step_size=args.lr_step_size,lr_gamma=args.lr_gamma, scheduler_type=args.scheduler_type, weight_decay=args.weight_decay, database_path=args.database_path, additional_loss_type=args.additional_loss)            
        elif args.method == "unet":
            mt = UNET(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                             batch_size=args.batch_size, lr=args.learning_rate,lambda_l1=args.lambda_l1,lambda_pearson=args.lambda_pearson,                                                database_path=args.database_path,weight_decay=args.weight_decay,dropout_rate=args.dropout_rate)
        elif args.method == "unet_res":
            mt = UNET_RES_Train(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs, lambda_mse=args.lambda_mse, lambda_l1=args.lambda_l1, lambda_pearson=args.lambda_pearson, lambda_ssim=args.lambda_ssim, lambda_additional=args.lambda_additional, lambda_kl=args.lambda_kl,
                        batch_size=args.batch_size, lr=args.learning_rate,lr_step_size=args.lr_step_size,lr_gamma=args.lr_gamma, scheduler_type=args.scheduler_type, weight_decay=args.weight_decay, database_path=args.database_path, additional_loss_type=args.additional_loss)            
        elif args.method == "srcnn_res":
            mt = SRCNN_RES_Train(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs, lambda_mse=args.lambda_mse, lambda_l1=args.lambda_l1, lambda_pearson=args.lambda_pearson, lambda_ssim=args.lambda_ssim, lambda_additional=args.lambda_additional, lambda_kl=args.lambda_kl,
                        batch_size=args.batch_size, lr=args.learning_rate,lr_step_size=args.lr_step_size,lr_gamma=args.lr_gamma, scheduler_type=args.scheduler_type, weight_decay=args.weight_decay, database_path=args.database_path, additional_loss_type=args.additional_loss)   
        elif args.method == "resunet_gan":
            mt = RESUNET_GAN(fc_size=args.fc_size, encoded_dim_size=args.latent_size, nr_epochs=args.nr_epochs,
                             batch_size=args.batch_size, lr=args.learning_rate, database_path=args.database_path)             
        elif args.method == "linear":
            mt = LinearModel(batch_size=args.batch_size, nr_epochs=args.nr_epochs, lr=args.learning_rate)

        if args.model_id:
            mt.set_model_id(args.model_id)

        # if specified, use the encoder/decoder layer specifications
        if args.layer_definitions_path:
            with open(args.layer_definitions_path) as f:
                spec = ModelSpec()
                spec.load(json.loads(f.read()))
                mt.spec = spec

    mt.train(args.input_variables, args.output_variable, training_ds=train_ds, testing_ds=test_ds, model_path=args.model_folder, training_paths=training_paths,
             testing_paths=test_paths)

if __name__ == '__main__':
    main()
