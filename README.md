# cae_tools

A convolutional auto-encoder library for modelling image-to-image transformations

## Dependencies

`train_cae` and `apply_cae` requires: pytorch, xarray, netcdf4, dask

`evaluate_cae` requires xarray, netcdf4, pillow, seaborn, matplotlib

A suitable conda environment can be obtained using:

```
conda create -n pyt python=3.8 pytorch xarray dask netcdf4 pillow seaborn matplotlib scipy
conda activate pyt
```

## Installation

Install cae_tools into a python or conda environment containing the above dependencies, then:

```
git clone git@github.com:surftemp/cae_tools.git
cd cae_tools`
pip install -e .
```

## CLI

To train:

```
train_cae --train-inputs test/data/16x16_256x256/train.nc --test-inputs test/data/16x16_256x256/test.nc --model-folder=/tmp/mymodel --input-variables lowres --output-variable=hires --nr-epochs=500 --method conv       
```
method options are "conv" or "var" for convolutional autoencder or variational autoencoder, "linear" for a simple linear model

Useful training parameters:

| Name | Description                                                                                      |
|------|--------------------------------------------------------------------------------------------------|
|--latent-size | size of the latent layer (compressed representation) in neurons, defaults to 4                   |
 |--fc-size | size of the fully connected layers immediately before and after the latent layer, defaults to 16 |
| --batch-size | size of each training batch, defaults to 10                                                      |
| --learning-rate | the optimisers learning rate, defaults to 0.001                                                  |
| --stride | stride to use in convolutional layers", defaults to 2                                            |
| --kernel-size | kernel size to use in convolutional layers, defaults to 3                                        |
| --input-layer-count | number of input convolutional layers, defaults to auto (reduce image size to approximately 3x3)  |
| --output-layer-count | number of output convolutional layers, defaults to auto (reduce image size to approximately 3x3)  |

To apply:

```
apply_cae test/data/16x16_256x256/train.nc train_scores.nc --model-folder=/tmp/mymodel  --input-variables lowres --prediction-variable hires_estimate
apply_cae test/data/16x16_256x256/test.nc test_scores.nc --model-folder=/tmp/mymodel  --input-variables lowres --prediction-variable hires_estimate
```

To evaluate:

```
evaluate_cae train_scores.nc test_score.nc evaluation.html --input-variables lowres --output-variable hires --model-folder=/tmp/mymodel --prediction-variable hires_estimate
```

To retrain an existing model, use `--continue-training` (note that training parameters are reused from the model and cannot be set on the command line):

```
train_cae test/data/16x16_256x256/train.nc test/data/16x16_256x256/test.nc --continue-training --model-folder=/tmp/mymodel --input-variables lowres --output-variable=hires --nr-epochs=500
```

For help, run `train_cae --help` or `apply_cae --help`

## API

See source code for ConvAEModel for documentation 

```python
import xarray as xr
from cae_tools.models.conv_ae_model import ConvAEModel
from cae_tools.models.model_evaluator import ModelEvaluator

train_path = "train.nc"
test_path = "test.nc"

# train the model to reconstruct variable "hires" from variable "lowres"
mt = ConvAEModel(fc_size=8, encoded_dim_size=4, nr_epochs=500)

train_ds = xr.open_dataset(train_path)
test_ds = xr.open_dataset(test_path)
mt.train(["lowres"], "hires", training_ds=train_ds, testing_ds=test_ds, training_paths=train_path, testing_paths=test_path)

# print a summary of the layers
print(mt.summary())

# persist the model
mt.save("/tmp/mymodel")

# now reload the trained model to create estimates of the "hires" variable from the train/test datasets
mt2 = ConvAEModel()
mt2.load("/tmp/mymodel")

mt2.apply(train_ds, ["lowres"], "hires_estimate")
train_ds.to_netcdf( "train_scores.nc")

mt2.apply(test_ds, ["lowres"], "hires_estimate")
test_ds.to_netcdf("test_scores.nc")

# perform model evaluation, write results to evaluation_results.html
me = ModelEvaluator("train_scores.nc","test_scores.nc",["lowres"],"hires","evaluation_results.html","hires_estimate","/tmp/mymodel")
me.run()
```

## Data formats

input and output data needs to be 4-dimensional, organised by (N,channel,y,x)

training and test data should be supplied in separate files.

Example file format for training or test data

[netcdf4 files for model train or test](test/data/circle/16x16_256x256)

## Example

Inputs are low resolution 16x16, for example:

![image](https://github.com/surftemp/cae_tools/assets/58978249/885a4377-1b2c-4940-acd7-a663ef0b5233)

Outputs are high resolution 256x256, for example:

![image](https://github.com/surftemp/cae_tools/assets/58978249/3c1a57a8-5c21-4dc8-b61f-7eb91e9691a0)

Model trained on input-output pairs can then reconstruct the high resolution outputs from low resolution inputs:

![image](https://github.com/surftemp/cae_tools/assets/58978249/a9b357a2-7117-4c64-8763-a9d4b7139c17)

## Known Limitations

### CLI

Not all API options are available in the CLI tools `apply_cae` and `train_cae`

### API

* Currently not possible to control the number of input (encoder) and output (decoder) layers, these are decided automatically
* Need proper documentation of the API parameters that exist

### Testing

* model retraining not tested
* multiple channel input or output data not tested
* not yet tested with a range of input/output geometries (especially non-square)

To run all unit tests, first run the test data generator (only one set of synthetic data is stored in the repo)

```
python test/datagen/gen.py
```

Run unit tests using:

```
python tests/unittests/quick.py
```
