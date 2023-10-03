# cae_tools

A convolutional auto-encoder library for modelling image-to-image transformations

## Installation

Requires: pytorch, xarray, netcdf4

Either:

* Clone this repo, and add the `src` folder to your PYTHONPATH

Or:

* run `pip install -e .`


## CLI

To train:

```
train_cae test/data/16x16_256x256/train.nc test/data/16x16_256x256/test.nc --model-folder=/tmp/mymodel --input-variable=lowres --output-variable=hires --nr-epochs=500
```

To apply:

```
apply_cae test/data/16x16_256x256/test.nc /tmp/mymodel output.nc --input-variable=lowres
```

For help, run `train_cae --help` or `apply_cae --help`

## API

See source code for ConvAEModel for documentation 

```python

from cae_tools.models.conv_ae_model import ConvAEModel

train_path = "train.nc"
test_path = "test.nc"

# train the model to reconstruct variable "hires" from variable "lowres"
mt = ConvAEModel(fc_size=8, encoded_dim_size=4, nr_epochs=500)
mt.train("lowres", "hires", train_path, test_path)

# print a summary of the layers
mt.print_layer_summary()

# persist the model
mt.save("/tmp/mymodel")

# now reload the trained model to create estimates of the "hires" variable from the train/test datasets
mt2 = ConvAEModel("lowres", "hires")
mt2.load("/tmp/mymodel")
mt2.apply(train_path, "lowres", "train_scores.nc", "hires_estimate")
mt2.apply(test_path, "lowres", "test_scores.nc", "hires_estimate")
```

## Data formats

input and output data needs to be 4-dimensional, organised by (N,channel,y,x)

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