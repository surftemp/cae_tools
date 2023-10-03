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

## API

See source code for ConvAEModel for documentation

```python

from cae_tools.models.conv_ae_model import ConvAEModel

train_path = "train.nc"
test_path = "test.nc"
# train the model to reconstruct variable "hires" from variable "lowres"
mt = ConvAEModel(fc_size=8, encoded_dim_size=4, nr_epochs=500)
mt.train("lowres", "hires", train_path, test_path)
mt.save("/tmp/mymodel")

# now use the trained model to create estimates of the "hires" variable from the train/test datasets
mt2 = ConvAEModel("lowres", "hires")
mt2.load("/tmp/mymodel")
mt2.predict(train_path, "lowres", "train_scores.nc", "hires_estimate")
mt2.predict(test_path, "lowres", "test_scores.nc", "hires_estimate")
```

