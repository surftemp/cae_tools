# cae_tools

A convolutional auto-encoder library for modelling image-to-image transformations

## Installation

Requires: pytorch, xarray, netcdf4

Either:
* Clone this repo, and add the `src` folder to your PYTHONPATH

Or:
* run `pip install .`

## CLI

To train:

```
train_cae test/data/16x16_256x256/train.nc test/data/16x16_256x256/test.nc --model-path=/tmp/foo --input-variable=lowres --output-variable=hires --nr-epochs=10
```

To apply:

```
apply_cae test/data/16x16_256x256/test.nc /tmp/foo output.nc --input-variable=lowres
```


## API

See source code for ConvAEModel for documentation

```python

from cae_tools.models.conv_ae_model import ConvAEModel

train_path = "train.nc"
test_path = "test.nc"
mt = ConvAEModel("lowres", "hires", fc_size=8, encoded_dim_size=4, nr_epochs=500)
mt.train(train_path, test_path)
mt.save("/tmp/foo")

mt2 = ConvAEModel("lowres", "hires")
mt2.load("/tmp/foo")
mt2.predict(train_path, "train_scores.nc")
mt2.predict(test_path, "test_scores.nc")
```