#!/bin/bash

# call the CLI to train, apply and evaluate with example data
# this should be run from the root directory of the repo!

echo training CAE model
train_cae test/data/16x16_256x256/train.nc test/data/16x16_256x256/test.nc --model-folder=./mymodel --input-variable=lowres --output-variable=hires --nr-epochs=500

echo applying model to training and test datasets
apply_cae test/data/16x16_256x256/train.nc train_scores.nc --model-folder=./mymodel --input-variable=lowres --prediction-variable hires_estimate
apply_cae test/data/16x16_256x256/test.nc test_scores.nc  --model-folder=./mymodel --input-variable=lowres --prediction-variable hires_estimate

echo evaluating
evaluate_cae train_scores.nc test_scores.nc evaluation.html --input-variable lowres --output-variable hires --model-folder=./mymodel --prediction-variable hires_estimate