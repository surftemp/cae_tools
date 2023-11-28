#!/bin/bash

# call the CLI to train, apply and evaluate with example data
# this should be run from the root directory of the repo!

here=`dirname $0`

echo training CAE model
train_cae $here/../data/circle/16x16_256x256/train.nc $here/../data/circle/16x16_256x256/test.nc --model-folder=$here/results/mymodel --latent-size=4 --fc-size=16 --input-variable=lowres --output-variable=hires --nr-epochs=500

echo applying model to training and test datasets
apply_cae $here/../data/circle/16x16_256x256/train.nc $here/results/train_scores.nc --model-folder=$here/results/mymodel --input-variable=lowres --prediction-variable hires_estimate
apply_cae $here/../data/circle/16x16_256x256/test.nc $here/results/test_scores.nc  --model-folder=$here/results/mymodel --input-variable=lowres --prediction-variable hires_estimate

echo evaluating
evaluate_cae $here/results/train_scores.nc $here/results/test_scores.nc $here/results/model_evaluation.html --input-variable lowres --output-variable hires --model-folder=$here/results/mymodel --prediction-variable hires_estimate