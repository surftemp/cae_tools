#!/bin/bash

# call the CLI to train, apply and evaluate with example data
# this should be run from the root directory of the repo!

here=`dirname $0`

echo training CAE model
train_cae $here/../data/circle/16x16_256x256/train.nc $here/../data/circle/16x16_256x256/test.nc --model-folder=$here/results/mymodel --latent-size=8 --learning-rate 0.0005 --batch-size 20 --fc-size=32 --input-variables lowres --output-variable=hires --nr-epochs=500
exit 0

echo applying trained model to training and test datasets
apply_cae $here/../data/circle/16x16_256x256/train.nc $here/results/train_scores.nc --model-folder=$here/results/mymodel --input-variables lowres --prediction-variable hires_estimate
apply_cae $here/../data/circle/16x16_256x256/test.nc $here/results/test_scores.nc  --model-folder=$here/results/mymodel --input-variables lowres --prediction-variable hires_estimate

echo evaluating trained model
evaluate_cae $here/results/train_scores.nc $here/results/test_scores.nc $here/results/model_evaluation.html --input-variables lowres --output-variable hires --model-folder=$here/results/mymodel --prediction-variable hires_estimate

echo continue training CAE model
train_cae $here/../data/circle/16x16_256x256/train.nc $here/../data/circle/16x16_256x256/test.nc --continue-training --input-variables lowres --output-variable=hires --model-folder=$here/results/mymodel --nr-epochs=500

echo applying retrained model to training and test datasets
apply_cae $here/../data/circle/16x16_256x256/train.nc $here/results/train_scores.nc --model-folder=$here/results/mymodel --input-variables lowres --prediction-variable hires_estimate
apply_cae $here/../data/circle/16x16_256x256/test.nc $here/results/test_scores.nc  --model-folder=$here/results/mymodel --input-variables lowres --prediction-variable hires_estimate

echo evaluating retrained model
evaluate_cae $here/results/train_scores.nc $here/results/test_scores.nc $here/results/model_evaluation_retrained.html --input-variables lowres --output-variable hires --model-folder=$here/results/mymodel --prediction-variable hires_estimate
