#!/bin/bash

# call the CLI to train, apply and evaluate with example data
# this should be run from the root directory of the repo!

here=`dirname $0`

NR_EPOCHS=20

# for each supported method, go through the train, apply, evaluate, retrain, apply, evaluate cycle

for method in linear conv var
do
  echo tests for $method
  echo training model
  train_cae --database-path $here/results.db --train-inputs $here/../data/circle/16x16_256x256/train.nc --test-inputs $here/../data/circle/16x16_256x256/test.nc --method $method --model-folder=$here/results/$method/mymodel --latent-size=8 --learning-rate 0.0005 --batch-size 20 --fc-size=32 --kernel-size=3 --stride=2 --input-variables lowres --output-variable=hires --nr-epochs=$NR_EPOCHS

  echo applying trained model to training and test datasets
  apply_cae $here/../data/circle/16x16_256x256/train.nc $here/results/$method/train_scores.nc --model-folder=$here/results/$method/mymodel --input-variables lowres --prediction-variable hires_estimate
  apply_cae $here/../data/circle/16x16_256x256/test.nc $here/results/$method/test_scores.nc  --model-folder=$here/results/$method/mymodel --input-variables lowres --prediction-variable hires_estimate

  echo evaluating trained model
  evaluate_cae $here/results/$method/train_scores.nc $here/results/$method/test_scores.nc $here/results/$method/model_evaluation.html --input-variables lowres --output-variable hires --model-folder=$here/results/$method/mymodel --prediction-variable hires_estimate

  echo retrain model
  train_cae --database-path $here/results.db --train-inputs $here/../data/circle/16x16_256x256/train.nc --test-inputs $here/../data/circle/16x16_256x256/test.nc --continue-training --input-variables lowres --output-variable=hires --model-folder=$here/results/$method/mymodel --nr-epochs=$NR_EPOCHS

  echo applying retrained model to training and test datasets
  apply_cae $here/../data/circle/16x16_256x256/train.nc $here/results/$method/retrain_scores.nc --model-folder=$here/results/$method/mymodel --input-variables lowres --prediction-variable hires_estimate
  apply_cae $here/../data/circle/16x16_256x256/test.nc $here/results/$method/retest_scores.nc  --model-folder=$here/results/$method/mymodel --input-variables lowres --prediction-variable hires_estimate

  echo evaluating retrained model
  evaluate_cae $here/results/$method/retrain_scores.nc $here/results/$method/retest_scores.nc $here/results/$method/model_evaluation_retrained.html --input-variables lowres --output-variable hires --model-folder=$here/results/$method/mymodel --prediction-variable hires_estimate
done