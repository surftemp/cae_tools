#!/bin/bash

here=`dirname $0`

NR_EPOCHS=100

echo training model
train_cae --database-path $here/results.db --train-inputs $here/../data/circle/16x16_256x256/train.nc --test-inputs $here/../data/circle/16x16_256x256/test.nc --method conv --model-folder=$here/results/$method/mymodel --latent-size=8 --learning-rate 0.0005 --batch-size 20 --fc-size=32 --kernel-size=3 --stride=2 --input-variables lowres --output-variable=hires --nr-epochs=$NR_EPOCHS
echo applying model to score test data
apply_cae $here/../data/circle/16x16_256x256/test.nc $here/results/$method/mymodel_scores.nc --model-folder=$here/results/$method/mymodel --prediction-variable estimate
echo evaluating model based on original test and train data
evaluate_cae --database-path $here/results.db --train-inputs $here/../data/circle/16x16_256x256/train.nc --test-inputs $here/../data/circle/16x16_256x256/test.nc  --output-html-folder=html_test --model-folder=$here/results/$method/mymodel --prediction-variable estimate
echo evaluating model based on test score results
evaluate_cae --database-path $here/results.db --test-inputs $here/results/$method/mymodel_scores.nc  --output-html-folder=html_test_train --model-folder=$here/results/$method/mymodel --prediction-variable estimate

