here=`dirname $0`

NR_EPOCHS=500

echo training model
train_cae --database-path $here/results.db --train-inputs $here/../data/circle/16x16_256x256/train.nc --test-inputs $here/../data/circle/16x16_256x256/test.nc --method conv --model-folder=$here/results/$method/mymodel --latent-size=8 --learning-rate 0.0005 --nr-epochs 2000 --batch-size 20 --fc-size=32 --kernel-size=3 --stride=2 --input-variables lowres --output-variable=hires --nr-epochs=$NR_EPOCHS
# apply_cae $here/../data/circle/16x16_256x256/test.nc $here/results/$method/mymodel_scores.nc --model-folder=$here/results/$method/mymodel --prediction-variable estimate
# evaluate_cae --database-path $here/results.db --train-inputs $here/../data/circle/16x16_256x256/train.nc --test-inputs $here/../data/circle/16x16_256x256/test.nc  --output-html-path=html_test.html --model-folder=$here/results/$method/mymodel --prediction-variable estimate
evaluate_cae --database-path $here/results.db --test-inputs $here/results/$method/mymodel_scores.nc  --output-html-path=html_test.html --model-folder=$here/results/$method/mymodel --prediction-variable estimate
