nrEpochs=1000
learningRate=0.001
lr_step_size=500
lr_gamma=0.5
lambda_mse=1
lambda_ssim=0.5
lambda_pearson=1
lambda_l1=0.001
lambda_kl=0.1
lambda_additional=0
weight_decay=1e-5
batchSize=256
fcSize=8192        # 8192
latentSize=2048     # 2048
method="conv"
# additional_loss="variance"         # 'contrastive', 'histogram', 'perceptual', 'variance' 'None'
scheduler_type="None"        # 'StepLR', 'ReduceLROnPlateau', 'ExponentialLR','CosineAnnealingLR'
layerDefinitionsPath="spec_skip_vae_newdata.json"   
databasePath="database_v12_vae.db"

hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
echo "Output Hash: $hash"

trainFolder="/lustre_scratch/shaerdan/data_folder/train/" 
trainPaths=()
for file in ${trainFolder}*.nc; do     trainPaths+=("$file"); done
trainPathsString="${trainPaths[@]}"
# # #test path: 
trainPathsString="/lustre_scratch/shaerdan/data_folder/train/train_259_29.nc"

testFolder="/lustre_scratch/shaerdan/data_folder/test/" 
testPaths=()
for file in ${testFolder}*.nc; do     testPaths+=("$file"); done
testPathsString="${testPaths[@]}"
# # #test path: 
testPathsString="/lustre_scratch/shaerdan/data_folder/test/test_289_12.nc"

train_cae --train-inputs ${trainPathsString}     --test-inputs ${testPathsString}     --model-folder "/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall land_cover albedo_monthly_climatology_means  --output-variable="ST_slices"     --nr-epochs="$nrEpochs"     --learning-rate="$learningRate"  --lr-step-size="$lr_step_size" --lr-gamma="$lr_gamma" --scheduler-type="$scheduler_type" --lambda-mse="$lambda_mse" --lambda-l1="$lambda_l1" --lambda-kl="$lambda_kl" --lambda-ssim="$lambda_ssim" --lambda-pearson="$lambda_pearson" --weight-decay="$weight_decay" --batch-size="$batchSize"     --fc-size="$fcSize"     --latent-size="$latentSize"     --method="$method"     --layer-definitions-path="$layerDefinitionsPath"  --database-path=$databasePath --lambda-additional="$lambda_additional" #--additional-loss="$additional_loss"    

apply_cae ${trainPathsString} "/lustre_scratch/shaerdan/scores/train_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall land_cover albedo_monthly_climatology_means    --prediction-variable="hires_estimate"

apply_cae ${testPathsString} "/lustre_scratch/shaerdan/scores/test_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall land_cover albedo_monthly_climatology_means   --prediction-variable="hires_estimate"
