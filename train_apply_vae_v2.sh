nrEpochs=4000
learningRate=0.001
lambda_tv=0.00001
lambda_pearson=2
lambda_additional=0.1
weight_decay=1e-5
batchSize=512
fcSize=8192
latentSize=2056
method="var"
additional_loss="variance"         # 'contrastive', 'histogram', 'perceptual', 'variance'
layerDefinitionsPath="spec_skip_resnet.json"   
databasePath="database_v8_resnet.db"

hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
echo "Output Hash: $hash"

trainFolder="/lustre_scratch/shaerdan/data_folder/train/" 
trainPaths=()
for file in ${trainFolder}*cleaned.nc; do     trainPaths+=("$file"); done
trainPathsString="${trainPaths[@]}"
# # #test path: 
# trainPathsString="/lustre_scratch/shaerdan/data_folder/train/train_230_cleaned.nc"

testFolder="/lustre_scratch/shaerdan/data_folder/test/" 
testPaths=()
for file in ${testFolder}*cleaned.nc; do     testPaths+=("$file"); done
testPathsString="${testPaths[@]}"
# # #test path: 
# testPathsString="/lustre_scratch/shaerdan/data_folder/test/test_202_cleaned.nc"

train_cae --train-inputs ${trainPathsString}     --test-inputs ${testPathsString}     --model-folder "/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall land_cover albedo   --output-variable="ST_slices"     --nr-epochs="$nrEpochs"     --learning-rate="$learningRate"  --lambda-tv="$lambda_tv" --lambda-pearson="$lambda_pearson" --weight-decay="$weight_decay" --batch-size="$batchSize"     --fc-size="$fcSize"     --latent-size="$latentSize"     --method="$method"     --layer-definitions-path="$layerDefinitionsPath"  --database-path=$databasePath --additional-loss="$additional_loss"    --lambda-additional="$lambda_additional"

apply_cae ${trainPathsString} "/lustre_scratch/shaerdan/scores/train_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall land_cover albedo    --prediction-variable="hires_estimate"

apply_cae ${testPathsString} "/lustre_scratch/shaerdan/scores/test_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall land_cover albedo   --prediction-variable="hires_estimate"
