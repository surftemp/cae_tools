nrEpochs=500
learningRate=0.001
batchSize=32
fcSize=8192
latentSize=2056
method="unet"
layerDefinitionsPath="spec_skip_urban.json"
databasePath="database_v2_urbanexp.db"

hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
echo "Output Hash: $hash"

trainFolder="/home/jovyan/data_folder/train/" 
trainPaths=()
for file in ${trainFolder}*cleaned.nc; do     trainPaths+=("$file"); done
trainPathsString="${trainPaths[@]}"

testFolder="/home/jovyan/data_folder/test/" 
testPaths=()
for file in ${testFolder}*cleaned.nc; do     testPaths+=("$file"); done
testPathsString="${testPaths[@]}"

train_cae --train-inputs ${trainPathsString}     --test-inputs ${testPathsString}     --model-folder "/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall urban_area suburban_area albedo   --output-variable="ST_slices"     --nr-epochs="$nrEpochs"     --learning-rate="$learningRate"     --batch-size="$batchSize"     --fc-size="$fcSize"     --latent-size="$latentSize"     --method="$method"     --layer-definitions-path="$layerDefinitionsPath"  --database-path=$databasePath

apply_cae ${trainPathsString} "/lustre_scratch/shaerdan/scores/train_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall urban_area suburban_area albedo    --prediction-variable="hires_estimate"

apply_cae ${testPathsString} "/lustre_scratch/shaerdan/scores/test_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall urban_area suburban_area albedo   --prediction-variable="hires_estimate"
