nrEpochs=800
learningRate=0.001
batchSize=128
fcSize=8192
latentSize=2056
method="unet"
layerDefinitionsPath="spec_skip_unet.json"
databasePath="database_v100_unet_sst.db"

hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
echo "Output Hash: $hash"

trainPathsString="/lustre_scratch/shaerdan/data_folder/train_sst/data_slice_added_channel.nc"

# testFolder="/lustre_scratch/shaerdan/data_folder/test/" 
# testPaths=()
# for file in ${testFolder}*cleaned.nc; do     testPaths+=("$file"); done
# testPathsString="${testPaths[@]}"
# #test path: 
testPathsString="/lustre_scratch/shaerdan/data_folder/train_sst/data_slice_added_channel.nc"

train_cae --train-inputs ${trainPathsString}     --test-inputs ${testPathsString}     --model-folder "/lustre_scratch/shaerdan/models/model_$hash"     --input-variables sstL3   --output-variable="sstL4"     --nr-epochs="$nrEpochs"     --learning-rate="$learningRate"     --batch-size="$batchSize"     --fc-size="$fcSize"     --latent-size="$latentSize"     --method="$method"     --layer-definitions-path="$layerDefinitionsPath"  --database-path=$databasePath

# apply_cae ${trainPathsString} "/lustre_scratch/shaerdan/scores/train_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall land_cover albedo    --prediction-variable="hires_estimate"

# apply_cae ${testPathsString} "/lustre_scratch/shaerdan/scores/test_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables tasmax tasmin rainfall land_cover albedo   --prediction-variable="hires_estimate"
