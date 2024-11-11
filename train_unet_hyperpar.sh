nrEpochs=6000
learningRate=0.001
lr_step_size=500
lr_gamma=0.5
lambda_mse=1
lambda_ssim=0
lambda_pearson=0.0005
lambda_l1=0
lambda_kl=0
lambda_additional=0
weight_decay=1e-5
dropout_rate=0.1
batchSize=512
fcSize=3200        # 8192
latentSize=800      # 2048
method="unet"                     # "conv","unet" ,"resunet","var"
# additional_loss="variance"         # 'contrastive', 'histogram', 'perceptual', 'variance' 'None'
scheduler_type="None"        # 'StepLR', 'ReduceLROnPlateau', 'ExponentialLR','CosineAnnealingLR'
layerDefinitionsPath="spec_august_ver.json"   
databasePath="database_v23_wo_ssrd.db"

select_files() {
    folder=$1
    percentage=$2
    # List all .nc files in the folder
    mapfile -t total_files < <(ls ${folder}*.nc)
    # Get the total number of files
    total_count=${#total_files[@]}

    # Calculate the number of files to select
    selected_count=$((total_count * percentage / 100))

    if [ $selected_count -gt $total_count ]; then
        selected_count=$total_count
    fi

    # Randomly select the calculated number of unique files
    mapfile -t selected_files < <(shuf -n ${selected_count} -e "${total_files[@]}")

    # Return the selected files
    echo "${selected_files[@]}"
}

trainFolder="/lustre_scratch/shaerdan/data_folder/train_v3/train/processed_train/"  
# Select 10% of the files from the train folder
trainPaths=($(select_files $trainFolder 100))
trainPathsString=$(printf "%s " "${trainPaths[@]}")

testFolder="/lustre_scratch/shaerdan/data_folder/test_v3/test/processed_test/"
# Select 20% of the files from the test folder
testPaths=($(select_files $testFolder 100))
testPathsString=$(printf "%s " "${testPaths[@]}")


hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
echo "Output Hash: $hash"


train_cae --train-inputs ${trainPathsString}     --test-inputs ${testPathsString}     --model-folder "/lustre_scratch/shaerdan/models/model_$hash"     --input-variables land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction era5_u10 era5_v10 era5_ssrd   --output-variable="ST_slices"     --nr-epochs="$nrEpochs"     --learning-rate="$learningRate"  --lr-step-size="$lr_step_size" --lr-gamma="$lr_gamma" --scheduler-type="$scheduler_type" --lambda-mse="$lambda_mse" --lambda-l1="$lambda_l1" --lambda-kl="$lambda_kl" --lambda-ssim="$lambda_ssim" --lambda-pearson="$lambda_pearson" --weight-decay="$weight_decay"  --dropout-rate="$dropout_rate" --batch-size="$batchSize"     --fc-size="$fcSize"     --latent-size="$latentSize"     --method="$method"     --layer-definitions-path="$layerDefinitionsPath"  --database-path=$databasePath --lambda-additional="$lambda_additional" #--additional-loss="$additional_loss"    

apply_cae ${trainPathsString} "/lustre_scratch/shaerdan/scores/train_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction era5_u10 era5_v10 era5_ssrd  --prediction-variable="hires_estimate"

apply_cae ${testPathsString} "/lustre_scratch/shaerdan/scores/test_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction era5_u10 era5_v10 era5_ssrd --prediction-variable="hires_estimate"


# nrEpochs=2000
# learningRate=0.001
# lr_step_size=500
# lr_gamma=0.5
# lambda_mse=1
# lambda_ssim=0.1
# lambda_pearson=0.001
# lambda_l1=0
# lambda_kl=0.1
# lambda_additional=0.1
# weight_decay=0
# batchSize=256
# fcSize=3200        # 8192
# latentSize=800      # 2048
# method="unet"                     # "conv","unet" ,"resunet","var"
# # additional_loss="variance"         # 'contrastive', 'histogram', 'perceptual', 'variance' 'None'
# scheduler_type="None"        # 'StepLR', 'ReduceLROnPlateau', 'ExponentialLR','CosineAnnealingLR'
# layerDefinitionsPath="spec_others.json"   
# databasePath="database_v19_gan.db"


# hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
# echo "Output Hash: $hash"



# train_cae --train-inputs ${trainPathsString}     --test-inputs ${testPathsString}     --model-folder "/lustre_scratch/shaerdan/models/model_$hash"     --input-variables land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction era5_u10 era5_v10  --output-variable="ST_slices"     --nr-epochs="$nrEpochs"     --learning-rate="$learningRate"  --lr-step-size="$lr_step_size" --lr-gamma="$lr_gamma" --scheduler-type="$scheduler_type" --lambda-mse="$lambda_mse" --lambda-l1="$lambda_l1" --lambda-kl="$lambda_kl" --lambda-ssim="$lambda_ssim" --lambda-pearson="$lambda_pearson" --weight-decay="$weight_decay" --batch-size="$batchSize"     --fc-size="$fcSize"     --latent-size="$latentSize"     --method="$method"     --layer-definitions-path="$layerDefinitionsPath"  --database-path=$databasePath --lambda-additional="$lambda_additional" #--additional-loss="$additional_loss"    

# apply_cae ${trainPathsString} "/lustre_scratch/shaerdan/scores/train_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction era5_u10 era5_v10   --prediction-variable="hires_estimate"

# apply_cae ${testPathsString} "/lustre_scratch/shaerdan/scores/test_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy cos_doy slope_magnitude slope_direction era5_u10 era5_v10 --prediction-variable="hires_estimate"



# nrEpochs=1000
# learningRate=0.001
# lr_step_size=500
# lr_gamma=0.5
# lambda_mse=1
# lambda_ssim=0.1
# lambda_pearson=0.1
# lambda_l1=0
# lambda_kl=0.1
# lambda_additional=0.1
# weight_decay=1e-8
# batchSize=256
# fcSize=2048        # 8192
# latentSize=512      # 2048
# method="unet"                     # "conv","unet" ,"resunet","var"
# # additional_loss="variance"         # 'contrastive', 'histogram', 'perceptual', 'variance' 'None'
# scheduler_type="None"        # 'StepLR', 'ReduceLROnPlateau', 'ExponentialLR','CosineAnnealingLR'
# layerDefinitionsPath="spec_others.json"   
# databasePath="database_v16_unet_vs_unetres.db"


# hash=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
# echo "Output Hash: $hash"



# train_cae --train-inputs ${trainPathsString}     --test-inputs ${testPathsString}     --model-folder "/lustre_scratch/shaerdan/models/model_$hash"     --input-variables land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy  --output-variable="ST_slices"     --nr-epochs="$nrEpochs"     --learning-rate="$learningRate"  --lr-step-size="$lr_step_size" --lr-gamma="$lr_gamma" --scheduler-type="$scheduler_type" --lambda-mse="$lambda_mse" --lambda-l1="$lambda_l1" --lambda-kl="$lambda_kl" --lambda-ssim="$lambda_ssim" --lambda-pearson="$lambda_pearson" --weight-decay="$weight_decay" --batch-size="$batchSize"     --fc-size="$fcSize"     --latent-size="$latentSize"     --method="$method"     --layer-definitions-path="$layerDefinitionsPath"  --database-path=$databasePath --lambda-additional="$lambda_additional" #--additional-loss="$additional_loss"    

# apply_cae ${trainPathsString} "/lustre_scratch/shaerdan/scores/train_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy   --prediction-variable="hires_estimate"

# apply_cae ${testPathsString} "/lustre_scratch/shaerdan/scores/test_scores_$hash.nc"     --model-folder="/lustre_scratch/shaerdan/models/model_$hash"     --input-variables land_cover albedo_monthly_climatology_means elevation era5_skt sin_doy --prediction-variable="hires_estimate"