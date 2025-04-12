#!/bin/bash

iterations=5

for ((i=1; i<=iterations; i++))
do
    # train_apply_1.sh with no albedo
    echo "running train_apply_unet, iteration $i"
    ./train_apply_unet.sh
    
    # if train_apply_1.sh executed successfully
    if [ $? -ne 0 ]; then
        echo "train_apply_unet.sh failed on iteration $i"
        exit 1
    fi

done

echo "All iterations completed successfully"
