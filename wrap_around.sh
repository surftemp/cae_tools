#!/bin/bash

iterations=5

for ((i=1; i<=iterations; i++))
do

    echo "running train_unet_res"
    ./train_unet_res.sh
    
    if [ $? -ne 0 ]; then
        echo "train_unet_res.sh failed on iteration $i"
        exit 1
    fi  

    echo "running train_unet, iteration $i"
    ./train_unet.sh
    
    if [ $? -ne 0 ]; then
        echo "train_unet.sh failed on iteration $i"
        exit 1
    fi  

done

echo "All iterations completed successfully"
