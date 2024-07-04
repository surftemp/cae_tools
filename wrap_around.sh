#!/bin/bash

iterations=5

for ((i=1; i<=iterations; i++))
do
    echo "Iteration $i"
    #  train_apply.sh with all inputs
    echo "running train_apply_vae_newdata"
    ./train_apply_vae_newdata.sh
    
    # if train_apply.sh executed successfully
    if [ $? -ne 0 ]; then
        if [ $? -eq 2 ]; then
            # NaN detected, retry this iteration
            echo "NaN detected, retrying iteration $i"
            continue
        else
            # Other errors
            echo "train_apply_vae.sh failed on iteration $i"
            exit 1
        fi
    fi
done

echo "All iterations completed successfully"
