#!/bin/bash

iterations=5

for ((i=1; i<=iterations; i++))
do
    echo "Iteration $i"
    #  train_apply.sh with all inputs
    echo "running train_apply_landcover"
    ./train_apply_landcover.sh
    
    # if train_apply.sh executed successfully
    if [ $? -ne 0 ]; then
        echo "train_apply.sh failed on iteration $i"
        exit 1
    fi

    # train_apply_1.sh with no albedo
    echo "running train_apply_urban_frac"
    ./train_apply_urban_frac.sh
    
    # if train_apply_1.sh executed successfully
    if [ $? -ne 0 ]; then
        echo "train_apply_no_albedo.sh failed on iteration $i"
        exit 1
    fi
done

echo "All iterations completed successfully"
