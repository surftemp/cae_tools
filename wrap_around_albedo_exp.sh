#!/bin/bash

iterations=5

for ((i=1; i<=iterations; i++))
do
    echo "Iteration $i"
    #  train_apply.sh with all inputs
    echo "running train_apply"
    ./train_apply.sh
    
    # if train_apply.sh executed successfully
    if [ $? -ne 0 ]; then
        echo "train_apply.sh failed on iteration $i"
        exit 1
    fi

    # train_apply_1.sh with no albedo
    echo "running train_apply_no_albedo"
    ./train_apply_no_albedo.sh
    
    # if train_apply_1.sh executed successfully
    if [ $? -ne 0 ]; then
        echo "train_apply_no_albedo.sh failed on iteration $i"
        exit 1
    fi
done

echo "All iterations completed successfully"
