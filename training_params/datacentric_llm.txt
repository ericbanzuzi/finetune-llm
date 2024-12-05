#!/bin/bash	

cd /home/ericbanzuzi/finetune-llm/
export PYTHONPATH=$PWD
echo "-- TRAINING STARTS --"
python3 finetuning/train_models.py \
    --model_name unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
    --batch_size 8 \
    --num_train_epochs 2 \
    --verbose \
    --dataset 'arcee-ai/The-Tome'  # using a subset of 200K rows with seed 42
echo "-- TRAINING DONE --"