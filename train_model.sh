#!/bin/bash	

cd /home/rosameliacarioni/finetune-llm/
export PYTHONPATH=$PWD
echo "-- TRAINING STARTS --"
python3 finetuning/train_models.py \
    --model_name unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
    --batch_size 64 \
    --num_train_epochs 2 \
    --verbose \
    --dataset 'arcee-ai/infini-instruct-top-500k' 
echo "-- TRAINING DONE --"