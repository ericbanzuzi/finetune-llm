#!/bin/bash	

cd /home/rosameliacarioni/finetune-llm/
export PYTHONPATH=$PWD
echo "-- TRAINING STARTS --"
python3 finetuning/train_models.py \
    --model_name unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
    --batch_size 64 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --verbose
echo "-- TRAINING DONE --"