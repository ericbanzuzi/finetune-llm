#!/bin/bash	

cd /home/rosameliacarioni/finetune-llm/
export PYTHONPATH=$PWD
echo "-- TRAINING STARTS --"
python3 train_models.py \
    --model_name unsloth/Llama-3.2-1B-Instruct-bnb-4bit \
    --batch_size 2 \
    --num_train_epochs 2 \
    --verbose \
    --hf <username> \
    --hf_token <token> \
    --hf_model_name test_script_llm \
    --hf_gguf
echo "-- TRAINING DONE --"