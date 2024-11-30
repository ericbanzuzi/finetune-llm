import argparse
from argparse import ArgumentParser
import torch
from unsloth import FastLanguageModel

parser = ArgumentParser()
# Model arguments
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--load_in_4bit", type=bool, default=True) 
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--r", type=int, default=16, help='Controls the capacity of the LoRA layers. A higher value increases the expressiveness of the added parameters but also increases computational cost. Suggested: 8, 16, 32, 64, 128') 
parser.add_argument("--lora_dropout", type=int, default=0, help='The dropout rate for the LoRA layers')
parser.add_argument("--lora_alpha", type=int, default=16, help='Scales the LoRA updates before adding them to the original weights, balancing the contribution of LoRA updates during training')

# Training arguments
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='Number of steps to accumulate gradients before performing a backward pass.') 
parser.add_argument("--warmup_steps", type=int, default=0, help='Number of steps during which the learning rate is gradually increased from 0 to its initial value.')
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=int, default=2e-4)
parser.add_argument("--weight_decay", type=int, default=0.01, help= 'Regularization term that penalizes large weights to prevent overfitting. Encourages smaller weights in the mode')
parser.add_argument("--lr_scheduler_type", type=str, default='linear')

# Dataset
parser.add_argument("--dataset", type=str, default='mlabonne/FineTome-100k')


