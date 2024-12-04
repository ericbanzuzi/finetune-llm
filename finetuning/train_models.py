import argparse
import torch
import os
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only

WEIGHTS_DIR = "weights"
SAVE_STEPS = 10 # save every x steps 

# TODO: add documentation
def get_parser():
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", type=bool, default=True, help='Use 4bit quantization to reduce memory usage. Can be False.') 
    parser.add_argument("--model_name", type=str, required=True, help='Choose one of the following models: [unsloth/Llama-3.2-1B-bnb-4bit, unsloth/Llama-3.2-1B-Instruct-bnb-4bit, unsloth/Llama-3.2-3B-bnb-4bit, unsloth/Llama-3.2-3B-Instruct-bnb-4bit]')
    parser.add_argument("--r", type=int, default=16, help='Controls the capacity of the LoRA layers. A higher value increases the expressiveness of the added parameters but also increases computational cost. Suggested: 8, 16, 32, 64, 128') 
    parser.add_argument("--lora_dropout", type=int, default=0, help='The dropout rate for the LoRA layers')
    parser.add_argument("--lora_alpha", type=int, default=16, help='Scales the LoRA updates before adding them to the original weights, balancing the contribution of LoRA updates during training')

    # Training arguments
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='Number of steps to accumulate gradients before performing a backward pass.') # DO NOT MODIFY THIS 
    parser.add_argument("--warmup_steps", type=int, default=0, help='Number of steps during which the learning rate is gradually increased from 0 to its initial value.')
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=int, default=2e-4)
    parser.add_argument("--weight_decay", type=int, default=0.01, help= 'Regularization term that penalizes large weights to prevent overfitting. Encourages smaller weights in the mode')
    parser.add_argument("--lr_scheduler_type", type=str, default='linear')
    parser.add_argument("--hf", type=str, default=None, help='If you want to save the model to HuggingFace, set hf to your username.')
    parser.add_argument("--hf_token", type=str, default=None, help='If you want to save the model to HuggingFace, set hf_token to your token from https://huggingface.co/settings/tokens')
    parser.add_argument("--hf_model_name", type=str, default='model', help='Model name to be used in HuggingFace')
    parser.add_argument('--hf_gguf', default=False, action='store_true', help='Store HuggingFace model as gguf')
    parser.add_argument('--hf_push', default=False, action='store_true', help='Store HuggingFace model')


    # Dataset
    parser.add_argument("--dataset", type=str, default='mlabonne/FineTome-100k')
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help='Print more verbose output')
    return parser


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }


def get_model(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name, 
        max_seq_length = args.max_seq_length, 
        dtype = None, # None for auto detection.
        load_in_4bit = args.load_in_4bit
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 42,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model, tokenizer


def get_dataset(dataset):
    dataset = load_dataset(dataset, split = "train")
    dataset = standardize_sharegpt(dataset)
    return dataset.map(formatting_prompts_func, batched = True)


def get_tokenizer_from_chat_template(tokenizer):
     # TODO: maybe 3.2?
    return get_chat_template(tokenizer, chat_template = "llama-3.1")


def get_trainer(model, tokenizer, dataset, args):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_steps = args.warmup_steps,
            num_train_epochs = args.num_train_epochs,
            # max_steps = 6, # TODO: delete
            learning_rate = args.learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = args.weight_decay,
            lr_scheduler_type = args.lr_scheduler_type,
            seed = 42,
            report_to = "none", # Use this for WandB etc
            save_strategy="steps",
            save_steps = SAVE_STEPS,
            save_total_limit=2, # keep only the 2 most recent checkpoints 
            output_dir = WEIGHTS_DIR
        )
    )

    # TODO: learn the meaning of this method
    return train_on_responses_only(
        trainer, 
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
        
    model, tokenizer = get_model(args)
    if args.verbose:
        print('-- Model created --')

    dataset = get_dataset(args.dataset)
    tokenizer = get_tokenizer_from_chat_template(tokenizer)
    if args.verbose:
        print('-- Dataset and tokenizer created --')

    trainer = get_trainer(model, tokenizer, dataset, args)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    if args.verbose:
        print('-- Start training --')
    trainer_stats = trainer.train()
    if args.verbose:
        print('-- Training done --')

    if args.verbose:
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    model.save_pretrained('model')
    tokenizer.save_pretrained('model')

    if args.hf and args.hf_token:
        if args.hf_gguf:
            model.push_to_hub_gguf(f'{args.hf}/{args.hf_model_name}', tokenizer = tokenizer, token = args.hf_token, quantization_method = "q4_k_m")
        else:
            model.push_to_hub(f'{args.hf}/{args.hf_model_name}', token = args.hf_token) # Online saving
            tokenizer.push_to_hub(f'{args.hf}/{args.hf_model_name}', token = args.hf_token) # Online saving
    
    if args.verbose:
        print('-- Model saved --')

