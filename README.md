# Fine-Tune LLM with Unsloth

---
## Blabla - notekeeping  
TODO 

unsloth/Llama-3.2-3B-bnb-4bi
---
## Overview 
TODO 

---

## Finetuning strategies 

TODO

---
## Results

TODO


---
## BETTER TITLE - TODO 

### Environment

#### 1. Create a Python virtual environment
Run the following command to create a virtual environment:
```bash
python3.10 -m venv .unsloth_env
```

#### 2. Activate the virtual environment
- **Linux/macOS:**
  ```bash
  source .unsloth_env/bin/activate
  ```

#### 3. Follow the tutorial
Refer to the full tutorial [here](https://ridgerunai.medium.com/how-to-fine-tune-llms-with-unsloth-and-hugging-face-2a25f2a7cd00). Below are the key steps summarized.

## Install dependencies and Unsloth

#### 1. Check Your CUDA Version
Run the following command to determine your CUDA version:
```bash
nvcc --version
```

#### 2. Find Compatible PyTorch Versions
Visit the [PyTorch Previous Versions page](https://pytorch.org/get-started/previous-versions/) to find the appropriate PyTorch version for your CUDA setup. 

For example, for **CUDA 12.4**, use:
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

#### 3. Verify the Installation
Check that PyTorch and CUDA are correctly installed:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
This should output:
```
True
```

#### 4. Upgrade `pip`
Upgrade to the latest version of `pip`:
```bash
pip install --upgrade pip
```

#### 5. Find the Correct Installation Command
Refer to the [Unsloth GitHub README](https://github.com/unslothai/unsloth/blob/main/README.md) for the appropriate installation command. 

For example, if using **CUDA 12.4** and **PyTorch 2.5.0**, run:
```bash
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

--- 