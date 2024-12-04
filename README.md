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
# Setting up a training environment in GCP with Unsloth
## 1. Create a Google and GCP account
If you don't already have a Google account, [create one here](https://accounts.google.com/signup). Next, sign up for Google Cloud Platform (GCP) at [GCP Console](https://console.cloud.google.com/).

## 2. Set Up a New GCP Project and Enable Billing
1. Log in to the [GCP Console](https://console.cloud.google.com/).
2. Create a new project:
   - Click on the **Select a project** dropdown in the top navigation bar.
   - Click **New Project**, fill in the details, and click **Create**.
3. Enable billing for the project:
   - Navigate to **Billing** in the left-hand menu.
   - Link your project to a billing account.

## 3. Enable Required APIs
To ensure the smooth deployment of resources, enable the necessary APIs:
1. Go to **APIs & Services > Library** in the GCP Console.
2. Enable the following APIs:
   - **Compute Engine API**
   - **Cloud Deployment Manager V2 API**
   - **Cloud Runtime Configuration API**

## 4. Add Quota
To ensure the smooth deployment of resources, enable the necessary APIs:
1. Go to **IAM & Admin > Quotas & System Limits** in the GCP Console.
2. Request a GPU quota by editing the value for **GPU (all regions)** from 0 to 1. If you choose more than 1, it is likely that the request will be denied. This is part of the **Compute Engine API** engine.
3. Wait for the Quota to be approved. This might take a few minutes.

## 5. Find the Deep Learning VM template from GCP Market place
1. Go to **Market Place** and search for "deep learning vm", or optionally click [follow this link](https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance).
2. Follow the instructions after pressing **Launch**.

## 6. Selecting VM specifications
1. Select a zone for the VM. Some zones might not have available resources so you might have to try different ones. An overview of zones and their GPUs can be found [here](https://cloud.google.com/compute/docs/gpus) 
2. Use the following setups for a VM instance with a T4 GPU:
    - **Machine Type:** Choose a GPU-enabled machine, such as `n1-highmem-2` with a compatible GPU (e.g., NVIDIA Tesla T4).
    - **Framework:** Choose a Pytorch with CUDA 12.1 or 12.4 and Python 3.10.
    - Check the option for "Install NVIDIA GPU driver automatically on first startup?" 
    - Check the option for "Enable access to JupyterLab via URL instead of SSH. (Beta)" 
2. 

## 7. 



---

# Setting up VM Environment with Python Dependencies

## Option 1: Python Environment

### 1. Create a Python Virtual Environment
Run the following command to create a virtual environment:
```bash
python3.10 -m venv .unsloth_env
```

### 2. Activate the Virtual Environment
**Linux/macOS:**
  ```bash
  source .unsloth_env/bin/activate
  ```

### 3. Follow the Tutorial: Install Dependencies and Unsloth
Refer to the [Unsloth installation guide](https://ridgerunai.medium.com/how-to-fine-tune-llms-with-unsloth-and-hugging-face-2a25f2a7cd00). Below are the summarized steps:

#### 3.1. Check Your CUDA Version
Run the following command to determine your CUDA version:
```bash
nvcc --version
```


#### 3.2. Install Compatible PyTorch Version
Use the [PyTorch Compatibility Matrix](https://pytorch.org/get-started/previous-versions/) to find the appropriate version based on your CUDA setup. Example for CUDA 12.4:
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

#### 3.3. Upgrade `pip`
Upgrade to the latest version of `pip`:
```bash
pip install --upgrade pip
```

#### 3.4. Install Unsloth
Refer to the [Unsloth GitHub README](https://github.com/unslothai/unsloth/blob/main/README.md) for the appropriate installation command. 

For example, if using **CUDA 12.4** and **PyTorch 2.5.0**, run:
```bash
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

## Option 2: Conda Environment

### 1. Create a Conda Virtual Environment
Run the following command to create an environment:
```bash
conda create --name unsloth_env python=3.11 pytorch-cuda=12.4 cudatoolkit -c pytorch -c nvidia -y
```

### 2. Activate the Environment
Activate the new Conda environment:
```bash
conda activate unsloth_env
```

### 3. Install Dependencies
Install **unsloth** and other dependencies:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

--- 