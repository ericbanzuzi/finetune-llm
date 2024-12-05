# Fine-Tune LLM with Unsloth

## Overview  
This project focuses on **Parameter-Efficient Fine-Tuning (PEFT)** of a **Large Language Model (LLM)** using **Unsloth**, a framework designed to optimize the fine-tuning process for LLMs. It uses **Low-Rank Adaptation (LoRA)** to inject trainable low-rank matrices into the model's layers, enabling fine-tuning with fewer parameters while retaining the model's generalization capabilities. This method reduces memory usage and computational overhead, making it feasible to fine-tune large models even on limited resources.  

The fine-tuned LLMs are deployed using a **Gradio UI** hosted on **Hugging Face Spaces**, allowing for an interactive way to explore the model's capabilities. 

### Challenges and Solutions
During the setup, we faced issues installing **Unsloth** due to dependency compatibility problems. To overcome this, we identified and documented 2 alternative solutions, included at the end of the README. One of this uses Python environment and the second one Conda environment. 

To manage the computational demands of training, we used **Google Cloud Computing (GCC)**, taking advantage of a free credit gift card received during sign-up. The README also includes step-by-step instructions on setting up the GCC environment, including integrating it with **VS Code** for development and training.  

### Model and Datasets  
- **Model**: we used the open-source model **[Llama-3.2-1B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-bnb-4bit)** as a base for fine-tuning.  
- **Datasets**: the project used two datasets depending on the fine-tuning strategy:  
  - **[Fine Tome 100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)**. 
  - **[Fine Tome 500k](https://huggingface.co/datasets/arcee-ai/infini-instruct-top-500k)**.

--- 

## Access to chatbots
1. [Baseline fine-tuned model](https://huggingface.co/spaces/ericbanzuzi/finetuned-llm)
2. Data-centric fine-tuned model:
3. Model-centric fine-tuned model:
---  

## Finetuning Strategies  
To optimize the performance of our fine-tuned LLM, we explored 2 strategies:  

### 1. **Data-Centric**  
In the data-centric approach, we aimed to improve model performance by increasing the amount of training data. Initially, we fine-tuned the model using the **Fine Tome 100k** dataset. To further improve the model's ability to generalize and respond better to diverse instructions, we scaled up to the **Fine Tome 500k** dataset while keeping all other hyperparameters constant. This larger dataset provided a broader range of instruction styles and content, which increased the exposure of the model to varied contexts. 

### 2. **Model-Centric**  
The model-centric approach focused on fine-tuning the hyperparameters to maximize the model's performance without modifying the underlying data. We experimented with various hyperparameter configurations, including:  
- **TODO**: 

---
## Results

TODO


---
# Setting up a training environment in GCP with Unsloth
### 1. Create a Google and GCP account
If you don't already have a Google account, [create one here](https://accounts.google.com/signup). Next, sign up for Google Cloud Platform (GCP) at [GCP Console](https://console.cloud.google.com/).

### 2. Set Up a New GCP Project and Enable Billing
1. Log in to the [GCP Console](https://console.cloud.google.com/).
2. Create a new project:
   - Click on the **Select a project** dropdown in the top navigation bar.
   - Click **New Project**, fill in the details, and click **Create**.
3. Enable billing for the project:
   - Navigate to **Billing** in the left-hand menu.
   - Link your project to a billing account.

### 3. Enable Required APIs
To ensure the smooth deployment of resources, enable the necessary APIs:
1. Go to **APIs & Services > Library** in the GCP Console.
2. Enable the following APIs:
   - **Compute Engine API**
   - **Cloud Deployment Manager V2 API**
   - **Cloud Runtime Configuration API**

### 4. Add Quota
To ensure the smooth deployment of resources, enable the necessary APIs:
1. Go to **IAM & Admin > Quotas & System Limits** in the GCP Console.
2. Request a GPU quota by editing the value for **GPU (all regions)** from 0 to 1. If you choose more than 1, it is likely that the request will be denied. This is part of the **Compute Engine API** engine.
3. Wait for the Quota to be approved. This might take a few minutes.

### 5. Find the Deep Learning VM template from GCP Market place
1. Go to **Market Place** and search for "deep learning vm", or optionally click [follow this link](https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance).
2. Follow the instructions after pressing **Launch**.

### 6. Selecting VM specifications
1. Select a zone for the VM. Some zones might not have available resources so you might have to try different ones. An overview of zones and their GPUs can be found [here](https://cloud.google.com/compute/docs/gpus) 
2. Use the following setups for a VM instance with a T4 GPU:
    - **Machine Type:** Choose a GPU-enabled machine, such as `n1-highmem-2` with a compatible GPU (e.g., NVIDIA Tesla T4).
    - **Framework:** Choose a Pytorch with CUDA 12.1 or 12.4 and Python 3.10.
    - Check the option for "Install NVIDIA GPU driver automatically on first startup?" 
    - Check the option for "Enable access to JupyterLab via URL instead of SSH. (Beta)" 
2. Press deploy and wait for the VM instance to start.

### 7. Start and Install the gcloud CLI

Once your VM is up and running, you’ll need to set up the `gcloud` command-line interface (CLI) to interact with your GCP project.

1. **Install the gcloud CLI**:
   - Open a terminal on your local machine or in the VM.
   - Follow the [official instructions](https://cloud.google.com/sdk/docs/install) for your operating system. For Mac with brew, you can use:
     ```bash
     brew install --cask google-cloud-sdk 
     ```
   - Confirm the installation by running:
     ```bash
     gcloud --version
     ```

2. **Authenticate the gcloud CLI**:
   - Log in to your Google account:
     ```bash
     gcloud auth login
     ```
     This will open a browser for you to authenticate. If you’re using a VM without a browser, use `gcloud auth login --no-launch-browser` and follow the terminal instructions.

3. **Set up the project**:
   - Select your project:
     ```bash
     gcloud config set project [PROJECT_ID]
     ```
     Replace `[PROJECT_ID]` with your actual GCP project ID.

4. **Verify the gcloud CLI setup**:
   - Test connectivity to your VM:
     ```bash
     gcloud compute instances list
     ```
     This command lists all instances in your project, verifying that `gcloud` can connect.

### 8. Set Up VS Code for Your GCP Environment

Using VS Code can streamline your development and allow you to work efficiently with the GCP environment.

1. **Install VS Code**:
   - Download and install Visual Studio Code from the [official website](https://code.visualstudio.com/).

2. **Install the VS Code Extensions**:
   - Open VS Code and install the following extensions:
     - **Remote - SSH**: For connecting to your GCP VM over SSH.
     - **Python**: For Python development.
     - **Pylance**: For python type checking.

3. **Set Up SSH Access**:
   - In your terminal, set up an SSH key pair (if not already set up):
     ```bash
     ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
     ```
     Save the key to the default location (`~/.ssh/id_rsa`) or specify another path.
   - Add the public key to your GCP project:
     ```bash
     gcloud compute ssh [INSTANCE_NAME] --project [PROJECT_ID] --zone [ZONE] --ssh-key-file ~/.ssh/id_rsa.pub
     ```
     Replace `[INSTANCE_NAME]`, `[PROJECT_ID]`, and `[ZONE]` with your VM details.

4. **Connect to the VM in VS Code**:
   - Open VS Code and click the green "><" icon in the bottom-left corner.
   - Select **Remote-SSH: Connect to Host...**.
   - Add your GCP VM's SSH configuration. For example, if your VM's IP is `203.0.113.10`, your `~/.ssh/config` should look like:
     ```
     Host gcp-vm
         HostName 203.0.113.10
         User your-username
         IdentityFile ~/.ssh/id_rsa
     ```
   - Choose the `gcp-vm` host to connect. VS Code will set up the connection.

You are now set up to develop on GCP using VS Code, with direct access to your VM!



### Other useful gcloud cli commands

TODO:

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
conda create --name unsloth_env python=3.11 pytorch-cuda=12.4 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
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
