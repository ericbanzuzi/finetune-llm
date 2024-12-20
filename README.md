# Fine-Tune LLM with Unsloth

## Overview  
This project focuses on **Parameter-Efficient Fine-Tuning (PEFT)** of a **Large Language Model (LLM)** using **Unsloth**, a framework designed to optimize the fine-tuning process for LLMs. It uses **Low-Rank Adaptation (LoRA)** to inject trainable low-rank matrices into the model's layers, enabling fine-tuning with fewer parameters while retaining the model's generalization capabilities. This method reduces memory usage and computational overhead, making it feasible to fine-tune large models even on limited resources.  

The fine-tuned LLMs are deployed using a **Gradio UI** hosted on **Hugging Face Spaces**, allowing for an interactive way to explore the model's capabilities. 

### Challenges and Solutions
During the setup, we faced issues installing **Unsloth** due to dependency compatibility problems. To overcome this, we identified and documented 2 alternative solutions, included at the end of the README. One of these uses a Python environment and the second one uses a Conda environment. 

To manage the computational demands of training, we used **Google Cloud Platform (GCP)**, taking advantage of a free credit gift card received during sign-up. The README also includes step-by-step instructions on setting up the GCC environment, including integrating it with **VS Code** for development and training.  

### Model and Datasets  
- **Model**: we used the open-source model **[Llama-3.2-1B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-bnb-4bit)** as a base for fine-tuning.  
- **Datasets**: the project used two datasets depending on the fine-tuning strategy:  
  - **[Fine Tome 100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)**. 
  - **[The Tome](https://huggingface.co/datasets/arcee-ai/The-Tome)**.

--- 

## Access to chatbots and models
1. [Baseline fine-tuned chatbot](https://huggingface.co/spaces/ericbanzuzi/finetuned-llm) and [model](https://huggingface.co/rcarioniporras/model_baseline_llama).
2. [Data-centric fine-tuned chatbot](https://huggingface.co/spaces/ericbanzuzi/the-tome-llm) and [model](https://huggingface.co/ericbanzuzi/model_datacentric_llama_gguf).
3. [Model-centric fine-tuned chatbot](https://huggingface.co/spaces/rcarioniporras/finetuned-llm) and [model](https://huggingface.co/rcarioniporras/model_modelcentric_llama_gguf).

---  

## Finetuning Strategies  
To optimize the performance of our fine-tuned LLM, we explored 2 strategies:  

### 1. **Data-Centric**  
In the data-centric approach, we aimed to improve model performance by increasing the amount of training data. Initially, we fine-tuned the model using the **Fine Tome 100k** dataset. To further improve the model's ability to generalize and respond better to diverse instructions, we scaled up the dataset by taking a subset of 200K samples from **The Tome** dataset while keeping all other hyperparameters constant. This larger dataset provided a broader range of instruction styles and content, which increased the exposure of the model to varied contexts. We chose to use **The Tome** dataset since it is the same dataset **Fine Tome 100k** is subsampled from.

### 2. **Model-Centric**
The model-centric approach focuses on modifying some of the hyperparameters from the baseline to maximize the model's performance without modifying the underlying data. The goal was to fine-tune the training process and balance the trade-off between learning and overfitting, resulting in a more robust and generalizable model. Given the computational demand of the training, we only tested one set of parameter modifications, which can be seen next:

1. **Increased Training Time**:
   - Extended the training duration from **2 epochs** to **3 epochs** to allow the model to learn more complex representations.
   - This modification was based on the fact that additional epochs could improve learning, as long as overfitting is addressed.

2. **Increased Weight Decay**:
   - Raised the weight decay value from **0.01** to **0.025**.
   - Weight decay serves as a regularization technique, penalizing large weights to reduce the risk of overfitting. We decided to increase it to reduce the chances of overfitting, given the longer training time that we used.
  
3. **Increased Warm-Up Steps**:
   - Increased the number of warm-up steps from **0** to **5**.
   - Warm-up steps increase the learning rate from 0 to its initial value (2e-4) over the specified number of steps. We decided to change it from 0 to 5 because they help the model adjust gradually to the training process by avoiding sudden large weight updates at the start. This can prevent instability, especially in the initial stages when the model might otherwise make big updates due to high learning rates.

---

### Results

To evaluate the performance of our fine-tuned models, we used a performance metric derived from scoring their answers to a set of 10 questions we made. These questions were designed to assess the models' ability to follow instructions with a focus on mathematics and programming tasks. The questions and answers files can be found under `/testing`. 

The evaluation was done using ChatGPT-4o as a grader. For each question, ChatGPT-4 scored the correctness of the models' answers on a scale from 1 to 10, where:
- 10 indicated the response was as accurate and complete as possible,
- 1 indicated the response was completely incorrect.

The table below summarizes the train loss and average performance metric for the three models:

| Model          | Train Loss | Performance Metric |
|----------------|------------|-----------|
| **Baseline**   | 0.79360    | 6.3       |
| **Model-Centric** | 0.76986    | 5.7       |
| **Data-Centric**  | 0.86005    | 7.6       |

> [!IMPORTANT]
> - **Data-Centric Fine-Tuning** achieved the highest performance metric score (7.6), indicating that increasing the dataset size and exposing the model to more diverse instruction styles improved its ability to respond accurately to the evaluation questions.
> - **Model-Centric Fine-Tuning** had the lowest performance metric score (5.7) but achieved the best training loss (0.76). While the hyperparameter adjustments improved the training loss, we believe they may have led to overfitting. For instance, after looking at the results from the test set, we saw that, on question 8, which involved a simple probability calculation, this model failed to produce the correct result, whereas the other two models succeeded.


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

1. **Install the VS Code Extensions**:
   - Open VS Code and install the following extensions:
     - **Remote - SSH**: For connecting to your GCP VM over SSH.
     - **Python**: For Python development.
     - **Pylance**: For python type checking.

2. **Set Up SSH Access**:
   - In your terminal, set up an SSH connection:
     ```bash
     gcloud compute ssh [INSTANCE_NAME] --project [PROJECT_ID] --zone [ZONE]
     ```
     Replace `[INSTANCE_NAME]`, `[PROJECT_ID]`, and `[ZONE]` with your VM details. If a key `~/.ssh/google_compute_engine` does not exist, it will be generated for you. Otherwise it will be used as the ssh key file for the project.

4. **Connect to the VM in VS Code**:
   - Open VS Code and click the green "><" icon in the bottom-left corner.
   - Select **Remote-SSH: Connect to Host...**.
   - Add your GCP VM's SSH configuration. For example, if your VM's IP is `203.0.113.10`, your `~/.ssh/config` should look like:
     ```
     Host dl-vm
         HostName 203.0.113.10
         User your-username
         IdentityFile ~/.ssh/google_compute_engine
     ```
   - Choose the `dl-vm` host to connect. VS Code will set up the connection. Sometimes you might be required to press Retry a couple of times before the connections is established.

You are now set up to develop on GCP using VS Code, with direct access to your VM!


> [!TIP] 
> To get started with your VM, we have listed some usefult commands
> 
> **Start VM:** `gcloud compute instances start [INSTANCE_NAME] --zone [ZONE] --project [PROJECT_ID]`
> 
> **Stop VM:** `gcloud compute instances stop [INSTANCE_NAME] --zone [ZONE] --project [PROJECT_ID]`
> 
> **Copy from Local to VM::** `gcloud compute scp --recurse /path/to/local/file [INSTANCE_NAME]:/path/to/destination --zone [ZONE]  --project [PROJECT_ID]`
>
> **Copy from VM to Local:** `gcloud compute scp --recurse [INSTANCE_NAME]:/path/to/remote/file /path/to/local/destination --zone [ZONE] --project [PROJECT_ID]`
>
> 

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


