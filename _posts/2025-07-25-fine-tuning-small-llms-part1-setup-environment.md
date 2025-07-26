---
author: Saptak
categories:
- AI
- Machine Learning
- Docker
- LLM
- Fine-tuning
date: 2025-07-25 09:00:00 -0800
description: Part 1 of our comprehensive series on fine-tuning small language models.
  Learn how to set up Docker Desktop with Model Runner and prepare your development
  environment.
featured_image: /assets/images/llm-fine-tuning-part1.jpg
header_image_path: /assets/img/blog/headers/2025-07-25-fine-tuning-small-llms-part1-setup-environment.jpg
image_credit: Photo by Tai Bui on Unsplash
layout: post
part: 1
repository: https://github.com/saptak/fine-tuning-small-llms
series: Fine-Tuning Small LLMs with Docker Desktop
tags:
- llm
- fine-tuning
- docker
- docker-model-runner
- setup
- environment
thumbnail_path: /assets/img/blog/thumbnails/2025-07-25-fine-tuning-small-llms-part1-setup-environment.jpg
title: 'Fine-Tuning Small LLMs on a Desktop - Part 1: Setup and Environment'
toc: true
---

> 📚 **Reference Code Available**: All code examples from this blog series are available in the [GitHub repository](https://github.com/saptak/fine-tuning-small-llms). Clone it to follow along!

# Fine-Tuning Small LLMs on a Desktop - Part 1: Setup and Environment

Welcome to our comprehensive 6-part series on fine-tuning small language models on a desktop! In this first part, I'll establish the foundation by setting up your development environment and understanding the key concepts.

## Series Overview

This series will take you from zero to having a production-ready fine-tuned language model:

1. **Part 1: Setup and Environment** (This post)
2. [Part 2: Data Preparation and Model Selection](/writing/2025/07/25/fine-tuning-small-llms-part2-data-preparation/)
3. [Part 3: Fine-Tuning with Unsloth](/writing/2025/07/25/fine-tuning-small-llms-part3-training/)
4. [Part 4: Evaluation and Testing](/writing/2025/07/25/fine-tuning-small-llms-part4-evaluation/)
5. [Part 5: Deployment with Ollama and Docker](/writing/2025/07/25/fine-tuning-small-llms-part5-deployment/)
6. [Part 6: Production, Monitoring, and Scaling](/writing/2025/07/25/fine-tuning-small-llms-part6-production/)

## Why Fine-Tune Small Language Models?

Before diving into setup, let's understand why fine-tuning small language models has become increasingly popular:

### Cost Efficiency
Using API-based models like GPT-4 can quickly become expensive for high-volume applications. A fine-tuned 8B parameter model running locally eliminates per-token costs and can handle thousands of requests per day at zero marginal cost.

### Data Privacy and Control
Your sensitive data never leaves your infrastructure. This is crucial for:
- Healthcare applications with patient data
- Financial services with confidential information
- Internal corporate tools with proprietary data
- Government and defense applications

### Specialized Performance
A small model fine-tuned on your specific domain often outperforms larger general-purpose models. For example:
- A 7B model fine-tuned on SQL data can outperform GPT-4 for database queries
- A medical-fine-tuned model excels at clinical documentation
- Code-specific models generate better domain-specific solutions

### Reduced Latency
Local inference eliminates network latency, providing:
- Sub-second response times
- Offline capability
- Consistent performance regardless of internet connectivity

### Vendor Independence
Avoid vendor lock-in and maintain control over your AI capabilities regardless of external service changes, pricing models, or availability.

## Prerequisites and System Requirements

### Hardware Requirements

**Minimum Configuration:**
- 8GB RAM (16GB strongly recommended)
- Modern CPU with at least 4 cores
- 50GB free disk space
- Stable internet connection for initial downloads

**Recommended Configuration:**
- 16GB+ RAM (32GB ideal for larger models)
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- SSD storage for faster data loading and model access
- 100GB+ free disk space for models and datasets

**GPU Compatibility:**
- **NVIDIA**: RTX 20/30/40 series, Tesla, Quadro
- **AMD**: Limited support through ROCm (experimental)
- **Apple Silicon**: Full support through Metal Performance Shaders
- **Intel Arc**: Emerging support (check latest compatibility)

### Software Prerequisites

**Operating System Support:**
- **macOS**: macOS 12+ (Apple Silicon strongly recommended for GPU acceleration)
- **Windows**: Windows 10/11 with WSL2 enabled
- **Linux**: Ubuntu 20.04+, RHEL 8+, or equivalent distributions

**Required Software:**
- Docker Desktop (latest version)
- Python 3.8+ (Python 3.10 recommended)
- Git for version control
- VS Code with Python extension (recommended IDE)

## Setting Up Docker Desktop

### Step 1: Install Docker Desktop

1. **Download Docker Desktop** from [docker.com/products/docker-desktop](https://docker.com/products/docker-desktop)

2. **Platform-specific installation:**

   **macOS:**
   ```bash
   # Using Homebrew (recommended)
   brew install --cask docker

   # Or download DMG and install manually
   ```

   **Windows:**
   ```powershell
   # Ensure WSL2 is installed first
   wsl --install

   # Then install Docker Desktop from the downloaded installer
   ```

   **Linux:**
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Add user to docker group
   sudo usermod -aG docker $USER
   ```

3. **Start Docker Desktop** and ensure it's running

4. **Verify installation:**
   ```bash
   docker --version
   docker run hello-world
   ```

### Step 2: Enable Docker Model Runner

Docker Model Runner is a new feature that makes running LLMs locally incredibly simple. Here's how to enable it:

1. **Open Docker Desktop**
2. **Navigate to Settings** → **Beta Features**
3. **Enable the following options:**
   - ✅ "Docker Model Runner"
   - ✅ "Host-side TCP support" (for API access)
   - ✅ "GPU-backed inference" (if you have a compatible GPU)
4. **Click Apply & Restart**

After enabling these features, you'll see a new "Models" tab in Docker Desktop.

### Step 3: Verify Docker Model Runner

```bash
# Check if Model Runner is available
docker model --help

# You should see output like:
# Usage: docker model COMMAND
#
# Manage models
#
# Commands:
#   list     List models
#   pull     Pull a model
#   push     Push a model to a registry
#   remove   Remove a model
#   run      Run a model
```

### Step 4: Check GPU Availability

If you have an NVIDIA GPU, you can check if it's properly configured by running the `nvidia-smi` command:

```bash
nvidia-smi
```

This command should output a table with information about your GPU, including the driver version, CUDA version, and a list of running processes.

### Step 5: Pull Your First Model

Let's test the setup by pulling a small model:

```bash
# Pull a lightweight model for testing
docker model pull ai/smollm2:360M-Q4_K_M

# List available models
docker model list

# Test the model
docker model run ai/smollm2:360M-Q4_K_M "Hello, how are you?"
```

If this works, congratulations! Your Docker Model Runner is properly configured.

## Setting Up Python Environment

### Step 1: Create Project Structure

```bash
# Create project directory
mkdir llm-fine-tuning-workshop
cd llm-fine-tuning-workshop

# Create subdirectories
mkdir -p {data,models,notebooks,scripts,configs,logs}

# Create virtual environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Core Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch (choose based on your system)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For Apple Silicon Mac:
pip install torch torchvision torchaudio
```

### Step 3: Install Unsloth

Unsloth is our secret weapon for efficient fine-tuning. It provides up to 80% memory reduction and 2x speed improvements:

```bash
# For NVIDIA CUDA systems (Linux/Windows with NVIDIA GPU):
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# For Apple Silicon Macs (M1/M2/M3 chips):
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

# For CPU-only systems (Intel Macs, older hardware):
pip install "unsloth[cpu] @ git+https://github.com/unslothai/unsloth.git"
```

**Note for Apple Silicon Mac users**: Unsloth will automatically detect and use Metal Performance Shaders (MPS) for GPU acceleration when available. Make sure you have the latest PyTorch installed with MPS support.

### Step 4: Install Additional Dependencies

```bash
# Core ML libraries
pip install transformers accelerate datasets peft bitsandbytes

# Experiment tracking and monitoring
pip install wandb tensorboard

# Development and analysis tools
pip install jupyter notebook ipywidgets
pip install pandas numpy matplotlib seaborn plotly

# Evaluation libraries
pip install evaluate rouge-score bleu sacrebleu

# API and web interface tools
pip install fastapi uvicorn streamlit gradio

# Utility libraries
pip install tqdm rich typer click python-dotenv
```

### Step 5: Create Requirements File

```bash
# Generate requirements file
pip freeze > requirements.txt
```

### Step 6: Verify Installation

Create a test script to verify everything is working:

```python
# test_setup.py
import sys
import torch
import transformers
import unsloth
import pandas as pd
import numpy as np

print("🚀 Installation Verification")
print("=" * 40)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

# Check GPU support
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    Memory: {memory_gb:.1f} GB")
elif torch.backends.mps.is_available():
    print("✅ Apple Silicon GPU (MPS) available")
    print("   Training will use Metal Performance Shaders")
else:
    print("⚠️ No GPU acceleration available - will use CPU")

# Test Unsloth import
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth imported successfully")
except ImportError as e:
    print(f"❌ Unsloth import failed: {e}")

# Test device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"🎯 Recommended device for training: {device}")

print("\n🎉 Setup verification complete!")
```

Run the test:
```bash
python test_setup.py
```

## Installing Ollama

Ollama will help us serve our fine-tuned models locally with a simple API:

### Installation by Platform

**macOS:**
```bash
# Using Homebrew
brew install ollama

# Or download from website
curl -fsSL https://ollama.ai/install.sh | sh
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from [ollama.ai](https://ollama.ai/download) and run it.

### Verify Ollama Installation

```bash
# Check version
ollama --version

# Start Ollama service (in one terminal)
ollama serve

# In another terminal, test with a simple model
ollama run llama3.1:8b "Hello, world!"
```

## Understanding Docker Model Runner vs Ollama

You might wonder why we're using both Docker Model Runner and Ollama. Here's the key difference:

### Docker Model Runner
- **Native Integration**: Built into Docker Desktop
- **Performance**: Models run directly on host system (better performance)
- **Ecosystem**: Seamless integration with Docker containers and Compose
- **Port**: Uses port 12434
- **Best for**: Development workflows, containerized applications

### Ollama
- **Standalone Tool**: Independent model server
- **Community**: Larger model repository and community
- **Flexibility**: More customization options
- **Port**: Uses port 11434
- **Best for**: Production deployment, model experimentation

We'll use Docker Model Runner for development and Ollama for serving our final fine-tuned models.

## Project Structure Setup

Let's create a well-organized project structure:

```bash
# Create comprehensive project structure
mkdir -p llm-fine-tuning-workshop/{
  data/{raw,processed,datasets},
  models/{base,fine-tuned,quantized},
  notebooks/{exploration,training,evaluation},
  scripts/{preprocessing,training,evaluation,deployment},
  configs/{training,evaluation,deployment},
  logs/{training,evaluation,monitoring},
  tests/{unit,integration},
  docker/{images,compose},
  docs/{guides,api}
}

# Create configuration files
touch llm-fine-tuning-workshop/{
  .env,
  .gitignore,
  README.md,
  requirements.txt,
  Dockerfile,
  docker-compose.yml
}
```

### Create .gitignore

```bash
# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/
dist/
build/

# Models and Data
models/
*.bin
*.safetensors
*.gguf
*.ggml
data/raw/
data/processed/
*.csv
*.json
*.jsonl

# Logs and Outputs
logs/
outputs/
wandb/
*.log

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# System
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Docker
.dockerignore
EOF
```

### Create Environment Configuration

```bash
# .env
cat > .env << 'EOF'
# Model Configuration
BASE_MODEL="unsloth/llama-3.1-8b-instruct-bnb-4bit"
MAX_SEQ_LENGTH=2048
LOAD_IN_4BIT=true

# Training Configuration
LEARNING_RATE=2e-4
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
MAX_STEPS=500
WARMUP_STEPS=50

# API Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
DOCKER_MODEL_RUNNER_PORT=12434

# Paths
DATA_DIR=./data
MODELS_DIR=./models
LOGS_DIR=./logs

# Weights & Biases (optional)
WANDB_PROJECT=llm-fine-tuning
WANDB_ENTITY=your-username

# Hugging Face (optional)
HF_TOKEN=your-hf-token
EOF
```

## Testing the Complete Setup

Let's create a comprehensive test to ensure everything is working:

```python
# test_complete_setup.py
import os
import sys
import requests
import subprocess
from pathlib import Path
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_python_environment():
    """Test Python environment and packages"""
    print("🐍 Testing Python Environment")
    print("-" * 30)

    required_packages = [
        'torch', 'transformers', 'datasets', 'accelerate',
        'unsloth', 'bitsandbytes', 'peft', 'wandb'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    return True

def test_gpu_setup():
    """Test GPU availability and configuration"""
    print("\n🖥️  Testing GPU Setup")
    print("-" * 30)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA available with {gpu_count} GPU(s)")

        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {name} ({memory:.1f} GB)")
        return True
    else:
        print("⚠️  CUDA not available - will use CPU")
        return True  # Not a failure, just different configuration

def test_docker_setup():
    """Test Docker and Docker Model Runner"""
    print("\n🐳 Testing Docker Setup")
    print("-" * 30)

    try:
        # Test basic Docker
        result = subprocess.run(['docker', '--version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
        else:
            print("❌ Docker not available")
            return False

        # Test Docker Model Runner
        result = subprocess.run(['docker', 'model', '--help'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker Model Runner available")
        else:
            print("❌ Docker Model Runner not enabled")
            return False

        return True
    except FileNotFoundError:
        print("❌ Docker not found in PATH")
        return False

def test_ollama_setup():
    """Test Ollama installation and service"""
    print("\n🦙 Testing Ollama Setup")
    print("-" * 30)

    try:
        # Test Ollama binary
        result = subprocess.run(['ollama', '--version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama: {result.stdout.strip()}")
        else:
            print("❌ Ollama not available")
            return False

        # Test Ollama service
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama service running")
                models = response.json().get("models", [])
                print(f"   {len(models)} models available")
            else:
                print("⚠️  Ollama service not responding (run 'ollama serve')")
        except requests.exceptions.ConnectionError:
            print("⚠️  Ollama service not running (run 'ollama serve')")

        return True
    except FileNotFoundError:
        print("❌ Ollama not found in PATH")
        return False

def test_project_structure():
    """Test project directory structure"""
    print("\n📁 Testing Project Structure")
    print("-" * 30)

    required_dirs = [
        'data', 'models', 'notebooks', 'scripts',
        'configs', 'logs', 'tests', 'docker'
    ]

    missing_dirs = []
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/")
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        return False
    return True

def test_model_loading():
    """Test loading a small model"""
    print("\n🤖 Testing Model Loading")
    print("-" * 30)

    try:
        from unsloth import FastLanguageModel

        print("Loading small test model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/tinyllama-bnb-4bit",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        print("✅ Model loaded successfully")

        # Test tokenization
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization test passed ({len(tokens['input_ids'][0])} tokens)")

        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Complete Setup Testing")
    print("=" * 50)

    tests = [
        test_python_environment,
        test_gpu_setup,
        test_docker_setup,
        test_ollama_setup,
        test_project_structure,
        test_model_loading
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)

    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Your environment is ready for fine-tuning.")
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
        print("\n💡 Common fixes:")
        print("- Install missing Python packages: pip install -r requirements.txt")
        print("- Enable Docker Model Runner in Docker Desktop settings")
        print("- Start Ollama service: ollama serve")
        print("- Create missing directories: mkdir -p data models notebooks scripts configs logs tests docker")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

Run the complete test:
```bash
python test_complete_setup.py
```

## Troubleshooting Common Issues

### Docker Issues

**Issue: Docker Model Runner not available**
```bash
# Solution: Enable in Docker Desktop
# Settings → Beta Features → Enable "Docker Model Runner"
```

**Issue: Permission denied (Linux)**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### GPU Issues

**Issue: CUDA out of memory**
```python
# Solution: Use smaller batch sizes and 4-bit quantization
load_in_4bit = True
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
```

**Issue: GPU not detected**
```bash
# Check CUDA installation
nvidia-smi
# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon Mac Issues

**Issue: MPS not available on Apple Silicon**
```bash
# Ensure you have the latest PyTorch
pip install --upgrade torch torchvision torchaudio

# Check macOS version (MPS requires macOS 12.3+)
sw_vers

# Verify MPS in Python
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**Issue: Unsloth installation fails on Apple Silicon**
```bash
# Install Xcode command line tools
xcode-select --install

# Install without extras first
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

# If compilation issues persist, try:
export PYTORCH_ENABLE_MPS_FALLBACK=1
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
```

**Issue: Training crashes with MPS backend**
```python
# Solution: Add MPS fallback for unsupported operations
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Alternative: Force CPU for problematic operations
device = "cpu"  # Instead of "mps" if issues persist
```

### Network Issues

**Issue: Ollama service not starting**
```bash
# Check port availability
sudo lsof -i :11434
# Kill conflicting processes if needed
sudo kill -9 <PID>
# Restart Ollama
ollama serve
```

## 📁 Reference Code Repository

All code examples from this blog series are available in the GitHub repository:

**🔗 [fine-tuning-small-llms](https://github.com/saptak/fine-tuning-small-llms)**

```bash
# Clone the repository to follow along
git clone https://github.com/saptak/fine-tuning-small-llms.git
cd fine-tuning-small-llms

# Run the Part 1 setup script
./part1-setup/scripts/setup_environment.sh
```

The repository includes:
- Complete setup scripts and system checks
- Docker configurations and environment templates
- All code examples organized by blog post parts
- Documentation and usage guides
- Requirements files and dependencies

## What's Next?

Congratulations! You've successfully set up your development environment for fine-tuning small language models with Docker Desktop. In the next part of our series, we'll dive into:

**[Part 2: Data Preparation and Model Selection](/writing/2025/07/25/fine-tuning-small-llms-part2-data-preparation/)**
- Creating high-quality training datasets
- Data formatting and preprocessing
- Choosing the right base model for your use case
- Understanding different model architectures

### Quick Preview

In Part 2, you'll learn how to:
- Format data for different fine-tuning approaches
- Create synthetic datasets for specialized domains
- Select optimal base models (Llama 3.1, Phi-3, Mistral)
- Implement data quality validation

## Resources and References

- [Docker Model Runner Documentation](https://docs.docker.com/ai/model-runner/)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Ollama Model Library](https://ollama.ai/library)
- [Hugging Face Transformers Guide](https://huggingface.co/docs/transformers/)

**Join the Discussion**: Share your setup experience and questions in the comments below!

---

*This is Part 1 of our 6-part series on fine-tuning small LLMs with Docker Desktop. Each part builds upon the previous ones, so make sure to follow along!*
