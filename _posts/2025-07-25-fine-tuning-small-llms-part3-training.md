---
author: Saptak
categories:
- AI
- Machine Learning
- Docker
- LLM
- Fine-tuning
date: 2025-07-25 11:00:00 -0800
description: Part 3 of our comprehensive series. Learn how to fine-tune your selected
  model using Unsloth with LoRA adapters for efficient, memory-optimized training.
featured_image: /assets/images/llm-fine-tuning-part3.jpg
header_image_path: /assets/img/blog/headers/2025-07-25-fine-tuning-small-llms-part3-training.jpg
image_credit: Photo by Volodymyr Dobrovolskyy on Unsplash
layout: post
part: 3
repository: https://github.com/saptak/fine-tuning-small-llms
series: Fine-Tuning Small LLMs on a Desktop
tags:
- llm
- fine-tuning
- unsloth
- training
- lora
- qlora
- huggingface
thumbnail_path: /assets/img/blog/thumbnails/2025-07-25-fine-tuning-small-llms-part3-training.jpg
title: 'Fine-Tuning Small LLMs on a Desktop - Part 3: Fine-Tuning with Unsloth'
toc: true
---

> 📚 **Reference Code Available**: All training scripts and configurations are available in the [GitHub repository](https://github.com/saptak/fine-tuning-small-llms). See `part3-training/` for complete training workflows!

# Fine-Tuning Small LLMs on a Desktop - Part 3: Fine-Tuning with Unsloth

Welcome to the most exciting part of our series! In [Part 1](/writing/2025/07/25/fine-tuning-small-llms-part1-setup-environment), we set up our environment, and in [Part 2](/writing/2025/07/25/fine-tuning-small-llms-part2-data-preparation), we prepared our high-quality dataset. Now it's time to fine-tune our model using Unsloth's revolutionary approach to efficient training.

## Series Navigation

1. [Part 1: Setup and Environment](/writing/2025/07/25/fine-tuning-small-llms-part1-setup-environment)
2. [Part 2: Data Preparation and Model Selection](/writing/2025/07/25/fine-tuning-small-llms-part2-data-preparation)
3. **Part 3: Fine-Tuning with Unsloth** (This post)
4. [Part 4: Evaluation and Testing](/writing/2025/07/25/fine-tuning-small-llms-part4-evaluation)
5. [Part 5: Deployment with Ollama and Docker](/writing/2025/07/25/fine-tuning-small-llms-part5-deployment)
6. [Part 6: Production, Monitoring, and Scaling](/writing/2025/07/25/fine-tuning-small-llms-part6-production)

## Why Unsloth is a Game-Changer

Unsloth revolutionizes LLM fine-tuning by providing:

- **80% Less Memory Usage**: Fine-tune 8B models on consumer GPUs
- **2x Faster Training**: Optimized kernels and efficient attention mechanisms
- **No Accuracy Loss**: Maintains full model quality
- **Simple Interface**: Easy-to-use API that works out of the box

### The Magic Behind Unsloth

Unsloth achieves these improvements through:

1. **Custom Triton Kernels**: Hand-optimized GPU kernels for attention operations
2. **Smart Memory Management**: Efficient gradient checkpointing and optimizer states
3. **LoRA Integration**: Seamless Low-Rank Adaptation support
4. **Quantization**: Native 4-bit and 8-bit quantization support

## Understanding LoRA and QLoRA

Before we start training, let's understand the techniques that make efficient fine-tuning possible:

### LoRA (Low-Rank Adaptation)

Instead of updating all parameters, LoRA adds small "adapter" matrices:

```
Original: W = W₀ (frozen)
LoRA: W = W₀ + B×A (where B and A are trainable, low-rank matrices)
```

**Benefits:**
- Only 1-10% of parameters are trainable
- Dramatically reduces memory requirements
- Prevents catastrophic forgetting
- Easy to merge or swap adapters

### QLoRA (Quantized LoRA)

QLoRA combines LoRA with quantization:
- Base model stored in 4-bit precision
- LoRA adapters in full precision
- Enables fine-tuning 65B models on single GPUs

## Setting Up the Training Environment

Let's start by creating our training notebook:

```python
# fine_tuning_notebook.py
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk, Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments
import wandb
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

print("🚀 Fine-Tuning Setup")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Clear GPU cache
    torch.cuda.empty_cache()
else:
    print("Using CPU - training will be slower")
```

## Model Loading and Configuration

```python
# Model configuration
MAX_SEQ_LENGTH = 2048  # Supports RoPE Scaling
DTYPE = None  # Auto-detection
LOAD_IN_4BIT = True  # Use 4bit quantization for memory efficiency

# Model selection based on your use case
MODEL_CONFIGS = {
    "llama-3.1-8b": "unsloth/llama-3.1-8b-instruct-bnb-4bit",
    "phi-3-mini": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "qwen2-7b": "unsloth/Qwen2-7B-Instruct-bnb-4bit"
}

# Choose your model
SELECTED_MODEL = MODEL_CONFIGS["llama-3.1-8b"]

print(f"📥 Loading model: {SELECTED_MODEL}")

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SELECTED_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    # token="hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

print("✅ Model loaded successfully!")

# Display model information
print(f"\n📊 Model Information:")
print(f"Model name: {model.config.name_or_path if hasattr(model.config, 'name_or_path') else 'Unknown'}")
print(f"Vocab size: {model.config.vocab_size:,}")
print(f"Hidden size: {model.config.hidden_size:,}")
print(f"Number of layers: {model.config.num_hidden_layers}")
print(f"Number of attention heads: {model.config.num_attention_heads}")
```

## Configuring LoRA Adapters

```python
# LoRA Configuration
print("\n🔧 Configuring LoRA Adapters")

# Get PEFT model with LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank - higher = more parameters but better adaptation
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # MLP layers
    ],
    lora_alpha=16,  # LoRA scaling parameter
    lora_dropout=0,  # Dropout for LoRA (0 is optimized for Unsloth)
    bias="none",    # Bias type
    use_gradient_checkpointing="unsloth",  # Memory optimization
    random_state=3407,  # For reproducibility
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None,  # LoftQ configuration
)

# Display trainable parameters
trainable_params = model.get_nb_trainable_parameters()
total_params = sum(p.numel() for p in model.parameters())

print(f"📊 Model Parameters:")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable percentage: {trainable_params / total_params * 100:.2f}%")
```

## Loading and Preparing Training Data

```python
# Load preprocessed datasets
print("\n📂 Loading Training Data")

try:
    # Load from preprocessed datasets
    train_dataset = load_from_disk("./data/processed/train_dataset")
    val_dataset = load_from_disk("./data/processed/val_dataset")

    print(f"✅ Training examples: {len(train_dataset):,}")
    print(f"✅ Validation examples: {len(val_dataset):,}")

except Exception as e:
    print(f"❌ Error loading preprocessed data: {e}")
    print("Creating datasets from JSON files...")

    # Fallback to JSON loading
    with open("./data/processed/train_data.json", 'r') as f:
        train_data = json.load(f)

    with open("./data/processed/val_data.json", 'r') as f:
        val_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    print(f"✅ Training examples: {len(train_dataset):,}")
    print(f"✅ Validation examples: {len(val_dataset):,}")

# Show sample training example
print(f"\n📝 Sample Training Example:")
sample_text = train_dataset[0]['text']
print(f"Length: {len(sample_text)} characters")
print(f"Preview: {sample_text[:200]}...")
```

## Training Configuration

```python
# Advanced Training Configuration
print("\n⚙️ Configuring Training Parameters")

# Determine optimal batch size based on available memory
def get_optimal_batch_size():
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 16:
            return 4  # High memory
        elif gpu_memory_gb >= 8:
            return 2  # Medium memory
        else:
            return 1  # Low memory
    return 1  # CPU fallback

BATCH_SIZE = get_optimal_batch_size()
GRADIENT_ACCUMULATION_STEPS = max(1, 8 // BATCH_SIZE)  # Effective batch size of 8

# Training configuration
training_config = {
    "output_dir": "./models/sql-expert-v1",
    "num_train_epochs": 3,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "optim": "adamw_8bit",  # 8-bit optimizer for memory efficiency
    "warmup_steps": 50,
    "max_steps": 500,  # Increase for better results
    "learning_rate": 2e-4,
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),
    "logging_steps": 10,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "push_to_hub": False,
    "report_to": "wandb" if os.getenv("WANDB_API_KEY") else "none",
    "run_name": f"sql-expert-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "dataloader_pin_memory": torch.cuda.is_available(),
    "dataloader_num_workers": 2 if torch.cuda.is_available() else 0,
    "remove_unused_columns": False,
    "max_grad_norm": 1.0,  # Gradient clipping
}

print(f"📊 Training Configuration:")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Learning rate: {training_config['learning_rate']}")
print(f"Max steps: {training_config['max_steps']}")
print(f"Precision: {'bf16' if training_config['bf16'] else 'fp16'}")

# Create training arguments
training_args = TrainingArguments(**training_config)
```

## Setting Up Experiment Tracking

```python
# Weights & Biases Integration
if os.getenv("WANDB_API_KEY"):
    print("\n📊 Initializing Weights & Biases")

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "llm-fine-tuning"),
        entity=os.getenv("WANDB_ENTITY"),
        name=training_config["run_name"],
        config={
            "model_name": SELECTED_MODEL,
            "max_seq_length": MAX_SEQ_LENGTH,
            "lora_r": 16,
            "lora_alpha": 16,
            "dataset_size": len(train_dataset),
            **training_config
        },
        tags=["fine-tuning", "sql", "unsloth", "lora"]
    )

    print("✅ W&B initialized")
else:
    print("⚠️ W&B not configured - set WANDB_API_KEY to enable tracking")
```

## Creating the Trainer

```python
# Initialize the SFT Trainer
print("\n🏋️ Initializing Trainer")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # Disable packing for better control
    args=training_args,
)

print("✅ Trainer initialized")

# Display training statistics
print(f"\n📈 Training Statistics:")
print(f"Training steps: {len(trainer.get_train_dataloader()) * training_args.num_train_epochs}")
print(f"Evaluation steps: {len(trainer.get_eval_dataloader())}")
print(f"Estimated training time: {(training_args.max_steps * 0.5 / 60):.1f} minutes")
```

## Training Process with Real-Time Monitoring

### Monitoring with TensorBoard

TensorBoard is a powerful tool for visualizing and monitoring your training progress. You can use it to track metrics like loss, learning rate, and accuracy in real-time.

To start TensorBoard, run the following command in a new terminal:

```bash
tensorboard --logdir=./models/sql-expert-v1/runs
```

This will start a web server on port 6006. You can then open your browser to `http://localhost:6006` to view the TensorBoard dashboard.

```python
# Training with comprehensive monitoring
class TrainingMonitor:
    def __init__(self):
        self.start_time = None
        self.step_times = []
        self.losses = []

    def on_train_begin(self):
        self.start_time = datetime.now()
        print(f"🚀 Training started at {self.start_time.strftime('%H:%M:%S')}")

    def on_step_end(self, step, logs):
        current_time = datetime.now()
        if self.start_time:
            elapsed = (current_time - self.start_time).total_seconds()
            self.step_times.append(elapsed)

            if 'train_loss' in logs:
                self.losses.append(logs['train_loss'])

            # Print progress every 10 steps
            if step % 10 == 0:
                avg_step_time = sum(self.step_times[-10:]) / min(10, len(self.step_times))
                eta_seconds = avg_step_time * (trainer.state.max_steps - step)
                eta_minutes = eta_seconds / 60

                print(f"Step {step:4d}/{trainer.state.max_steps} | "
                      f"Loss: {logs.get('train_loss', 0):.4f} | "
                      f"LR: {logs.get('learning_rate', 0):.2e} | "
                      f"ETA: {eta_minutes:.1f}m")

                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    memory_cached = torch.cuda.memory_reserved() / 1e9
                    print(f"         GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")

# Custom callback for monitoring
from transformers.trainer_callback import TrainerCallback

class CustomTrainingCallback(TrainerCallback):
    def __init__(self):
        self.monitor = TrainingMonitor()

    def on_train_begin(self, args, state, control, **kwargs):
        self.monitor.on_train_begin()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.monitor.on_step_end(state.global_step, logs)

# Add callback to trainer
trainer.add_callback(CustomTrainingCallback())

print("\n🏁 Starting Training...")
print("=" * 60)

# Start training
try:
    trainer_stats = trainer.train()

    print("\n🎉 Training Completed!")
    print("=" * 60)

    # Training summary
    final_loss = trainer_stats.metrics.get('train_loss', 'N/A')
    training_time = trainer_stats.metrics.get('train_runtime', 0)

    print(f"📊 Training Summary:")
    print(f"Final loss: {final_loss}")
    print(f"Training time: {training_time / 60:.1f} minutes")
    print(f"Steps completed: {trainer_stats.global_step}")
    print(f"Samples processed: {trainer_stats.global_step * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {max_memory:.1f} GB")

except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
    print("Saving current progress...")
    trainer.save_model("./models/sql-expert-checkpoint")

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    print("Saving checkpoint for debugging...")
    trainer.save_model("./models/sql-expert-error-checkpoint")
    raise
```

## Saving and Managing Models

```python
# Comprehensive model saving
print("\n💾 Saving Models")

# Create output directories
output_dirs = {
    "lora": "./models/sql-expert-lora",
    "merged": "./models/sql-expert-merged",
    "quantized": "./models/sql-expert-quantized"
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Save LoRA adapters only
print("Saving LoRA adapters...")
model.save_pretrained(output_dirs["lora"])
tokenizer.save_pretrained(output_dirs["lora"])

# Save merged model (base model + LoRA adapters)
print("Saving merged model...")
model.save_pretrained_merged(
    output_dirs["merged"],
    tokenizer,
    save_method="merged_16bit"
)

# Save quantized version for efficient inference
print("Saving quantized model...")
model.save_pretrained_merged(
    output_dirs["quantized"],
    tokenizer,
    save_method="merged_4bit"
)

print("✅ All model variants saved!")

# Create model card
model_card_content = f"""---
author: Saptak
categories:
- AI
- Machine Learning
- Docker
- LLM
- Fine-tuning
date: 2025-07-25 11:00:00 -0800
description: Part 3 of our comprehensive series. Learn how to fine-tune your selected
  model using Unsloth with LoRA adapters for efficient, memory-optimized training.
featured_image: /assets/images/llm-fine-tuning-part3.jpg
header_image_path: /assets/img/blog/headers/2025-07-25-fine-tuning-small-llms-part3-training.jpg
image_credit: Photo by Volodymyr Dobrovolskyy on Unsplash
layout: post
part: 3
repository: https://github.com/saptak/fine-tuning-small-llms
series: Fine-Tuning Small LLMs with Docker Desktop
tags:
- llm
- fine-tuning
- unsloth
- training
- lora
- qlora
- huggingface
thumbnail_path: /assets/img/blog/thumbnails/2025-07-25-fine-tuning-small-llms-part3-training.jpg
title: 'Fine-Tuning Small LLMs with Docker Desktop - Part 3: Fine-Tuning with Unsloth'
toc: true
---

# SQL Expert Model

This model is a fine-tuned version of {SELECTED_MODEL} for SQL query generation.

## Model Details

- **Base Model**: {SELECTED_MODEL}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: Custom SQL dataset with {len(train_dataset)} examples
- **Training Framework**: Unsloth
- **Training Time**: {training_time / 60:.1f} minutes
- **Final Training Loss**: {final_loss}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./models/sql-expert-merged")
model = AutoModelForCausalLM.from_pretrained("./models/sql-expert-merged")

# Example usage
prompt = "Generate SQL to find all customers who registered in the last 30 days"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Configuration

- LoRA Rank: 16
- LoRA Alpha: 16
- Learning Rate: {training_config['learning_rate']}
- Batch Size: {BATCH_SIZE}
- Max Steps: {training_config['max_steps']}
- Precision: {'bf16' if training_config['bf16'] else 'fp16'}

## Performance

The model has been optimized for SQL query generation and shows strong performance on:
- Basic SELECT operations
- Complex JOINs
- Aggregation queries
- Window functions
- Common Table Expressions (CTEs)
"""

with open(f"{output_dirs['merged']}/README.md", "w") as f:
    f.write(model_card_content)

print("📄 Model card created")
```

## Testing the Fine-Tuned Model

```python
# Quick model testing
print("\n🧪 Testing Fine-Tuned Model")

# Enable inference mode for faster generation
FastLanguageModel.for_inference(model)

def test_model(prompt, max_new_tokens=256):
    """Test the fine-tuned model"""
    inputs = tokenizer(
        [prompt],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part
    generated_text = response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
    return generated_text.strip()

# Test cases
test_cases = [
    {
        "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Generate SQL to find all users who registered in the last 7 days\n\nTable Schema: users (id, username, email, registration_date)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "description": "Recent user registrations"
    },
    {
        "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Write a query to calculate the average order value per customer\n\nTable Schema: orders (id, customer_id, total_amount, order_date)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "description": "Average order value calculation"
    },
    {
        "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Create a query to find the top 5 customers by total spending\n\nTable Schema: customers (id, name, email), orders (id, customer_id, amount)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "description": "Top customers by spending"
    }
]

print("\n🎯 Test Results:")
print("=" * 60)

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test_case['description']}")
    print("-" * 40)

    try:
        start_time = datetime.now()
        response = test_model(test_case['prompt'])
        end_time = datetime.now()

        generation_time = (end_time - start_time).total_seconds()

        print(f"Generated SQL: {response}")
        print(f"Generation time: {generation_time:.2f}s")

    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ Model testing completed!")
```

## Memory Usage Optimization

```python
# Memory optimization utilities
class MemoryTracker:
    @staticmethod
    def get_gpu_memory_info():
        if not torch.cuda.is_available():
            return "GPU not available"

        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        return f"Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB, Peak: {max_allocated:.1f}GB"

    @staticmethod
    def optimize_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import gc
        gc.collect()

    @staticmethod
    def get_model_memory_footprint(model):
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size = (param_size + buffer_size) / 1e9
        return f"Model size: {model_size:.2f}GB"

# Use memory tracker
print(f"\n💾 Memory Usage: {MemoryTracker.get_gpu_memory_info()}")
print(f"📊 {MemoryTracker.get_model_memory_footprint(model)}")

# Optimize memory after training
MemoryTracker.optimize_memory()
print("🧹 Memory optimized")
```

## Training Troubleshooting

```python
# Common training issues and solutions
class TrainingTroubleshooter:
    @staticmethod
    def diagnose_memory_issues():
        if not torch.cuda.is_available():
            return "Using CPU - no GPU memory issues"

        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9

        issues = []
        solutions = []

        if allocated / total_memory > 0.9:
            issues.append("GPU memory usage > 90%")
            solutions.extend([
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Use 4-bit quantization",
                "Reduce max_seq_length"
            ])

        return {
            "total_memory_gb": total_memory,
            "allocated_gb": allocated,
            "usage_percent": (allocated / total_memory) * 100,
            "issues": issues,
            "solutions": solutions
        }

    @staticmethod
    def check_training_stability(losses):
        if len(losses) < 10:
            return "Not enough data points"

        recent_losses = losses[-10:]
        trend = "stable"

        if recent_losses[-1] > recent_losses[0] * 1.1:
            trend = "increasing"
        elif recent_losses[-1] < recent_losses[0] * 0.9:
            trend = "decreasing"

        return {
            "trend": trend,
            "recent_avg": sum(recent_losses) / len(recent_losses),
            "latest_loss": recent_losses[-1],
            "recommendation": {
                "increasing": "Consider reducing learning rate",
                "decreasing": "Training progressing well",
                "stable": "Loss has plateaued - consider early stopping"
            }.get(trend, "Unknown trend")
        }

# Diagnostic report
print("\n🔍 Training Diagnostics:")
memory_diag = TrainingTroubleshootir.diagnose_memory_issues()
print(f"Memory usage: {memory_diag.get('usage_percent', 0):.1f}%")

if memory_diag.get('issues'):
    print("⚠️ Issues found:", ', '.join(memory_diag['issues']))
    print("💡 Solutions:", ', '.join(memory_diag['solutions']))
```

## 📁 Reference Code Repository

All training code and configurations are available in the GitHub repository:

**🔗 [fine-tuning-small-llms/part3-training](https://github.com/saptak/fine-tuning-small-llms/tree/main/part3-training)**

```bash
# Clone the repository and navigate to training
git clone https://github.com/saptak/fine-tuning-small-llms.git
cd fine-tuning-small-llms

# Install dependencies
pip install -r requirements.txt

# Start training (example - full implementation in repository)
python part3-training/src/fine_tune_model.py --config part3-training/configs/sql_expert.yaml
```

The Part 3 directory includes:
- Complete Unsloth training scripts
- LoRA configuration templates
- Training monitoring and callbacks
- Jupyter notebooks for interactive training
- Model saving and checkpoint management
- W&B integration examples

## What's Next?

Congratulations! You've successfully fine-tuned your first small language model using Unsloth. Your model is now specialized for your specific use case and ready for evaluation.

**[Part 4: Evaluation and Testing](/writing/2025/07/25/fine-tuning-small-llms-part4-evaluation)**

In Part 4, you'll learn:
- Comprehensive evaluation frameworks
- Automated testing pipelines
- Performance benchmarking
- Quality assurance techniques
- A/B testing methodologies

### Key Achievements from Part 3

✅ **Efficient Training**: Used Unsloth for 80% memory reduction and 2x speed improvement
✅ **LoRA Integration**: Implemented parameter-efficient fine-tuning
✅ **Memory Optimization**: Handled large models on consumer hardware
✅ **Experiment Tracking**: Monitored training with Weights & Biases
✅ **Model Management**: Saved multiple model variants for different use cases

## Troubleshooting Quick Reference

| Issue | Symptoms | Solution |
|-------|----------|----------|
| CUDA OOM | "out of memory" error | Reduce batch size, enable 4-bit quantization |
| Slow training | Very long step times | Check GPU utilization, reduce sequence length |
| Poor convergence | Loss not decreasing | Adjust learning rate, check data quality |
| Unstable training | Loss oscillating | Reduce learning rate, add gradient clipping |
| Disk space | "No space left" error | Clean up checkpoints, use smaller models |

## Resources and References

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Weights & Biases Guides](https://docs.wandb.ai/)

---

*Continue to [Part 4: Evaluation and Testing](/writing/2025/07/25/fine-tuning-small-llms-part4-evaluation) to validate your model's performance!*
