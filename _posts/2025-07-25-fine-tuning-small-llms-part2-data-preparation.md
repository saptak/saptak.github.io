---
author: Saptak
categories:
- AI
- Machine Learning
- Docker
- LLM
- Fine-tuning
date: 2025-07-25 10:00:00 -0800
description: Part 2 of our comprehensive series. Learn how to prepare high-quality
  training datasets, select the optimal base model, and format data for efficient
  fine-tuning with Unsloth.
featured_image: /assets/images/llm-fine-tuning-part2.jpg
header_image_path: /assets/img/blog/headers/2025-07-25-fine-tuning-small-llms-part2-data-preparation.jpg
image_credit: Photo by Safar Safarov on Unsplash
layout: post
part: 2
repository: https://github.com/saptak/fine-tuning-small-llms
series: Fine-Tuning Small LLMs with Docker Desktop
tags:
- llm
- fine-tuning
- data-preparation
- model-selection
- datasets
- unsloth
thumbnail_path: /assets/img/blog/thumbnails/2025-07-25-fine-tuning-small-llms-part2-data-preparation.jpg
title: 'Fine-Tuning Small LLMs with Docker Desktop - Part 2: Data Preparation and
  Model Selection'
toc: true
---

> 📚 **Reference Code Available**: All code examples from this blog series are available in the [GitHub repository](https://github.com/saptak/fine-tuning-small-llms). See `part2-data-preparation/` for the complete data preparation toolkit!

# Fine-Tuning Small LLMs with Docker Desktop - Part 2: Data Preparation and Model Selection

Welcome back! In [Part 1](/2025/07/25/fine-tuning-small-llms-part1-setup-environment/), we set up our development environment with Docker Desktop, CUDA support, and all necessary tools. Now we dive into the critical foundation of any successful fine-tuning project: **data preparation and model selection**.

This is where the magic begins—the quality of your training data will ultimately determine the success of your fine-tuned model. We'll explore advanced techniques for creating, validating, and optimizing datasets that produce exceptional results.

## Series Navigation

1. [Part 1: Setup and Environment](/2025/07/25/fine-tuning-small-llms-part1-setup-environment/)
2. **Part 2: Data Preparation and Model Selection** (This post)
3. [Part 3: Fine-Tuning with Unsloth](/2025/07/25/fine-tuning-small-llms-part3-training/)
4. [Part 4: Evaluation and Testing](/2025/07/25/fine-tuning-small-llms-part4-evaluation/)
5. [Part 5: Deployment with Ollama and Docker](/2025/07/25/fine-tuning-small-llms-part5-deployment/)
6. [Part 6: Production, Monitoring, and Scaling](/2025/07/25/fine-tuning-small-llms-part6-production/)

## The Data Quality Imperative

Before we dive into code, let's understand why data quality is paramount:

> **"Garbage in, garbage out"** - This age-old principle is especially true for LLM fine-tuning. A model trained on 500 high-quality, diverse examples will consistently outperform one trained on 5,000 mediocre, repetitive samples.

### Key Principles for High-Quality Training Data

1. **Diversity Over Volume**: Cover edge cases and variations
2. **Consistency**: Maintain uniform formatting and style
3. **Accuracy**: Ensure all examples are factually correct
4. **Relevance**: Every example should serve your specific use case
5. **Balance**: Avoid overrepresenting any single pattern

## Understanding Data Formats for Fine-Tuning

Different training approaches require different data formats. Let's explore the most effective ones:

### Alpaca Format (Instruction-Following)
```json
{
  "instruction": "Generate SQL to find users registered in the last 30 days",
  "input": "Table Schema: users (id, name, email, registration_date)",
  "output": "SELECT * FROM users WHERE registration_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);"
}
```

### Chat Format (Conversational)
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert SQL developer."},
    {"role": "user", "content": "Write a query to find recent users"},
    {"role": "assistant", "content": "SELECT * FROM users WHERE registration_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);"}
  ]
}
```

### Completion Format (Text Generation)
```json
{
  "prompt": "### SQL Query Request:\nFind users registered recently\n\n### SQL:\n",
  "completion": "SELECT * FROM users WHERE registration_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);"
}
```

## Model Selection Strategy

Choosing the right base model is crucial for success. Here's our decision framework:

### Model Evaluation Matrix

| Model | Size | Memory (GB) | Speed | Use Case | Best For |
|-------|------|-------------|-------|----------|----------|
| **Phi-3-Mini** | 3.8B | 4-6 | Fast | General, Coding | Resource-constrained environments |
| **Llama-3.1-8B** | 8B | 8-12 | Medium | General, Reasoning | Balanced performance/resources |
| **Mistral-7B** | 7B | 6-10 | Medium | Code, Technical | Programming tasks |
| **Qwen2-7B** | 7B | 6-10 | Medium | Multilingual | International applications |
| **CodeLlama-7B** | 7B | 8-12 | Medium | Programming | Pure code generation |

### Smart Model Selection Function

```python
# model_selection.py
import torch
import psutil
from typing import Dict, List, Tuple, Optional

def analyze_system_resources() -> Dict[str, float]:
    """Analyze available system resources"""
    
    # Get CPU information
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    memory = psutil.virtual_memory()
    
    # Get GPU information if available
    gpu_memory_gb = 0
    gpu_count = 0
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return {
        "cpu_cores": cpu_count,
        "cpu_frequency_ghz": cpu_freq.current / 1000 if cpu_freq else 0,
        "ram_gb": memory.total / 1e9,
        "available_ram_gb": memory.available / 1e9,
        "gpu_count": gpu_count,
        "gpu_memory_gb": gpu_memory_gb
    }

def estimate_model_requirements(model_size_billion: float, 
                              quantization: str = "4bit") -> Dict[str, float]:
    """Estimate resource requirements for a model"""
    
    # Base memory requirements (rough estimates)
    base_memory_gb = {
        "fp16": model_size_billion * 2,
        "8bit": model_size_billion * 1.2,
        "4bit": model_size_billion * 0.75,
        "fp32": model_size_billion * 4
    }
    
    model_memory = base_memory_gb.get(quantization, base_memory_gb["4bit"])
    
    # Add overhead for training (LoRA adapters, optimizer states, etc.)
    training_overhead = model_memory * 0.3
    
    return {
        "inference_memory_gb": model_memory,
        "training_memory_gb": model_memory + training_overhead,
        "minimum_ram_gb": model_memory * 1.5,  # For CPU fallback
        "recommended_vram_gb": model_memory + training_overhead
    }

def select_optimal_model(use_case: str, 
                        memory_constraint_gb: float,
                        performance_priority: str = "balanced") -> str:
    """
    Select optimal model based on requirements
    """
    
    recommendations = {
        "general": {
            "high_memory": "unsloth/llama-3.1-8b-instruct-bnb-4bit",
            "low_memory": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
        },
        "coding": {
            "high_memory": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "low_memory": "unsloth/CodeLlama-7b-instruct-bnb-4bit"
        },
        "multilingual": {
            "high_memory": "unsloth/Qwen2-7B-Instruct-bnb-4bit",
            "low_memory": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
        },
        "conversational": {
            "high_memory": "unsloth/llama-3.1-8b-instruct-bnb-4bit",
            "low_memory": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
        }
    }
    
    memory_tier = "high_memory" if memory_constraint_gb >= 8 else "low_memory"
    
    if use_case in recommendations:
        return recommendations[use_case][memory_tier]
    else:
        return recommendations["general"][memory_tier]

# Example usage
recommended_model = select_optimal_model(
    use_case="coding", 
    memory_constraint_gb=16, 
    performance_priority="balanced"
)
print(f"Recommended model: {recommended_model}")
```

## Creating High-Quality Training Datasets

Let's create a practical example with SQL generation - a common and valuable use case:

### Step 1: Define Your Dataset Structure

```python
# dataset_creation.py
import pandas as pd
import json
from typing import List, Dict
from pathlib import Path

class SQLDatasetCreator:
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples = []
    
    def add_example(self, instruction: str, table_schema: str, sql_query: str, 
                   explanation: str = "", difficulty: str = "medium"):
        """Add a training example to the dataset"""
        
        example = {
            "instruction": instruction,
            "input": f"Table Schema: {table_schema}",
            "output": sql_query,
            "explanation": explanation,
            "difficulty": difficulty,
            "id": len(self.examples)
        }
        
        self.examples.append(example)
        return example
    
    def create_basic_examples(self):
        """Create fundamental SQL examples"""
        
        # Basic SELECT operations
        self.add_example(
            instruction="Select all columns from the users table",
            table_schema="users (id, name, email, created_at)",
            sql_query="SELECT * FROM users;",
            explanation="Basic SELECT statement to retrieve all columns and rows",
            difficulty="easy"
        )
        
        self.add_example(
            instruction="Find all users who registered in the last 30 days",
            table_schema="users (id, name, email, created_at)",
            sql_query="SELECT * FROM users WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);",
            explanation="Uses DATE_SUB function to filter recent registrations",
            difficulty="medium"
        )
        
        # Aggregation queries
        self.add_example(
            instruction="Count the total number of orders per customer",
            table_schema="orders (id, customer_id, amount, order_date)",
            sql_query="SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id;",
            explanation="Groups by customer and counts orders using COUNT(*)",
            difficulty="medium"
        )
        
        self.add_example(
            instruction="Find the average order amount per month",
            table_schema="orders (id, customer_id, amount, order_date)",
            sql_query="SELECT DATE_FORMAT(order_date, '%Y-%m') as month, AVG(amount) as avg_amount FROM orders GROUP BY DATE_FORMAT(order_date, '%Y-%m');",
            explanation="Uses DATE_FORMAT to group by month and AVG for average calculation",
            difficulty="medium"
        )
        
        # JOIN operations
        self.add_example(
            instruction="Show customer names with their total order amounts",
            table_schema="customers (id, name, email), orders (id, customer_id, amount, order_date)",
            sql_query="SELECT c.name, SUM(o.amount) as total_amount FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name;",
            explanation="INNER JOIN between customers and orders with SUM aggregation",
            difficulty="hard"
        )
        
        # Complex analytical queries
        self.add_example(
            instruction="Find customers who have spent more than the average customer spending",
            table_schema="customers (id, name, email), orders (id, customer_id, amount)",
            sql_query="""SELECT c.name, SUM(o.amount) as total_spent 
FROM customers c 
JOIN orders o ON c.id = o.customer_id 
GROUP BY c.id, c.name 
HAVING SUM(o.amount) > (
    SELECT AVG(customer_total) 
    FROM (
        SELECT SUM(amount) as customer_total 
        FROM orders 
        GROUP BY customer_id
    ) as customer_totals
);""",
            explanation="Complex query with subquery to find above-average spenders",
            difficulty="expert"
        )
    
    def create_advanced_examples(self):
        """Create advanced SQL examples"""
        
        # Window functions
        self.add_example(
            instruction="Rank customers by their total spending within each region",
            table_schema="customers (id, name, region), orders (id, customer_id, amount)",
            sql_query="""SELECT 
    c.name, 
    c.region,
    SUM(o.amount) as total_spent,
    RANK() OVER (PARTITION BY c.region ORDER BY SUM(o.amount) DESC) as spending_rank
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.region;""",
            explanation="Uses window function RANK() with PARTITION BY for regional rankings",
            difficulty="expert"
        )
        
        # Common Table Expressions (CTEs)
        self.add_example(
            instruction="Find the second highest order amount for each customer",
            table_schema="orders (id, customer_id, amount, order_date)",
            sql_query="""WITH ranked_orders AS (
    SELECT 
        customer_id,
        amount,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) as rn
    FROM orders
)
SELECT customer_id, amount as second_highest_amount
FROM ranked_orders 
WHERE rn = 2;""",
            explanation="Uses CTE with ROW_NUMBER() to find second highest values",
            difficulty="expert"
        )
    
    def format_for_training(self, format_type: str = "alpaca") -> List[Dict]:
        """Format examples for different training approaches"""
        
        formatted_examples = []
        
        for example in self.examples:
            if format_type == "alpaca":
                formatted = {
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "output": example["output"]
                }
            
            elif format_type == "chat":
                formatted = {
                    "messages": [
                        {"role": "system", "content": "You are an expert SQL developer who generates accurate and efficient SQL queries."},
                        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
                        {"role": "assistant", "content": example["output"]}
                    ]
                }
            
            elif format_type == "completion":
                formatted = {
                    "prompt": f"### SQL Request:\n{example['instruction']}\n\n{example['input']}\n\n### SQL Query:\n",
                    "completion": example["output"]
                }
            
            formatted_examples.append(formatted)
        
        return formatted_examples
    
    def save_dataset(self, filename: str = "sql_training_data", format_type: str = "alpaca"):
        """Save dataset in specified format"""
        
        formatted_data = self.format_for_training(format_type)
        
        # Save as JSON
        json_path = self.output_dir / f"{filename}_{format_type}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV (for Alpaca format)
        if format_type == "alpaca":
            df = pd.DataFrame(formatted_data)
            csv_path = self.output_dir / f"{filename}_alpaca.csv"
            df.to_csv(csv_path, index=False)
        
        print(f"Dataset saved: {json_path}")
        print(f"Total examples: {len(formatted_data)}")
        
        return json_path

# Create comprehensive SQL dataset
def create_sql_dataset():
    creator = SQLDatasetCreator(output_dir="./data/datasets")
    
    # Add examples
    creator.create_basic_examples()
    creator.create_advanced_examples()
    
    # Add domain-specific examples
    creator.add_example(
        instruction="Create a query to find the top 10 products by sales volume",
        table_schema="products (id, name, category), order_items (id, product_id, quantity, order_id)",
        sql_query="SELECT p.name, SUM(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id, p.name ORDER BY total_sold DESC LIMIT 10;",
        difficulty="medium"
    )
    
    creator.add_example(
        instruction="Calculate monthly revenue growth rate",
        table_schema="orders (id, amount, order_date)",
        sql_query="""WITH monthly_revenue AS (
    SELECT 
        DATE_FORMAT(order_date, '%Y-%m') as month,
        SUM(amount) as revenue
    FROM orders 
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
),
revenue_with_lag AS (
    SELECT 
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as prev_revenue
    FROM monthly_revenue
)
SELECT 
    month,
    revenue,
    ROUND(((revenue - prev_revenue) / prev_revenue) * 100, 2) as growth_rate_percent
FROM revenue_with_lag
WHERE prev_revenue IS NOT NULL;""",
        difficulty="expert"
    )
    
    # Save in multiple formats
    creator.save_dataset("sql_dataset", "alpaca")
    creator.save_dataset("sql_dataset", "chat")
    
    return creator

# Usage
if __name__ == "__main__":
    dataset_creator = create_sql_dataset()
    print(f"Created dataset with {len(dataset_creator.examples)} examples")
```

### Step 2: Data Quality Validation

```python
# data_validation.py
import pandas as pd
import json
import re
from typing import List, Dict, Tuple
import sqlparse
from sqlparse import sql, tokens

class DataQualityValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def validate_sql_syntax(self, sql_query: str) -> Tuple[bool, str]:
        """Validate SQL syntax using sqlparse"""
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Empty or invalid SQL"
            
            # Check for basic SQL structure
            formatted = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
            return True, "Valid SQL syntax"
            
        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"
    
    def validate_dataset(self, dataset_path: str, format_type: str = "alpaca") -> Dict:
        """Comprehensive dataset validation"""
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        validation_results = {
            "total_examples": len(data),
            "valid_examples": 0,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        for i, example in enumerate(data):
            example_errors = []
            example_warnings = []
            
            if format_type == "alpaca":
                # Check required fields
                required_fields = ["instruction", "input", "output"]
                for field in required_fields:
                    if field not in example or not example[field].strip():
                        example_errors.append(f"Missing or empty {field}")
                
                # Validate SQL in output
                if "output" in example:
                    is_valid, message = self.validate_sql_syntax(example["output"])
                    if not is_valid:
                        example_errors.append(f"Invalid SQL: {message}")
                
                # Check instruction quality
                if "instruction" in example:
                    if len(example["instruction"]) < 10:
                        example_warnings.append("Instruction too short")
                    if not example["instruction"].endswith(('?', '.')):
                        example_warnings.append("Instruction should end with punctuation")
            
            # Record results
            if not example_errors:
                validation_results["valid_examples"] += 1
            else:
                validation_results["errors"].append({
                    "example_index": i,
                    "errors": example_errors
                })
                
            if example_warnings:
                validation_results["warnings"].append({
                    "example_index": i,
                    "warnings": example_warnings
                })
        
        # Calculate statistics
        validation_results["statistics"] = {
            "success_rate": validation_results["valid_examples"] / len(data) * 100,
            "error_rate": len(validation_results["errors"]) / len(data) * 100,
            "warning_rate": len(validation_results["warnings"]) / len(data) * 100
        }
        
        return validation_results
    
    def generate_quality_report(self, validation_results: Dict) -> str:
        """Generate human-readable quality report"""
        
        report = f"""
📊 Dataset Quality Report
========================

Total Examples: {validation_results['total_examples']}
Valid Examples: {validation_results['valid_examples']}
Success Rate: {validation_results['statistics']['success_rate']:.1f}%

❌ Errors: {len(validation_results['errors'])}
⚠️  Warnings: {len(validation_results['warnings'])}

"""
        
        if validation_results['errors']:
            report += "🔍 Error Details:\n"
            for error in validation_results['errors'][:5]:  # Show first 5
                report += f"  Example {error['example_index']}: {', '.join(error['errors'])}\n"
        
        if validation_results['warnings']:
            report += "\n⚠️  Warning Details:\n"
            for warning in validation_results['warnings'][:5]:  # Show first 5
                report += f"  Example {warning['example_index']}: {', '.join(warning['warnings'])}\n"
        
        return report

# Usage example
def validate_sql_dataset():
    validator = DataQualityValidator()
    
    # Validate the dataset
    results = validator.validate_dataset("./data/datasets/sql_dataset_alpaca.json")
    
    # Generate report
    report = validator.generate_quality_report(results)
    print(report)
    
    # Save validation report
    with open("./data/validation_report.txt", "w") as f:
        f.write(report)
    
    return results

if __name__ == "__main__":
    validation_results = validate_sql_dataset()
```

### Step 3: Data Augmentation Techniques

```python
# data_augmentation.py
import random
import json
from typing import List, Dict

class SQLDataAugmenter:
    def __init__(self):
        self.table_variations = {
            'users': ['customers', 'clients', 'members', 'accounts'],
            'orders': ['purchases', 'transactions', 'sales', 'bookings'],
            'products': ['items', 'goods', 'services', 'offerings'],
            'categories': ['types', 'groups', 'classes', 'segments']
        }
        
        self.column_variations = {
            'id': ['id', 'user_id', 'customer_id', 'primary_key'],
            'name': ['name', 'full_name', 'title', 'label'],
            'email': ['email', 'email_address', 'contact_email'],
            'created_at': ['created_at', 'created_date', 'registration_date', 'signup_date'],
            'amount': ['amount', 'price', 'cost', 'value', 'total']
        }
    
    def augment_table_names(self, sql_query: str, schema: str) -> List[Dict]:
        """Create variations by changing table names"""
        variations = []
        
        # Extract original table names from schema
        tables = self.extract_tables_from_schema(schema)
        
        for table in tables:
            if table in self.table_variations:
                for variation in self.table_variations[table]:
                    new_sql = sql_query.replace(table, variation)
                    new_schema = schema.replace(table, variation)
                    variations.append({
                        'sql': new_sql,
                        'schema': new_schema,
                        'variation_type': f'table_name_{variation}'
                    })
        
        return variations
    
    def augment_conditions(self, base_example: Dict) -> List[Dict]:
        """Create variations with different WHERE conditions"""
        variations = []
        
        # Time-based variations
        time_conditions = [
            "WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)",
            "WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)",
            "WHERE created_at >= '2024-01-01'",
            "WHERE YEAR(created_at) = 2024"
        ]
        
        for condition in time_conditions:
            new_example = base_example.copy()
            new_example['instruction'] = new_example['instruction'].replace('30 days', self.extract_time_period(condition))
            new_example['output'] = new_example['output'].replace(
                "WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)",
                condition
            )
            variations.append(new_example)
        
        return variations
    
    def extract_tables_from_schema(self, schema: str) -> List[str]:
        """Extract table names from schema string"""
        import re
        tables = re.findall(r'(\w+)\s*\(', schema)
        return tables
    
    def extract_time_period(self, condition: str) -> str:
        """Extract time period description from SQL condition"""
        if '7 DAY' in condition:
            return '7 days'
        elif '90 DAY' in condition:
            return '90 days'
        elif '2024' in condition:
            return '2024'
        return 'specified period'
    
    def generate_variations(self, original_dataset: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
        """Generate augmented dataset"""
        augmented_data = original_dataset.copy()
        
        for example in original_dataset:
            # Generate table name variations
            if 'output' in example and 'SELECT' in example['output'].upper():
                table_variations = self.augment_table_names(
                    example['output'], 
                    example.get('input', '')
                )
                
                for var in table_variations[:augmentation_factor]:
                    new_example = example.copy()
                    new_example['output'] = var['sql']
                    new_example['input'] = var['schema']
                    new_example['variation_type'] = var['variation_type']
                    augmented_data.append(new_example)
        
        return augmented_data

# Usage
def augment_sql_dataset():
    # Load original dataset
    with open('./data/datasets/sql_dataset_alpaca.json', 'r') as f:
        original_data = json.load(f)
    
    # Create augmenter
    augmenter = SQLDataAugmenter()
    
    # Generate variations
    augmented_data = augmenter.generate_variations(original_data, augmentation_factor=3)
    
    # Save augmented dataset
    with open('./data/datasets/sql_dataset_augmented.json', 'w') as f:
        json.dump(augmented_data, f, indent=2)
    
    print(f"Original dataset: {len(original_data)} examples")
    print(f"Augmented dataset: {len(augmented_data)} examples")
    print(f"Augmentation ratio: {len(augmented_data) / len(original_data):.1f}x")

if __name__ == "__main__":
    augment_sql_dataset()
```

## Data Format Conversion

Let's create utilities to convert between different formats:

```python
# format_converter.py
import json
import pandas as pd
from typing import List, Dict

class DatasetFormatConverter:
    def __init__(self):
        pass
    
    @staticmethod
    def alpaca_to_chat(alpaca_data: List[Dict]) -> List[Dict]:
        """Convert Alpaca format to Chat format"""
        chat_data = []
        
        for example in alpaca_data:
            chat_example = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert SQL developer who generates accurate and efficient SQL queries based on user requirements."
                    },
                    {
                        "role": "user", 
                        "content": f"{example['instruction']}\n\n{example.get('input', '')}"
                    },
                    {
                        "role": "assistant", 
                        "content": example['output']
                    }
                ]
            }
            chat_data.append(chat_example)
        
        return chat_data
    
    @staticmethod
    def alpaca_to_completion(alpaca_data: List[Dict]) -> List[Dict]:
        """Convert Alpaca format to Completion format"""
        completion_data = []
        
        for example in alpaca_data:
            prompt = f"### Instruction:\n{example['instruction']}\n\n"
            if example.get('input', '').strip():
                prompt += f"### Input:\n{example['input']}\n\n"
            prompt += "### Response:\n"
            
            completion_example = {
                "prompt": prompt,
                "completion": example['output']
            }
            completion_data.append(completion_example)
        
        return completion_data
    
    @staticmethod
    def chat_to_alpaca(chat_data: List[Dict]) -> List[Dict]:
        """Convert Chat format to Alpaca format"""
        alpaca_data = []
        
        for example in chat_data:
            messages = example.get('messages', [])
            
            # Find user and assistant messages
            user_msg = next((msg for msg in messages if msg['role'] == 'user'), None)
            assistant_msg = next((msg for msg in messages if msg['role'] == 'assistant'), None)
            
            if user_msg and assistant_msg:
                # Try to split user message into instruction and input
                user_content = user_msg['content']
                lines = user_content.split('\n\n')
                
                if len(lines) >= 2 and lines[1].startswith(('Table', 'Schema', 'Context')):
                    instruction = lines[0]
                    input_text = lines[1]
                else:
                    instruction = user_content
                    input_text = ""
                
                alpaca_example = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": assistant_msg['content']
                }
                alpaca_data.append(alpaca_example)
        
        return alpaca_data
    
    def convert_dataset(self, input_path: str, output_path: str, 
                       from_format: str, to_format: str):
        """Convert dataset from one format to another"""
        
        # Load input data
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Convert data
        if from_format == "alpaca" and to_format == "chat":
            output_data = self.alpaca_to_chat(input_data)
        elif from_format == "alpaca" and to_format == "completion":
            output_data = self.alpaca_to_completion(input_data)
        elif from_format == "chat" and to_format == "alpaca":
            output_data = self.chat_to_alpaca(input_data)
        else:
            raise ValueError(f"Conversion from {from_format} to {to_format} not supported")
        
        # Save output data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Converted {len(input_data)} examples from {from_format} to {to_format}")
        print(f"Saved to: {output_path}")
        
        return output_data

# Usage example
def convert_formats():
    converter = DatasetFormatConverter()
    
    # Convert Alpaca to Chat format
    converter.convert_dataset(
        input_path="./data/datasets/sql_dataset_alpaca.json",
        output_path="./data/datasets/sql_dataset_chat.json",
        from_format="alpaca",
        to_format="chat"
    )
    
    # Convert Alpaca to Completion format
    converter.convert_dataset(
        input_path="./data/datasets/sql_dataset_alpaca.json",
        output_path="./data/datasets/sql_dataset_completion.json",
        from_format="alpaca",
        to_format="completion"
    )

if __name__ == "__main__":
    convert_formats()
```

## Best Practices for Dataset Creation

### 1. Quality Over Quantity
- **Minimum viable dataset**: 100-500 high-quality examples
- **Production dataset**: 1,000-5,000 examples
- **Research dataset**: 10,000+ examples

### 2. Balanced Distribution
```python
# Check dataset balance
import numpy as np
def analyze_dataset_distribution(dataset_path: str):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Analyze difficulty distribution
    difficulties = [ex.get('difficulty', 'unknown') for ex in data]
    difficulty_counts = pd.Series(difficulties).value_counts()
    
    # Analyze length distribution
    output_lengths = [len(ex['output']) for ex in data]
    
    print("📊 Dataset Distribution Analysis")
    print("=" * 40)
    print(f"Total examples: {len(data)}")
    print(f"\nDifficulty distribution:")
    print(difficulty_counts)
    print(f"\nOutput length statistics:")
    print(f"Mean: {np.mean(output_lengths):.1f} characters")
    print(f"Median: {np.median(output_lengths):.1f} characters")
    print(f"Min: {min(output_lengths)} characters")
    print(f"Max: {max(output_lengths)} characters")
```

### 3. Domain-Specific Considerations

**For SQL Generation:**
- Cover all major SQL operations (SELECT, INSERT, UPDATE, DELETE)
- Include various JOIN types
- Add window functions and CTEs for advanced cases
- Validate all SQL queries for syntax correctness

**For Code Generation:**  
- Include multiple programming languages
- Cover different complexity levels
- Add error handling examples
- Include best practices and common patterns

**For Text Analysis:**
- Ensure diverse text types and domains
- Include various output formats
- Cover different analysis types (sentiment, summarization, etc.)

## Preparing Data for Training

### Loading and Inspecting Datasets

Before we start training, it's important to load and inspect our datasets to ensure they are in the correct format. We can use the `datasets` library to do this:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("json", data_files="./data/datasets/sql_dataset_alpaca.json")

# Inspect the dataset
print(dataset)

# Print the first example
print(dataset["train"][0])
```

### Final Dataset Preparation Script

```python
# prepare_training_data.py
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import os

def prepare_final_dataset(dataset_path: str, test_size: float = 0.1, 
                         format_type: str = "alpaca"):
    """Prepare final dataset for training"""
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Dataset Statistics:")
    print(f"Total examples: {len(data)}")
    
    # Split into train/validation
    train_data, val_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=42,
        shuffle=True
    )
    
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Format for Unsloth training
    def format_example(example):
        if format_type == "alpaca":
            if example.get('input', '').strip():
                text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>{example['instruction']}\n\n{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{example['output']}<|eot_id|><|end_of_text|>"
            else:
                text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{example['output']}<|eot_id|><|end_of_text|>"
            
            return {"text": text}
    
    # Format training data
    formatted_train = [format_example(ex) for ex in train_data]
    formatted_val = [format_example(ex) for ex in val_data]
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_list(formatted_train)
    val_dataset = Dataset.from_list(formatted_val)
    
    # Save processed datasets
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset.save_to_disk(f"{output_dir}/train_dataset")
    val_dataset.save_to_disk(f"{output_dir}/val_dataset")
    
    # Also save as JSON for backup
    with open(f"{output_dir}/train_data.json", 'w') as f:
        json.dump(formatted_train, f, indent=2)
    
    with open(f"{output_dir}/val_data.json", 'w') as f:
        json.dump(formatted_val, f, indent=2)
    
    print(f"✅ Datasets saved to {output_dir}")
    
    # Show example
    print(f"\n📝 Training Example Preview:")
    print(formatted_train[0]['text'][:300] + "...")
    
    return train_dataset, val_dataset

# Usage
if __name__ == "__main__":
    train_ds, val_ds = prepare_final_dataset(
        "./data/datasets/sql_dataset_alpaca.json",
        test_size=0.15,
        format_type="alpaca"
    )
```

## 📁 Reference Code Repository

All data preparation code and examples are available in the GitHub repository:

**🔗 [fine-tuning-small-llms/part2-data-preparation](https://github.com/saptak/fine-tuning-small-llms/tree/main/part2-data-preparation)**

```bash
# Clone the repository if you haven't already
git clone https://github.com/saptak/fine-tuning-small-llms.git
cd fine-tuning-small-llms

# Create a SQL dataset
python part2-data-preparation/src/dataset_creation.py --output-dir ./data/datasets --format alpaca

# Validate the dataset
python part2-data-preparation/src/data_validation.py --dataset ./data/datasets/sql_dataset_alpaca.json
```

The Part 2 directory includes:
- `src/dataset_creation.py` - Complete dataset creation toolkit
- `src/data_validation.py` - Quality validation framework  
- `src/format_converter.py` - Format conversion utilities
- `src/model_selection.py` - Smart model recommendation system
- `examples/` - Sample datasets and templates
- Documentation and usage guides

## What's Next?

Excellent work! You now have a comprehensive understanding of data preparation and model selection. In our next installment, we'll put this knowledge to work:

**[Part 3: Fine-Tuning with Unsloth](/2025/07/25/fine-tuning-small-llms-part3-training/)**

In Part 3, you'll learn:
- Setting up Unsloth for efficient training
- Configuring LoRA adapters for parameter-efficient fine-tuning
- Running the complete training pipeline
- Monitoring training progress with Weights & Biases
- Saving and managing model checkpoints

### Key Takeaways from Part 2

1. **Quality > Quantity**: 500 high-quality examples beat 5,000 mediocre ones
2. **Format Matters**: Choose the right format for your training approach
3. **Model Selection**: Match your base model to your use case and resources
4. **Validation**: Always validate your data before training
5. **Preparation**: Proper formatting saves debugging time later

## Resources and Tools

**Dataset Creation:**
- [SQL Teaching](https://sqlteaching.com/) - For SQL example inspiration
- [Mockaroo](https://mockaroo.com/) - Generate synthetic data
- [JSONLint](https://jsonlint.com/) - Validate JSON formats

**Model Information:**
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Unsloth Supported Models](https://github.com/unslothai/unsloth)
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) - Model comparisons

---

*Continue to [Part 3: Fine-Tuning with Unsloth](/2025/07/25/fine-tuning-small-llms-part3-training/) to start training your model!*
