---
layout: post
title: "Fine-Tuning Small LLMs with Docker Desktop - Part 4: Evaluation and Testing"
date: 2025-07-25 12:00:00 -0800
categories: [AI, Machine Learning, Docker, LLM, Fine-tuning]
tags: [llm, evaluation, testing, metrics, benchmarking, validation]
author: Saptak
description: "Part 4 of our comprehensive series. Learn how to evaluate your fine-tuned model with comprehensive testing frameworks, automated benchmarks, and quality assurance techniques."
featured_image: "/assets/images/llm-fine-tuning-part4.jpg"
series: "Fine-Tuning Small LLMs with Docker Desktop"
part: 4
toc: true
repository: "https://github.com/saptak/fine-tuning-small-llms"
---

> 📚 **Reference Code Available**: All evaluation frameworks and testing utilities are available in the [GitHub repository](https://github.com/saptak/fine-tuning-small-llms). See `part4-evaluation/` for comprehensive testing tools!

# Fine-Tuning Small LLMs with Docker Desktop - Part 4: Evaluation and Testing

Welcome back! In [Part 3](/2025/07/25/fine-tuning-small-llms-part3-training/), we successfully fine-tuned our model using Unsloth. Now comes the critical phase: **rigorous evaluation and testing**. This determines whether your fine-tuning was successful and your model is ready for production use.

## Series Navigation

1. [Part 1: Setup and Environment](/2025/07/25/fine-tuning-small-llms-part1-setup-environment/)
2. [Part 2: Data Preparation and Model Selection](/2025/07/25/fine-tuning-small-llms-part2-data-preparation/)
3. [Part 3: Fine-Tuning with Unsloth](/2025/07/25/fine-tuning-small-llms-part3-training/)
4. **Part 4: Evaluation and Testing** (This post)
5. [Part 5: Deployment with Ollama and Docker](/2025/07/25/fine-tuning-small-llms-part5-deployment/)
6. [Part 6: Production, Monitoring, and Scaling](/2025/07/25/fine-tuning-small-llms-part6-production/)

## Why Comprehensive Evaluation Matters

Evaluation is not just about checking if your model works—it's about understanding:

- **Performance**: How well does it perform compared to the base model?
- **Quality**: Are the outputs accurate and helpful?
- **Consistency**: Does it perform reliably across different inputs?
- **Safety**: Does it avoid harmful or inappropriate outputs?
- **Efficiency**: How fast and resource-efficient is it?

### The Evaluation Framework

Our evaluation approach covers multiple dimensions:

```
📊 Evaluation Framework
├── 🎯 Accuracy Metrics
│   ├── ROUGE scores
│   ├── BLEU scores
│   └── Domain-specific metrics
├── 🧪 Functional Testing
│   ├── Unit tests
│   ├── Integration tests
│   └── Edge case testing
├── ⚡ Performance Benchmarking
│   ├── Latency measurements
│   ├── Throughput testing
│   └── Resource utilization
├── 👥 Human Evaluation
│   ├── Quality assessment
│   ├── Usefulness ratings
│   └── Comparative analysis
└── 🔒 Safety & Robustness
    ├── Bias detection
    ├── Adversarial testing
    └── Failure analysis
```

## Setting Up the Evaluation Environment

Let's start by creating a comprehensive evaluation framework:

```python
# evaluation_framework.py
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime
import asyncio
import aiohttp
import requests
from dataclasses import dataclass
from pathlib import Path

# Evaluation metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import sqlite3
import sqlparse
from difflib import SequenceMatcher

@dataclass
class EvaluationResult:
    """Structure for evaluation results"""
    test_id: str
    category: str
    input_text: str
    expected_output: str
    generated_output: str
    metrics: Dict[str, float]
    timestamp: datetime
    model_name: str
    latency_ms: float
    success: bool
    error_message: str = None

class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for fine-tuned models"""
    
    def __init__(self, model_path: str, model_name: str = "fine-tuned-model"):
        self.model_path = model_path
        self.model_name = model_name
        self.results: List[EvaluationResult] = []
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.bleu_smoother = SmoothingFunction().method4
        
        # Load model and tokenizer
        print(f"📥 Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> Tuple[str, float]:
        """Generate response and measure latency"""
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            generated_part = response[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
            
            latency_ms = (time.time() - start_time) * 1000
            return generated_part.strip(), latency_ms
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return f"Error: {str(e)}", latency_ms
    
    def calculate_rouge_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure,
        }
    
    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score"""
        reference_tokens = reference.split()
        generated_tokens = generated.split()
        
        if len(generated_tokens) == 0:
            return 0.0
        
        return sentence_bleu(
            [reference_tokens], 
            generated_tokens, 
            smoothing_function=self.bleu_smoother
        )
    
    def calculate_sql_specific_metrics(self, generated_sql: str, expected_sql: str) -> Dict[str, float]:
        """Calculate SQL-specific evaluation metrics"""
        metrics = {}
        
        # Syntax validity
        try:
            sqlparse.parse(generated_sql)
            metrics['sql_syntax_valid'] = 1.0
        except:
            metrics['sql_syntax_valid'] = 0.0
        
        # Normalize SQL for comparison
        def normalize_sql(sql):
            try:
                parsed = sqlparse.parse(sql.strip())[0]
                return sqlparse.format(str(parsed), reindent=True, keyword_case='upper')
            except:
                return sql.strip().upper()
        
        norm_generated = normalize_sql(generated_sql)
        norm_expected = normalize_sql(expected_sql)
        
        # Structural similarity
        metrics['sql_structural_similarity'] = SequenceMatcher(
            None, norm_generated, norm_expected
        ).ratio()
        
        # Exact match
        metrics['sql_exact_match'] = 1.0 if norm_generated == norm_expected else 0.0
        
        # Keyword presence
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'JOIN']
        expected_keywords = [kw for kw in sql_keywords if kw in norm_expected]
        generated_keywords = [kw for kw in expected_keywords if kw in norm_generated]
        
        if expected_keywords:
            metrics['sql_keyword_coverage'] = len(generated_keywords) / len(expected_keywords)
        else:
            metrics['sql_keyword_coverage'] = 1.0
        
        return metrics
    
    def evaluate_single_example(self, test_case: Dict) -> EvaluationResult:
        """Evaluate a single test example"""
        
        # Generate response
        generated_output, latency_ms = self.generate_response(
            test_case['prompt'],
            max_new_tokens=test_case.get('max_tokens', 256)
        )
        
        # Calculate metrics
        metrics = {}
        success = True
        error_message = None
        
        try:
            expected_output = test_case.get('expected_output', '')
            
            # ROUGE metrics
            rouge_metrics = self.calculate_rouge_metrics(generated_output, expected_output)
            metrics.update(rouge_metrics)
            
            # BLEU score
            metrics['bleu_score'] = self.calculate_bleu_score(generated_output, expected_output)
            
            # Domain-specific metrics
            if test_case.get('category') == 'sql':
                sql_metrics = self.calculate_sql_specific_metrics(generated_output, expected_output)
                metrics.update(sql_metrics)
            
            # Length metrics
            metrics['output_length'] = len(generated_output)
            metrics['length_ratio'] = len(generated_output) / max(len(expected_output), 1)
            
        except Exception as e:
            success = False
            error_message = str(e)
            print(f"❌ Error evaluating test {test_case.get('id', 'unknown')}: {e}")
        
        return EvaluationResult(
            test_id=test_case.get('id', f"test_{len(self.results)}"),
            category=test_case.get('category', 'general'),
            input_text=test_case['prompt'],
            expected_output=test_case.get('expected_output', ''),
            generated_output=generated_output,
            metrics=metrics,
            timestamp=datetime.now(),
            model_name=self.model_name,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message
        )
    
    def run_evaluation_suite(self, test_cases: List[Dict]) -> List[EvaluationResult]:
        """Run comprehensive evaluation on test suite"""
        
        print(f"🧪 Running evaluation on {len(test_cases)} test cases")
        print("=" * 60)
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i:3d}/{len(test_cases)} | {test_case.get('category', 'general'):10s} | ", end="")
            
            result = self.evaluate_single_example(test_case)
            results.append(result)
            
            # Print progress
            if result.success:
                primary_metric = result.metrics.get('rouge1_f', result.metrics.get('bleu_score', 0))
                print(f"Score: {primary_metric:.3f} | Latency: {result.latency_ms:.0f}ms")
            else:
                print(f"❌ FAILED: {result.error_message}")
        
        self.results.extend(results)
        return results
    
    def generate_evaluation_report(self, results: List[EvaluationResult]) -> Dict:
        """Generate comprehensive evaluation report"""
        
        if not results:
            return {"error": "No results to analyze"}
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful evaluations"}
        
        # Overall statistics
        report = {
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_results),
                "success_rate": len(successful_results) / len(results) * 100,
                "average_latency_ms": np.mean([r.latency_ms for r in successful_results]),
                "median_latency_ms": np.median([r.latency_ms for r in successful_results])
            },
            "metrics": {},
            "category_analysis": {},
            "recommendations": []
        }
        
        # Aggregate metrics
        all_metrics = {}
        for result in successful_results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        for metric_name, values in all_metrics.items():
            report["metrics"][metric_name] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        # Category analysis
        categories = set(r.category for r in successful_results)
        for category in categories:
            category_results = [r for r in successful_results if r.category == category]
            
            category_metrics = {}
            for result in category_results:
                for metric_name, value in result.metrics.items():
                    if metric_name not in category_metrics:
                        category_metrics[metric_name] = []
                    category_metrics[metric_name].append(value)
            
            category_analysis = {}
            for metric_name, values in category_metrics.items():
                category_analysis[metric_name] = np.mean(values)
            
            report["category_analysis"][category] = {
                "count": len(category_results),
                "avg_latency_ms": np.mean([r.latency_ms for r in category_results]),
                "metrics": category_analysis
            }
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate improvement recommendations based on results"""
        recommendations = []
        
        metrics = report.get("metrics", {})
        summary = report.get("summary", {})
        
        # Success rate recommendations
        if summary.get("success_rate", 0) < 95:
            recommendations.append("🔧 Success rate is below 95% - investigate failing test cases")
        
        # Performance recommendations
        avg_latency = summary.get("average_latency_ms", 0)
        if avg_latency > 5000:
            recommendations.append("⚡ High latency detected - consider model optimization or quantization")
        elif avg_latency > 2000:
            recommendations.append("⚡ Moderate latency - monitor performance in production")
        
        # Quality recommendations
        rouge1_score = metrics.get("rouge1_f", {}).get("mean", 0)
        if rouge1_score < 0.3:
            recommendations.append("📈 Low ROUGE-1 score - consider additional training or data quality improvement")
        elif rouge1_score < 0.5:
            recommendations.append("📈 Moderate ROUGE-1 score - model performs adequately but has room for improvement")
        
        # SQL-specific recommendations
        if "sql_syntax_valid" in metrics:
            syntax_validity = metrics["sql_syntax_valid"]["mean"]
            if syntax_validity < 0.8:
                recommendations.append("🔍 Low SQL syntax validity - review training data for syntax errors")
        
        if not recommendations:
            recommendations.append("✅ Model performance looks good across all metrics!")
        
        return recommendations
    
    def save_detailed_results(self, results: List[EvaluationResult], output_dir: str = "./evaluation_results"):
        """Save detailed evaluation results"""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        results_data = []
        for result in results:
            results_data.append({
                "test_id": result.test_id,
                "category": result.category,
                "input_text": result.input_text,
                "expected_output": result.expected_output,
                "generated_output": result.generated_output,
                "metrics": result.metrics,
                "timestamp": result.timestamp.isoformat(),
                "model_name": result.model_name,
                "latency_ms": result.latency_ms,
                "success": result.success,
                "error_message": result.error_message
            })
        
        json_path = f"{output_dir}/detailed_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for analysis
        df_data = []
        for result in results:
            row = {
                "test_id": result.test_id,
                "category": result.category,
                "success": result.success,
                "latency_ms": result.latency_ms,
                "model_name": result.model_name
            }
            row.update(result.metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_path = f"{output_dir}/results_summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Detailed results saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        
        return json_path, csv_path

print("✅ Comprehensive evaluation framework loaded")
```

## Creating Test Suites

Now let's create comprehensive test suites for different scenarios:

```python
# test_suites.py
def create_sql_test_suite() -> List[Dict]:
    """Create comprehensive SQL test suite"""
    
    test_cases = [
        # Basic SELECT operations
        {
            "id": "sql_basic_001",
            "category": "sql",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Select all columns from the users table<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "SELECT * FROM users;",
            "max_tokens": 128
        },
        {
            "id": "sql_basic_002",
            "category": "sql",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Find all users where the status is 'active'\n\nTable Schema: users (id, name, email, status)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "SELECT * FROM users WHERE status = 'active';",
            "max_tokens": 128
        },
        
        # Filtered queries
        {
            "id": "sql_filter_001",
            "category": "sql",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Find all users who registered in the last 30 days\n\nTable Schema: users (id, name, email, registration_date)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "SELECT * FROM users WHERE registration_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);",
            "max_tokens": 128
        },
        
        # Aggregation queries
        {
            "id": "sql_agg_001", 
            "category": "sql",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Count the total number of orders per customer\n\nTable Schema: orders (id, customer_id, amount, order_date)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id;",
            "max_tokens": 128
        },
        
        # JOIN operations
        {
            "id": "sql_join_001",
            "category": "sql",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Show customer names with their total order amounts\n\nTable Schema: customers (id, name, email), orders (id, customer_id, amount)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "SELECT c.name, SUM(o.amount) as total_amount FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name;",
            "max_tokens": 256
        },
        
        # Complex analytical queries
        {
            "id": "sql_complex_001",
            "category": "sql",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Find the top 5 customers by total spending\n\nTable Schema: customers (id, name), orders (id, customer_id, amount)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "SELECT c.name, SUM(o.amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total_spent DESC LIMIT 5;",
            "max_tokens": 256
        },
        
        # Window functions
        {
            "id": "sql_window_001",
            "category": "sql",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Rank customers by their spending within each region\n\nTable Schema: customers (id, name, region), orders (id, customer_id, amount)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "SELECT c.name, c.region, SUM(o.amount) as total_spent, RANK() OVER (PARTITION BY c.region ORDER BY SUM(o.amount) DESC) as spending_rank FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name, c.region;",
            "max_tokens": 512
        },
        
        # Edge cases
        {
            "id": "sql_edge_001",
            "category": "sql",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Handle NULL values when calculating average order amount\n\nTable Schema: orders (id, customer_id, amount)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "SELECT AVG(COALESCE(amount, 0)) as avg_amount FROM orders;",
            "max_tokens": 128
        }
    ]
    
    return test_cases

def create_stress_test_suite() -> List[Dict]:
    """Create stress test suite for performance evaluation"""
    
    # Generate repeated tests with slight variations
    base_test = {
        "category": "sql",
        "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Find all users who registered recently\n\nTable Schema: users (id, name, email, registration_date)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "expected_output": "SELECT * FROM users WHERE registration_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);",
        "max_tokens": 128
    }
    
    stress_tests = []
    for i in range(50):  # 50 similar tests
        test = base_test.copy()
        test["id"] = f"stress_test_{i:03d}"
        stress_tests.append(test)
    
    return stress_tests

def create_edge_case_suite() -> List[Dict]:
    """Create edge case test suite"""
    
    edge_cases = [
        # Empty/minimal input
        {
            "id": "edge_empty_001",
            "category": "edge",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>SQL<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "I need more information to write a SQL query. Please provide the table schema and describe what you want to accomplish.",
            "max_tokens": 256
        },
        
        # Very long input
        {
            "id": "edge_long_001",
            "category": "edge", 
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Generate a very complex SQL query that joins multiple tables, uses window functions, CTEs, and subqueries to analyze customer behavior patterns across different time periods and geographical regions with detailed aggregations and sophisticated filtering conditions\n\nTable Schema: customers (id, name, email, region, registration_date), orders (id, customer_id, amount, order_date, status), products (id, name, category, price), order_items (id, order_id, product_id, quantity)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "WITH customer_metrics AS (\n  SELECT c.id, c.name, c.region, \n         COUNT(o.id) as order_count,\n         SUM(o.amount) as total_spent,\n         AVG(o.amount) as avg_order_value\n  FROM customers c\n  LEFT JOIN orders o ON c.id = o.customer_id\n  WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)\n  GROUP BY c.id, c.name, c.region\n)\nSELECT region, \n       COUNT(*) as customer_count,\n       AVG(total_spent) as avg_customer_value,\n       RANK() OVER (ORDER BY AVG(total_spent) DESC) as region_rank\nFROM customer_metrics\nGROUP BY region\nORDER BY avg_customer_value DESC;",
            "max_tokens": 512
        },
        
        # Ambiguous requests
        {
            "id": "edge_ambiguous_001",
            "category": "edge",
            "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert SQL developer.<|eot_id|><|start_header_id|>user<|end_header_id|>Get the things from the place where stuff happened<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "expected_output": "I need more specific information to write a SQL query. Please provide:\n1. Table name and schema\n2. What columns you want to select\n3. What conditions to filter by",
            "max_tokens": 256
        }
    ]
    
    return edge_cases

print("✅ Test suites created")
```

## Running Comprehensive Evaluations

```python
# run_evaluation.py
async def run_comprehensive_evaluation():
    """Run complete evaluation pipeline"""
    
    print("🚀 Starting Comprehensive Model Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    model_path = "./models/sql-expert-merged"  # Adjust path as needed
    evaluator = ComprehensiveEvaluator(model_path, "sql-expert-v1")
    
    # Create test suites
    print("📋 Creating test suites...")
    sql_tests = create_sql_test_suite()
    edge_tests = create_edge_case_suite()
    stress_tests = create_stress_test_suite()
    
    all_tests = sql_tests + edge_tests + stress_tests[:10]  # Limit stress tests for demo
    
    print(f"Created {len(all_tests)} test cases:")
    print(f"  - SQL tests: {len(sql_tests)}")
    print(f"  - Edge cases: {len(edge_tests)}")
    print(f"  - Stress tests: {min(len(stress_tests), 10)}")
    
    # Run evaluations
    print(f"\n🧪 Running evaluations...")
    results = evaluator.run_evaluation_suite(all_tests)
    
    # Generate comprehensive report
    print(f"\n📊 Generating evaluation report...")
    report = evaluator.generate_evaluation_report(results)
    
    # Print summary
    print(f"\n📋 Evaluation Summary")
    print("=" * 40)
    print(f"Total tests: {report['summary']['total_tests']}")
    print(f"Successful tests: {report['summary']['successful_tests']}")
    print(f"Success rate: {report['summary']['success_rate']:.1f}%")
    print(f"Average latency: {report['summary']['average_latency_ms']:.0f}ms")
    
    # Key metrics
    print(f"\n📈 Key Metrics:")
    if 'rouge1_f' in report['metrics']:
        print(f"ROUGE-1 F1: {report['metrics']['rouge1_f']['mean']:.3f}")
    if 'bleu_score' in report['metrics']:
        print(f"BLEU Score: {report['metrics']['bleu_score']['mean']:.3f}")
    if 'sql_syntax_valid' in report['metrics']:
        print(f"SQL Syntax Valid: {report['metrics']['sql_syntax_valid']['mean']:.1%}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Save detailed results
    json_path, csv_path = evaluator.save_detailed_results(results)
    
    # Save report
    report_path = f"./evaluation_results/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Results saved:")
    print(f"  - Report: {report_path}")
    print(f"  - Details: {json_path}")
    print(f"  - Summary: {csv_path}")
    
    return report, results

# Run evaluation
if __name__ == "__main__":
    import asyncio
    report, results = asyncio.run(run_comprehensive_evaluation())
```

## Performance Benchmarking

```python
# performance_benchmark.py
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd

class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    def __init__(self, evaluator: ComprehensiveEvaluator):
        self.evaluator = evaluator
        self.benchmark_results = []
    
    def measure_throughput(self, test_cases, num_concurrent=1, duration_seconds=60):
        """Measure model throughput"""
        
        print(f"📊 Measuring throughput ({num_concurrent} concurrent, {duration_seconds}s)")
        
        start_time = time.time()
        completed_requests = 0
        errors = 0
        latencies = []
        
        def worker():
            nonlocal completed_requests, errors
            while time.time() - start_time < duration_seconds:
                try:
                    test_case = test_cases[completed_requests % len(test_cases)]
                    _, latency = self.evaluator.generate_response(test_case['prompt'])
                    latencies.append(latency)
                    completed_requests += 1
                except Exception:
                    errors += 1
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(worker) for _ in range(num_concurrent)]
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker error: {e}")
        
        actual_duration = time.time() - start_time
        
        return {
            'requests_completed': completed_requests,
            'errors': errors,
            'duration_seconds': actual_duration,
            'throughput_rps': completed_requests / actual_duration,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'median_latency_ms': sorted(latencies)[len(latencies)//2] if latencies else 0,
            'p95_latency_ms': sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0
        }
    
    def measure_resource_usage(self, test_cases, num_requests=50):
        """Measure CPU, memory, and GPU usage during inference"""
        
        print(f"📊 Measuring resource usage ({num_requests} requests)")
        
        # Resource monitoring
        cpu_usage = []
        memory_usage = []
        gpu_usage = []
        
        def monitor_resources():
            while self.monitoring:
                cpu_usage.append(psutil.cpu_percent())
                memory_usage.append(psutil.virtual_memory().percent)
                
                # GPU monitoring (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage.append(gpus[0].load * 100)
                    else:
                        gpu_usage.append(0)
                except:
                    gpu_usage.append(0)
                
                time.sleep(0.1)  # Sample every 100ms
        
        # Start monitoring
        self.monitoring = True
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Run inference
        start_time = time.time()
        for i in range(num_requests):
            test_case = test_cases[i % len(test_cases)]
            self.evaluator.generate_response(test_case['prompt'], max_new_tokens=128)
            
            if i % 10 == 0:
                print(f"  Progress: {i+1}/{num_requests}")
        
        # Stop monitoring
        self.monitoring = False
        monitor_thread.join()
        
        duration = time.time() - start_time
        
        return {
            'duration_seconds': duration,
            'requests_per_second': num_requests / duration,
            'avg_cpu_percent': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
            'max_cpu_percent': max(cpu_usage) if cpu_usage else 0,
            'avg_memory_percent': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            'max_memory_percent': max(memory_usage) if memory_usage else 0,
            'avg_gpu_percent': sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0,
            'max_gpu_percent': max(gpu_usage) if gpu_usage else 0
        }
    
    def run_scalability_test(self, test_cases):
        """Test performance under different load levels"""
        
        print("📊 Running scalability test")
        
        concurrent_levels = [1, 2, 4, 8, 16]
        scalability_results = []
        
        for concurrency in concurrent_levels:
            print(f"  Testing {concurrency} concurrent requests...")
            
            result = self.measure_throughput(
                test_cases, 
                num_concurrent=concurrency, 
                duration_seconds=30
            )
            
            result['concurrency'] = concurrency
            scalability_results.append(result)
            
            print(f"    Throughput: {result['throughput_rps']:.1f} RPS")
            print(f"    Avg latency: {result['avg_latency_ms']:.0f}ms")
        
        return scalability_results
    
    def generate_performance_report(self, scalability_results, resource_results):
        """Generate performance analysis report"""
        
        report = {
            "scalability_analysis": {
                "max_throughput_rps": max(r['throughput_rps'] for r in scalability_results),
                "optimal_concurrency": None,  # Will be calculated
                "latency_trend": "stable",  # Will be analyzed
            },
            "resource_efficiency": {
                "cpu_efficiency": resource_results['avg_cpu_percent'],
                "memory_efficiency": resource_results['avg_memory_percent'],
                "gpu_utilization": resource_results['avg_gpu_percent']
            },
            "recommendations": []
        }
        
        # Find optimal concurrency (best throughput with acceptable latency)
        acceptable_latency_ms = 2000  # 2 seconds
        viable_configs = [
            r for r in scalability_results 
            if r['avg_latency_ms'] <= acceptable_latency_ms
        ]
        
        if viable_configs:
            optimal = max(viable_configs, key=lambda x: x['throughput_rps'])
            report["scalability_analysis"]["optimal_concurrency"] = optimal['concurrency']
        
        # Generate recommendations
        recommendations = []
        
        if resource_results['avg_cpu_percent'] > 80:
            recommendations.append("High CPU usage - consider model quantization or optimization")
        
        if resource_results['avg_memory_percent'] > 80:
            recommendations.append("High memory usage - consider reducing model size or batch size")
        
        if resource_results['avg_gpu_percent'] < 50:
            recommendations.append("Low GPU utilization - consider increasing batch size or concurrent requests")
        
        max_throughput = max(r['throughput_rps'] for r in scalability_results)
        if max_throughput < 1:
            recommendations.append("Low throughput - consider model optimization or hardware upgrade")
        
        report["recommendations"] = recommendations
        
        return report

# Usage
def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    
    # Initialize
    model_path = "./models/sql-expert-merged"
    evaluator = ComprehensiveEvaluator(model_path, "sql-expert-v1")
    benchmark = PerformanceBenchmark(evaluator)
    
    # Create test cases
    test_cases = create_sql_test_suite()[:5]  # Use subset for benchmarking
    
    print("🚀 Starting Performance Benchmark")
    print("=" * 50)
    
    # Run scalability test
    scalability_results = benchmark.run_scalability_test(test_cases)
    
    # Run resource usage test
    resource_results = benchmark.measure_resource_usage(test_cases, num_requests=20)
    
    # Generate report
    perf_report = benchmark.generate_performance_report(scalability_results, resource_results)
    
    # Print results
    print(f"\n📊 Performance Report")
    print("=" * 40)
    print(f"Max throughput: {perf_report['scalability_analysis']['max_throughput_rps']:.1f} RPS")
    print(f"Optimal concurrency: {perf_report['scalability_analysis']['optimal_concurrency']}")
    print(f"CPU efficiency: {perf_report['resource_efficiency']['cpu_efficiency']:.1f}%")
    print(f"Memory efficiency: {perf_report['resource_efficiency']['memory_efficiency']:.1f}%")
    print(f"GPU utilization: {perf_report['resource_efficiency']['gpu_utilization']:.1f}%")
    
    print(f"\n💡 Performance Recommendations:")
    for rec in perf_report['recommendations']:
        print(f"  • {rec}")
    
    return perf_report, scalability_results, resource_results

if __name__ == "__main__":
    perf_report, scalability_results, resource_results = run_performance_benchmark()
```

## A/B Testing Framework

### Detailed Evaluation with ExplainaBoard

ExplainaBoard is a library for detailed analysis of NLP models. It can be used to get a more in-depth understanding of your model's performance, including fine-grained error analysis.

To use ExplainaBoard, you first need to install it:

```bash
pip install explainaboard
```

Then, you can use the following code to generate a detailed evaluation report:

```python
from explainaboard import get_processor, get_loader, get_explainer

# Load the data
loader = get_loader("from_dict", data=test_cases)

# Get the processor
processor = get_processor("text-classification")

# Get the explainer
explainer = get_explainer(processor)

# Generate the report
report = explainer.explain(loader)

# Print the report
report.print_summary()
```

```python
# ab_testing.py
import random
from typing import Dict, List, Tuple
from scipy import stats

class ABTestFramework:
    """A/B testing framework for model comparison"""
    
    def __init__(self):
        self.experiments = {}
        self.results = {}
    
    def create_experiment(self, name: str, model_a_path: str, model_b_path: str, 
                         test_cases: List[Dict], traffic_split: float = 0.5):
        """Create A/B test experiment"""
        
        self.experiments[name] = {
            'model_a': ComprehensiveEvaluator(model_a_path, "model_a"),
            'model_b': ComprehensiveEvaluator(model_b_path, "model_b"),
            'test_cases': test_cases,
            'traffic_split': traffic_split,
            'results_a': [],
            'results_b': []
        }
        
        print(f"🧪 Created A/B test experiment: {name}")
        print(f"  Model A: {model_a_path}")
        print(f"  Model B: {model_b_path}")
        print(f"  Test cases: {len(test_cases)}")
        print(f"  Traffic split: {traffic_split:.1%} / {1-traffic_split:.1%}")
    
    def run_experiment(self, experiment_name: str) -> Dict:
        """Run A/B test experiment"""
        
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        exp = self.experiments[experiment_name]
        
        print(f"🚀 Running A/B test: {experiment_name}")
        print("=" * 50)
        
        # Run tests with traffic split
        for i, test_case in enumerate(exp['test_cases']):
            # Determine which model to use
            use_model_a = random.random() < exp['traffic_split']
            
            if use_model_a:
                result = exp['model_a'].evaluate_single_example(test_case)
                exp['results_a'].append(result)
                print(f"Test {i+1:3d} | Model A | {'✅' if result.success else '❌'}")
            else:
                result = exp['model_b'].evaluate_single_example(test_case)
                exp['results_b'].append(result)
                print(f"Test {i+1:3d} | Model B | {'✅' if result.success else '❌'}")
        
        # Analyze results
        analysis = self.analyze_experiment(experiment_name)
        self.results[experiment_name] = analysis
        
        return analysis
    
    def analyze_experiment(self, experiment_name: str) -> Dict:
        """Analyze A/B test results"""
        
        exp = self.experiments[experiment_name]
        results_a = [r for r in exp['results_a'] if r.success]
        results_b = [r for r in exp['results_b'] if r.success]
        
        if not results_a or not results_b:
            return {"error": "Insufficient successful results for analysis"}
        
        analysis = {
            "sample_sizes": {
                "model_a": len(results_a),
                "model_b": len(results_b)
            },
            "metrics_comparison": {},
            "statistical_significance": {},
            "winner": None,
            "confidence_level": 0.95
        }
        
        # Compare key metrics
        metrics_to_compare = ['rouge1_f', 'bleu_score', 'sql_syntax_valid', 'latency_ms']
        
        for metric in metrics_to_compare:
            if metric == 'latency_ms':
                values_a = [r.latency_ms for r in results_a]
                values_b = [r.latency_ms for r in results_b]
            else:
                values_a = [r.metrics.get(metric, 0) for r in results_a]
                values_b = [r.metrics.get(metric, 0) for r in results_b]
            
            if values_a and values_b:
                mean_a = sum(values_a) / len(values_a)
                mean_b = sum(values_b) / len(values_b)
                
                # Statistical significance test
                try:
                    stat, p_value = stats.ttest_ind(values_a, values_b)
                    is_significant = p_value < 0.05
                    
                    analysis['metrics_comparison'][metric] = {
                        'model_a_mean': mean_a,
                        'model_b_mean': mean_b,
                        'difference': mean_b - mean_a,
                        'relative_improvement': ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0
                    }
                    
                    analysis['statistical_significance'][metric] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'better_model': 'B' if mean_b > mean_a else 'A' if mean_a > mean_b else 'tie'
                    }
                    
                except Exception as e:
                    print(f"Statistical test failed for {metric}: {e}")
        
        # Determine overall winner
        significant_wins_a = sum(1 for m in analysis['statistical_significance'].values() 
                               if m.get('is_significant') and m.get('better_model') == 'A')
        significant_wins_b = sum(1 for m in analysis['statistical_significance'].values() 
                               if m.get('is_significant') and m.get('better_model') == 'B')
        
        if significant_wins_b > significant_wins_a:
            analysis['winner'] = 'Model B'
        elif significant_wins_a > significant_wins_b:
            analysis['winner'] = 'Model A'
        else:
            analysis['winner'] = 'No clear winner'
        
        return analysis
    
    def print_experiment_results(self, experiment_name: str):
        """Print formatted experiment results"""
        
        if experiment_name not in self.results:
            print(f"No results found for experiment: {experiment_name}")
            return
        
        analysis = self.results[experiment_name]
        
        print(f"\n📊 A/B Test Results: {experiment_name}")
        print("=" * 60)
        
        print(f"Sample sizes:")
        print(f"  Model A: {analysis['sample_sizes']['model_a']} tests")
        print(f"  Model B: {analysis['sample_sizes']['model_b']} tests")
        
        print(f"\n📈 Metrics Comparison:")
        for metric, comparison in analysis['metrics_comparison'].items():
            print(f"\n{metric.upper()}:")
            print(f"  Model A: {comparison['model_a_mean']:.4f}")
            print(f"  Model B: {comparison['model_b_mean']:.4f}")
            print(f"  Difference: {comparison['difference']:.4f}")
            print(f"  Improvement: {comparison['relative_improvement']:.1f}%")
            
            if metric in analysis['statistical_significance']:
                sig = analysis['statistical_significance'][metric]
                print(f"  Significant: {'Yes' if sig['is_significant'] else 'No'} (p={sig['p_value']:.4f})")
                print(f"  Better model: {sig['better_model']}")
        
        print(f"\n🏆 Overall Winner: {analysis['winner']}")

# Usage example
def run_ab_test():
    """Run A/B test comparing base model vs fine-tuned model"""
    
    # Initialize A/B test framework
    ab_test = ABTestFramework()
    
    # Create experiment
    ab_test.create_experiment(
        name="base_vs_finetuned",
        model_a_path="unsloth/llama-3.1-8b-instruct-bnb-4bit",  # Base model
        model_b_path="./models/sql-expert-merged",  # Fine-tuned model
        test_cases=create_sql_test_suite()[:10],  # Subset for demo
        traffic_split=0.5
    )
    
    # Run experiment
    results = ab_test.run_experiment("base_vs_finetuned")
    
    # Print results
    ab_test.print_experiment_results("base_vs_finetuned")
    
    return results

if __name__ == "__main__":
    ab_results = run_ab_test()
```

## Quality Assurance Dashboard

```python
# qa_dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_evaluation_dashboard():
    """Create Streamlit dashboard for evaluation results"""
    
    st.set_page_config(
        page_title="LLM Evaluation Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🧪 LLM Fine-Tuning Evaluation Dashboard")
    st.markdown("*Comprehensive analysis of your fine-tuned model performance*")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File uploader for evaluation results
    uploaded_file = st.sidebar.file_uploader(
        "Upload Evaluation Results",
        type=['json', 'csv'],
        help="Upload your evaluation results file"
    )
    
    if uploaded_file is not None:
        # Load results
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = df['success'].mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col2:
            avg_latency = df['latency_ms'].mean()
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
        
        with col3:
            if 'rouge1_f' in df.columns:
                avg_rouge = df['rouge1_f'].mean()
                st.metric("ROUGE-1 F1", f"{avg_rouge:.3f}")
        
        with col4:
            if 'sql_syntax_valid' in df.columns:
                sql_validity = df['sql_syntax_valid'].mean() * 100
                st.metric("SQL Validity", f"{sql_validity:.1f}%")
        
        # Performance over time
        st.subheader("📈 Performance Trends")
        
        # Latency distribution
        fig_latency = px.histogram(
            df, 
            x='latency_ms', 
            title="Latency Distribution",
            nbins=30
        )
        st.plotly_chart(fig_latency, use_container_width=True)
        
        # Metrics by category
        if 'category' in df.columns:
            st.subheader("📊 Performance by Category")
            
            category_metrics = df.groupby('category').agg({
                'success': 'mean',
                'latency_ms': 'mean',
                'rouge1_f': 'mean' if 'rouge1_f' in df.columns else lambda x: 0
            }).round(3)
            
            fig_category = go.Figure()
            
            fig_category.add_trace(go.Bar(
                name='Success Rate',
                x=category_metrics.index,
                y=category_metrics['success'],
                yaxis='y'
            ))
            
            fig_category.add_trace(go.Scatter(
                name='Avg Latency (ms)',
                x=category_metrics.index,
                y=category_metrics['latency_ms'],
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='orange')
            ))
            
            fig_category.update_layout(
                title="Performance Metrics by Category",
                xaxis_title="Category",
                yaxis=dict(title="Success Rate", side="left"),
                yaxis2=dict(title="Latency (ms)", side="right", overlaying="y"),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_category, use_container_width=True)
        
        # Detailed results table
        st.subheader("🔍 Detailed Results")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_successful_only = st.checkbox("Show successful tests only", value=True)
        with col2:
            if 'category' in df.columns:
                selected_categories = st.multiselect(
                    "Filter by category",
                    options=df['category'].unique(),
                    default=df['category'].unique()
                )
            else:
                selected_categories = None
        
        # Apply filters
        filtered_df = df.copy()
        if show_successful_only:
            filtered_df = filtered_df[filtered_df['success'] == True]
        
        if selected_categories:
            filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
        
        # Display table
        display_columns = ['test_id', 'category', 'success', 'latency_ms']
        if 'rouge1_f' in filtered_df.columns:
            display_columns.append('rouge1_f')
        if 'sql_syntax_valid' in filtered_df.columns:
            display_columns.append('sql_syntax_valid')
        
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            height=400
        )
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Generate Report"):
                # Generate summary report
                report = generate_summary_report(df)
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col2:
            if st.button("📈 Export Charts"):
                # Export charts as HTML
                charts_html = export_charts_as_html(df)
                st.download_button(
                    label="Download Charts",
                    data=charts_html,
                    file_name=f"evaluation_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
    
    else:
        st.info("👆 Upload your evaluation results file to view the dashboard")
        
        # Show example format
        st.subheader("📋 Expected File Format")
        
        example_data = {
            'test_id': ['test_001', 'test_002', 'test_003'],
            'category': ['sql', 'sql', 'edge'],
            'success': [True, True, False],
            'latency_ms': [150, 200, 500],
            'rouge1_f': [0.85, 0.92, 0.0],
            'sql_syntax_valid': [1.0, 1.0, 0.0]
        }
        
        st.dataframe(pd.DataFrame(example_data))

def generate_summary_report(df: pd.DataFrame) -> str:
    """Generate markdown summary report"""
    
    success_rate = df['success'].mean() * 100
    avg_latency = df['latency_ms'].mean()
    
    report = f"""# Model Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Tests**: {len(df):,}
- **Success Rate**: {success_rate:.1f}%
- **Average Latency**: {avg_latency:.0f}ms

## Key Metrics
"""
    
    if 'rouge1_f' in df.columns:
        avg_rouge = df['rouge1_f'].mean()
        report += f"- **ROUGE-1 F1**: {avg_rouge:.3f}\n"
    
    if 'sql_syntax_valid' in df.columns:
        sql_validity = df['sql_syntax_valid'].mean() * 100
        report += f"- **SQL Validity**: {sql_validity:.1f}%\n"
    
    # Category breakdown
    if 'category' in df.columns:
        report += "\n## Performance by Category\n"
        category_stats = df.groupby('category').agg({
            'success': 'mean',
            'latency_ms': 'mean'
        }).round(3)
        
        for category, stats in category_stats.iterrows():
            report += f"- **{category.title()}**: {stats['success']*100:.1f}% success, {stats['latency_ms']:.0f}ms avg latency\n"
    
    return report

# Run dashboard
if __name__ == "__main__":
    create_evaluation_dashboard()
```

## 📁 Reference Code Repository

All evaluation frameworks and testing code are available in the GitHub repository:

**🔗 [fine-tuning-small-llms/part4-evaluation](https://github.com/saptak/fine-tuning-small-llms/tree/main/part4-evaluation)**

```bash
# Clone the repository and run evaluations
git clone https://github.com/saptak/fine-tuning-small-llms.git
cd fine-tuning-small-llms

# Run comprehensive evaluation
python part4-evaluation/src/run_evaluation.py --model-path ./models/sql-expert-merged

# Start evaluation dashboard
streamlit run part4-evaluation/src/evaluation_dashboard.py
```

The Part 4 directory includes:
- Comprehensive evaluation framework
- A/B testing utilities
- Performance benchmarking tools
- Quality assurance dashboards
- Test suite generators
- Metrics collection and analysis

## What's Next?

Fantastic work! You've now implemented a comprehensive evaluation framework that thoroughly tests your fine-tuned model across multiple dimensions. Your model has been rigorously validated and is ready for deployment.

**[Part 5: Deployment with Ollama and Docker](/2025/07/25/fine-tuning-small-llms-part5-deployment/)**

In Part 5, you'll learn:
- Converting models to GGUF format for Ollama
- Setting up production-ready Docker containers
- Creating API endpoints for your model
- Building user interfaces with Streamlit
- Implementing load balancing and scaling

### Key Achievements from Part 4

✅ **Comprehensive Testing**: Multi-dimensional evaluation framework  
✅ **Performance Benchmarking**: Throughput and latency analysis  
✅ **A/B Testing**: Statistical comparison between models  
✅ **Quality Assurance**: Automated QA pipelines  
✅ **Interactive Dashboard**: Real-time monitoring and analysis  

## Evaluation Best Practices

1. **Test Early and Often**: Run evaluations throughout development
2. **Use Diverse Test Cases**: Cover edge cases and failure modes
3. **Monitor Performance**: Track latency and resource usage
4. **Statistical Rigor**: Use proper statistical tests for comparisons
5. **Human Evaluation**: Complement automated metrics with human judgment

## Resources and References

- [ROUGE Score Documentation](https://github.com/google-research/google-research/tree/master/rouge)
- [BLEU Score Implementation](https://www.nltk.org/api/nltk.translate.bleu_score.html)
- [A/B Testing Best Practices](https://blog.optimizely.com/2017/10/26/ab-testing-best-practices/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

*Continue to [Part 5: Deployment with Ollama and Docker](/2025/07/25/fine-tuning-small-llms-part5-deployment/) to deploy your validated model!*
