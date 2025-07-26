---
author: Saptak
categories:
- AI
- Machine Learning
- Docker
- LLM
- Fine-tuning
date: 2025-07-25 13:00:00 -0800
description: Part 5 of our comprehensive series. Learn how to deploy your fine-tuned
  model using Ollama and Docker with production-ready APIs, web interfaces, and containerized
  solutions.
featured_image: /assets/images/llm-fine-tuning-part5.jpg
header_image_path: /assets/img/blog/headers/2025-07-25-fine-tuning-small-llms-part5-deployment.jpg
image_credit: Photo by Glen Carrie on Unsplash
layout: post
part: 5
repository: https://github.com/saptak/fine-tuning-small-llms
series: Fine-Tuning Small LLMs with Docker Desktop
tags:
- llm
- deployment
- ollama
- docker
- gguf
- api
- streamlit
- production
thumbnail_path: /assets/img/blog/thumbnails/2025-07-25-fine-tuning-small-llms-part5-deployment.jpg
title: 'Fine-Tuning Small LLMs with Docker Desktop - Part 5: Deployment with Ollama
  and Docker'
toc: true
---

> 📚 **Reference Code Available**: All deployment configurations and production code are available in the [GitHub repository](https://github.com/saptak/fine-tuning-small-llms). See `part5-deployment/` for complete deployment solutions!

# Fine-Tuning Small LLMs with Docker Desktop - Part 5: Deployment with Ollama and Docker

Welcome to Part 5! In [Part 4](/2025/07/25/fine-tuning-small-llms-part4-evaluation/), we thoroughly evaluated our fine-tuned model. Now it's time for the exciting finale: **deploying your model for real-world use**. We'll explore multiple deployment strategies using Ollama, Docker, and create production-ready APIs and interfaces.

## Series Navigation

1. [Part 1: Setup and Environment](/2025/07/25/fine-tuning-small-llms-part1-setup-environment/)
2. [Part 2: Data Preparation and Model Selection](/2025/07/25/fine-tuning-small-llms-part2-data-preparation/)
3. [Part 3: Fine-Tuning with Unsloth](/2025/07/25/fine-tuning-small-llms-part3-training/)
4. [Part 4: Evaluation and Testing](/2025/07/25/fine-tuning-small-llms-part4-evaluation/)
5. **Part 5: Deployment with Ollama and Docker** (This post)
6. [Part 6: Production, Monitoring, and Scaling](/2025/07/25/fine-tuning-small-llms-part6-production/)

## Deployment Architecture Overview

Our deployment strategy encompasses multiple approaches to suit different use cases:

```
🚀 Deployment Architecture
├── 🦙 Ollama Deployment
│   ├── GGUF Model Conversion
│   ├── Local Model Serving
│   └── API Endpoints
├── 🐳 Docker Containers
│   ├── Model Runner Integration
│   ├── Containerized APIs
│   └── Web Interfaces
├── 🌐 Production APIs
│   ├── FastAPI Services
│   ├── Load Balancing
│   └── Authentication
└── 📱 User Interfaces
    ├── Streamlit Dashboard
    ├── Gradio Interface
    └── REST API Clients
```

## Converting Models to GGUF Format

Before deploying with Ollama, we need to convert our fine-tuned model to GGUF format for optimal inference performance:

```python
# model_conversion.py
import os
import subprocess
import json
from pathlib import Path
import requests
import time
from typing import Optional, Dict

class ModelConverter:
    """Utility class for converting models to various formats"""
    
    def __init__(self, model_path: str, output_dir: str = "./converted_models"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_to_gguf_huggingface(self, model_name: str, quantization: str = "Q8_0") -> str:
        """Convert model using Hugging Face GGUF-my-repo service"""
        
        print(f"🔄 Converting {model_name} to GGUF format using Hugging Face service")
        print(f"   Quantization: {quantization}")
        
        # First, upload model to Hugging Face Hub (if not already uploaded)
        hf_model_name = f"your-username/{model_name}"
        
        # Use GGUF-my-repo service
        gguf_my_repo_url = "https://huggingface.co/spaces/ggml-org/gguf-my-repo"
        
        print(f"📋 Manual steps required:")
        print(f"1. Visit: {gguf_my_repo_url}")
        print(f"2. Enter model name: {hf_model_name}")
        print(f"3. Select quantization: {quantization}")
        print(f"4. Click Submit and wait for conversion")
        print(f"5. Download the resulting GGUF file")
        
        # Return expected output path
        expected_output = self.output_dir / f"{model_name}-{quantization.lower()}.gguf"
        return str(expected_output)
    
    def convert_to_gguf_local(self, model_name: str) -> str:
        """Convert model using local llama.cpp installation"""
        
        print(f"🔄 Converting {model_name} to GGUF format locally")
        
        # Check if llama.cpp is available
        llama_cpp_path = self._find_llama_cpp()
        if not llama_cpp_path:
            print("❌ llama.cpp not found. Installing...")
            self._install_llama_cpp()
            llama_cpp_path = self._find_llama_cpp()
        
        if not llama_cpp_path:
            raise Exception("Failed to install llama.cpp")
        
        # Convert model
        output_path = self.output_dir / f"{model_name}.gguf"
        
        convert_cmd = [
            "python", f"{llama_cpp_path}/convert.py",
            str(self.model_path),
            "--outtype", "f16",
            "--outfile", str(output_path)
        ]
        
        print(f"🔧 Running conversion command...")
        try:
            result = subprocess.run(convert_cmd, capture_output=True, text=True, check=True)
            print(f"✅ Model converted successfully: {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"❌ Conversion failed: {e.stderr}")
            raise
    
    def _find_llama_cpp(self) -> Optional[str]:
        """Find llama.cpp installation"""
        common_paths = [
            "./llama.cpp",
            "../llama.cpp",
            "~/llama.cpp",
            "/opt/llama.cpp"
        ]
        
        for path in common_paths:
            expanded_path = Path(path).expanduser()
            if (expanded_path / "convert.py").exists():
                return str(expanded_path)
        
        return None
    
    def _install_llama_cpp(self):
        """Install llama.cpp locally"""
        print("📥 Installing llama.cpp...")
        
        clone_cmd = ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"]
        subprocess.run(clone_cmd, check=True)
        
        # Build llama.cpp
        build_cmd = ["make", "-C", "llama.cpp"]
        subprocess.run(build_cmd, check=True)
        
        print("✅ llama.cpp installed successfully")
    
    def quantize_gguf(self, gguf_path: str, quantization: str = "Q8_0") -> str:
        """Quantize GGUF model for smaller size"""
        
        print(f"🗜️ Quantizing GGUF model to {quantization}")
        
        input_path = Path(gguf_path)
        output_path = input_path.parent / f"{input_path.stem}-{quantization.lower()}.gguf"
        
        llama_cpp_path = self._find_llama_cpp()
        if not llama_cpp_path:
            raise Exception("llama.cpp not found")
        
        quantize_cmd = [
            f"{llama_cpp_path}/quantize",
            str(input_path),
            str(output_path),
            quantization
        ]
        
        try:
            subprocess.run(quantize_cmd, check=True)
            print(f"✅ Model quantized: {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"❌ Quantization failed: {e}")
            raise

# Usage example
def convert_fine_tuned_model():
    """Convert our fine-tuned model to GGUF"""
    
    converter = ModelConverter(
        model_path="./models/sql-expert-merged",
        output_dir="./models/gguf"
    )
    
    # For demonstration, we'll show the manual process
    # In practice, you'd run the conversion
    gguf_path = converter.convert_to_gguf_huggingface("sql-expert", "Q8_0")
    
    print(f"📁 Expected GGUF model path: {gguf_path}")
    print("💡 For this tutorial, we'll assume you have the GGUF file ready")
    
    return gguf_path

if __name__ == "__main__":
    gguf_path = convert_fine_tuned_model()
```

## Setting Up Ollama Deployment

Now let's create a comprehensive Ollama deployment setup:

```python
# ollama_deployment.py
import subprocess
import json
import requests
import time
import os
from pathlib import Path
from typing import Dict, List, Optional

class OllamaDeployment:
    """Comprehensive Ollama deployment manager"""
    
    def __init__(self, ollama_host: str = "localhost", ollama_port: int = 11434):
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.base_url = f"http://{ollama_host}:{ollama_port}"
        
    def check_ollama_service(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def start_ollama_service(self):
        """Start Ollama service"""
        if self.check_ollama_service():
            print("✅ Ollama service is already running")
            return
        
        print("🚀 Starting Ollama service...")
        try:
            # Start Ollama in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for service to start
            for _ in range(30):  # Wait up to 30 seconds
                if self.check_ollama_service():
                    print("✅ Ollama service started successfully")
                    return
                time.sleep(1)
            
            raise Exception("Ollama service failed to start")
            
        except FileNotFoundError:
            raise Exception("Ollama not found. Please install Ollama first.")
    
    def create_model_from_gguf(self, model_name: str, gguf_path: str, 
                              system_prompt: str = None, template: str = None) -> bool:
        """Create Ollama model from GGUF file"""
        
        print(f"🔨 Creating Ollama model: {model_name}")
        
        # Create Modelfile
        modelfile_content = f"FROM {gguf_path}\n"
        
        if template:
            modelfile_content += f'TEMPLATE """{template}"""\n'
        else:
            # Default template for our SQL expert
            modelfile_content += '''TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""\n'''
        
        if system_prompt:
            modelfile_content += f'SYSTEM """{system_prompt}"""\n'
        else:
            modelfile_content += 'SYSTEM """You are an expert SQL developer who generates accurate and efficient SQL queries based on user requirements and table schemas. Always provide clean, well-formatted SQL code."""\n'
        
        # Add parameters
        modelfile_content += "PARAMETER temperature 0.7\n"
        modelfile_content += "PARAMETER top_p 0.9\n"
        modelfile_content += "PARAMETER stop <|eot_id|>\n"
        modelfile_content += "PARAMETER stop <|end_of_text|>\n"
        
        # Save Modelfile
        modelfile_path = Path("./Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        print(f"📄 Modelfile created:")
        print(modelfile_content)
        
        # Create model using Ollama CLI
        try:
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✅ Model {model_name} created successfully")
            
            # Clean up Modelfile
            modelfile_path.unlink()
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create model: {e.stderr}")
            return False
    
    def list_models(self) -> List[Dict]:
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
            else:
                print(f"❌ Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def test_model(self, model_name: str, prompt: str) -> Optional[str]:
        """Test model with a prompt"""
        print(f"🧪 Testing model {model_name}...")
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                print(f"✅ Model response: {generated_text}")
                return generated_text
            else:
                print(f"❌ Request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            return None
    
    def deploy_model(self, model_name: str, gguf_path: str) -> bool:
        """Complete model deployment workflow"""
        
        print(f"🚀 Deploying model: {model_name}")
        print("=" * 50)
        
        # Step 1: Start Ollama service
        self.start_ollama_service()
        
        # Step 2: Create model from GGUF
        if not self.create_model_from_gguf(model_name, gguf_path):
            return False
        
        # Step 3: Test model
        test_prompt = "Generate SQL to find all users who registered in the last 7 days from a users table with columns: id, username, email, registration_date"
        response = self.test_model(model_name, test_prompt)
        
        if response:
            print(f"🎉 Model {model_name} deployed successfully!")
            return True
        else:
            print(f"❌ Model deployment failed")
            return False

# Usage
def deploy_sql_expert_model():
    """Deploy our SQL expert model with Ollama"""
    
    # Initialize deployment manager
    deployer = OllamaDeployment()
    
    # Deploy model (assuming GGUF file exists)
    gguf_path = "./models/gguf/sql-expert-q8_0.gguf"
    
    # Check if GGUF file exists (for demo purposes)
    if not Path(gguf_path).exists():
        print(f"⚠️ GGUF file not found: {gguf_path}")
        print("For this demo, we'll create a placeholder path")
        gguf_path = "path/to/your/sql-expert.gguf"
    
    success = deployer.deploy_model("sql-expert", gguf_path)
    
    if success:
        # List all models
        models = deployer.list_models()
        print(f"\n📋 Available models:")
        for model in models:
            print(f"  - {model.get('name', 'Unknown')}")
    
    return success

if __name__ == "__main__":
    deploy_sql_expert_model()
```

## Creating Production APIs

Let's build a robust FastAPI service for our deployed model:

```python
# api_service.py
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import requests
import time
import json
import asyncio
import aiohttp
from datetime import datetime
import logging
import os
from pathlib import Path
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

class SQLRequest(BaseModel):
    """Request model for SQL generation"""
    instruction: str = Field(..., description="Description of what SQL query to generate")
    table_schema: Optional[str] = Field(None, description="Database table schema information")
    max_tokens: Optional[int] = Field(256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Generation temperature")
    
class SQLResponse(BaseModel):
    """Response model for SQL generation"""
    sql_query: str = Field(..., description="Generated SQL query")
    execution_time_ms: float = Field(..., description="Time taken to generate response")
    model_name: str = Field(..., description="Model used for generation")
    timestamp: datetime = Field(..., description="Response timestamp")
    success: bool = Field(..., description="Whether generation was successful")
    error_message: Optional[str] = Field(None, description="Error message if generation failed")

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    model_available: bool
    version: str = "1.0.0"

class SQLAPIService:
    """Production-ready SQL generation API service"""
    
    def __init__(self, ollama_host: str = "localhost", ollama_port: int = 11434, 
                 model_name: str = "sql-expert"):
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.model_name = model_name
        self.base_url = f"http://{ollama_host}:{ollama_port}"
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="SQL Expert API",
            description="Production API for SQL query generation using fine-tuned LLM",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Request tracking
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthCheck)
        async def health_check():
            """Health check endpoint"""
            model_available = await self._check_model_availability()
            
            return HealthCheck(
                status="healthy" if model_available else "unhealthy",
                timestamp=datetime.now(),
                model_available=model_available
            )
        
        @self.app.post("/generate-sql", response_model=SQLResponse)
        async def generate_sql(request: SQLRequest, 
                             credentials: HTTPAuthorizationCredentials = Security(security)):
            """Generate SQL query from natural language description"""
            
            # Simple token validation (implement proper auth in production)
            if not self._validate_token(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid authentication token")
            
            start_time = time.time()
            
            try:
                # Build prompt
                prompt = self._build_prompt(request.instruction, request.table_schema)
                
                # Generate SQL
                sql_query = await self._generate_sql_async(
                    prompt, 
                    request.max_tokens, 
                    request.temperature
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                # Update metrics
                self.request_count += 1
                self.total_response_time += execution_time
                
                return SQLResponse(
                    sql_query=sql_query,
                    execution_time_ms=execution_time,
                    model_name=self.model_name,
                    timestamp=datetime.now(),
                    success=True
                )
                
            except Exception as e:
                self.error_count += 1
                execution_time = (time.time() - start_time) * 1000
                
                logger.error(f"SQL generation failed: {e}")
                
                return SQLResponse(
                    sql_query="",
                    execution_time_ms=execution_time,
                    model_name=self.model_name,
                    timestamp=datetime.now(),
                    success=False,
                    error_message=str(e)
                )
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get API metrics"""
            avg_response_time = (
                self.total_response_time / self.request_count 
                if self.request_count > 0 else 0
            )
            
            return {
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "success_rate": (
                    (self.request_count - self.error_count) / self.request_count * 100
                    if self.request_count > 0 else 0
                ),
                "average_response_time_ms": avg_response_time,
                "model_name": self.model_name,
                "uptime_hours": self._get_uptime_hours()
            }
        
        @self.app.get("/models")
        async def list_available_models():
            """List available models"""
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/api/tags") as response:
                        if response.status == 200:
                            data = await response.json()
                            return {"models": data.get("models", [])}
                        else:
                            raise HTTPException(status_code=502, detail="Failed to fetch models")
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Service unavailable: {e}")
    
    def _validate_token(self, token: str) -> bool:
        """Validate authentication token (implement proper validation)"""
        # In production, implement proper JWT validation or OAuth
        return token == os.getenv("API_TOKEN", "demo-token-12345")
    
    def _build_prompt(self, instruction: str, table_schema: Optional[str]) -> str:
        """Build prompt for SQL generation"""
        if table_schema:
            return f"{instruction}\n\nTable Schema: {table_schema}"
        else:
            return instruction
    
    async def _generate_sql_async(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate SQL using Ollama API asynchronously"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "").strip()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
    
    async def _check_model_availability(self) -> bool:
        """Check if model is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        return any(model.get("name", "").startswith(self.model_name) for model in models)
                    return False
        except:
            return False
    
    def _get_uptime_hours(self) -> float:
        """Get service uptime (simplified implementation)"""
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
        return (time.time() - self._start_time) / 3600
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Run the API service"""
        logger.info(f"Starting SQL Expert API on {host}:{port}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Ollama endpoint: {self.base_url}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )

# Startup script
def start_api_service():
    """Start the API service with configuration"""
    
    # Configuration from environment variables
    config = {
        "ollama_host": os.getenv("OLLAMA_HOST", "localhost"),
        "ollama_port": int(os.getenv("OLLAMA_PORT", "11434")),
        "model_name": os.getenv("MODEL_NAME", "sql-expert"),
        "api_host": os.getenv("API_HOST", "0.0.0.0"),
        "api_port": int(os.getenv("API_PORT", "8000")),
        "workers": int(os.getenv("WORKERS", "1"))
    }
    
    print(f"🚀 Starting SQL Expert API Service")
    print("=" * 50)
    print(f"Ollama endpoint: {config['ollama_host']}:{config['ollama_port']}")
    print(f"Model name: {config['model_name']}")
    print(f"API endpoint: {config['api_host']}:{config['api_port']}")
    print(f"Workers: {config['workers']}")
    
    # Initialize and run service
    service = SQLAPIService(
        ollama_host=config["ollama_host"],
        ollama_port=config["ollama_port"],
        model_name=config["model_name"]
    )
    
    service.run(
        host=config["api_host"],
        port=config["api_port"],
        workers=config["workers"]
    )

if __name__ == "__main__":
    start_api_service()
```

## Building Web Interfaces

Let's create both Streamlit and Gradio interfaces for user interaction:

```python
# streamlit_interface.py
import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class StreamlitSQLInterface:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.api_token = "demo-token-12345"  # In production, use secure token management
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="SQL Expert AI",
            page_icon="🗃️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sql-output {
            background-color: #f0f0f0;
            padding: 1rem;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            border-left: 4px solid #1f77b4;
        }
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        st.sidebar.header("⚙️ Configuration")
        
        # API Settings
        st.sidebar.subheader("API Settings")
        api_url = st.sidebar.text_input("API Base URL", value=self.api_base_url)
        api_token = st.sidebar.text_input("API Token", value=self.api_token, type="password")
        
        # Generation Parameters
        st.sidebar.subheader("Generation Parameters")
        max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 256)
        temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        
        # Health Check
        st.sidebar.subheader("Service Status")
        if st.sidebar.button("🔍 Check Health"):
            health_status = self.check_api_health(api_url, api_token)
            if health_status:
                st.sidebar.success("✅ API Service Healthy")
                st.sidebar.json(health_status)
            else:
                st.sidebar.error("❌ API Service Unavailable")
        
        return {
            "api_url": api_url,
            "api_token": api_token,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    
    def check_api_health(self, api_url: str, token: str) -> dict:
        """Check API health status"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(f"{api_url}/health", headers=headers, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            st.sidebar.error(f"Health check failed: {e}")
            return None
    
    def generate_sql(self, instruction: str, table_schema: str, config: dict) -> dict:
        """Generate SQL using the API"""
        payload = {
            "instruction": instruction,
            "table_schema": table_schema if table_schema.strip() else None,
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"]
        }
        
        headers = {"Authorization": f"Bearer {config['api_token']}"}
        
        try:
            response = requests.post(
                f"{config['api_url']}/generate-sql",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error_message": f"API Error: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Request failed: {str(e)}"
            }
    
    def render_main_interface(self, config: dict):
        """Render main SQL generation interface"""
        
        # Header
        st.markdown('<h1 class="main-header">🗃️ SQL Expert AI</h1>', unsafe_allow_html=True)
        st.markdown("*Generate accurate SQL queries from natural language descriptions*")
        
        # Quick Examples
        st.subheader("🚀 Quick Examples")
        
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            if st.button("📊 Customer Analytics", key="ex1"):
                st.session_state.instruction = "Find the top 10 customers by total spending"
                st.session_state.table_schema = "customers (id, name, email), orders (id, customer_id, amount, order_date)"
        
        with example_col2:
            if st.button("📅 Recent Activity", key="ex2"):
                st.session_state.instruction = "Get all users who registered in the last 30 days"
                st.session_state.table_schema = "users (id, username, email, registration_date)"
        
        # Main Input Form
        st.subheader("💬 Generate SQL Query")
        
        with st.form("sql_generation_form"):
            # Instruction input
            instruction = st.text_area(
                "Describe what you want to achieve:",
                value=st.session_state.get("instruction", ""),
                height=100,
                placeholder="Example: Find all customers who made purchases in the last month and spent more than $100"
            )
            
            # Table schema input
            table_schema = st.text_area(
                "Table Schema (optional):",
                value=st.session_state.get("table_schema", ""),
                height=80,
                placeholder="Example: customers (id, name, email), orders (id, customer_id, amount, date)"
            )
            
            # Submit button
            submitted = st.form_submit_button("🔮 Generate SQL", type="primary")
        
        # Generate SQL when form is submitted
        if submitted and instruction.strip():
            with st.spinner("Generating SQL query..."):
                result = self.generate_sql(instruction, table_schema, config)
            
            if result.get("success"):
                # Display results
                st.success("✅ SQL Query Generated Successfully!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Time", f"{result['execution_time_ms']:.0f}ms")
                with col2:
                    st.metric("Model", result['model_name'])
                with col3:
                    st.metric("Timestamp", result['timestamp'][:19])
                
                # SQL Output
                st.subheader("📋 Generated SQL Query")
                st.markdown(f'<div class="sql-output">{result["sql_query"]}</div>', 
                           unsafe_allow_html=True)
                
                # Copy button
                st.code(result["sql_query"], language="sql")
                
                # Save to history
                if "sql_history" not in st.session_state:
                    st.session_state.sql_history = []
                
                st.session_state.sql_history.append({
                    "instruction": instruction,
                    "schema": table_schema,
                    "sql": result["sql_query"],
                    "timestamp": result["timestamp"],
                    "execution_time": result["execution_time_ms"]
                })
                
            else:
                st.error(f"❌ Generation Failed: {result.get('error_message', 'Unknown error')}")
        
        elif submitted:
            st.warning("⚠️ Please provide an instruction for SQL generation")
    
    def render_history(self):
        """Render SQL generation history"""
        if "sql_history" in st.session_state and st.session_state.sql_history:
            st.subheader("📚 Generation History")
            
            # Convert to DataFrame for display
            df = pd.DataFrame(st.session_state.sql_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", len(df))
            with col2:
                avg_time = df['execution_time'].mean()
                st.metric("Avg Response Time", f"{avg_time:.0f}ms")
            with col3:
                recent_queries = len(df[df['timestamp'] > datetime.now() - pd.Timedelta(hours=1)])
                st.metric("Last Hour", recent_queries)
            
            # Response time chart
            if len(df) > 1:
                fig = px.line(df, x='timestamp', y='execution_time', 
                             title="Response Time Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            # History table
            display_df = df[['timestamp', 'instruction', 'execution_time']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            display_df.columns = ['Time', 'Instruction', 'Response Time (ms)']
            
            st.dataframe(display_df, use_container_width=True, height=300)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.sql_history = []
                st.rerun()
    
    def run(self):
        """Run the Streamlit interface"""
        self.setup_page()
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Main tabs
        tab1, tab2 = st.tabs(["🔮 Generate SQL", "📚 History"])
        
        with tab1:
            self.render_main_interface(config)
        
        with tab2:
            self.render_history()

# Launch Streamlit interface
if __name__ == "__main__":
    interface = StreamlitSQLInterface()
    interface.run()
```

## Docker Compose for Complete Stack

Let's create a comprehensive Docker Compose setup for the entire stack:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Ollama service for model serving
  ollama:
    image: ollama/ollama:latest
    container_name: sql-expert-ollama
    volumes:
      - ollama_data:/root/.ollama
      - ./models/gguf:/models
      - ./Modelfile:/tmp/Modelfile
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_ORIGINS=*
    command: >
      sh -c "
        ollama serve &
        sleep 30 &&
        if [ -f /tmp/Modelfile ]; then
          ollama create sql-expert -f /tmp/Modelfile
        fi &&
        wait
      "
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped

  # FastAPI service
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: sql-expert-api
    environment:
      - OLLAMA_HOST=ollama
      - OLLAMA_PORT=11434
      - MODEL_NAME=sql-expert
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - API_TOKEN=your-secure-token-here
    ports:
      - "8000:8000"
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Streamlit interface
  web:
    build:
      context: .
      dockerfile: Dockerfile.web
    container_name: sql-expert-web
    environment:
      - API_BASE_URL=http://api:8000
      - API_TOKEN=your-secure-token-here
    ports:
      - "8501:8501"
    depends_on:
      - api
    restart: unless-stopped

  # Nginx reverse proxy and load balancer
  nginx:
    image: nginx:alpine
    container_name: sql-expert-nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - api
      - web
    restart: unless-stopped

  # Redis for caching and session management
  redis:
    image: redis:alpine
    container_name: sql-expert-redis
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    command: redis-server --appendonly yes

  # Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: sql-expert-prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: sql-expert-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  ollama_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: sql-expert-network
```

## Docker Configuration Files

Let's create the necessary Dockerfiles and configuration files:

```dockerfile
# Dockerfile.api
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY api_service.py .
COPY utils/ ./utils/

# Create non-root user
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "api_service.py"]
```

```dockerfile
# Dockerfile.web
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy application code
COPY streamlit_interface.py .
COPY utils/ ./utils/

# Create non-root user
RUN useradd -m -u 1000 webuser && chown -R webuser:webuser /app
USER webuser

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit_interface.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }

    upstream web_backend {
        server web:8501;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=web_limit:10m rate=30r/s;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # API routes
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            rewrite ^/api/(.*) /$1 break;
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Web interface routes
        location / {
            limit_req zone=web_limit burst=50 nodelay;
            
            proxy_pass http://web_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

## Deployment Scripts

Let's create deployment scripts for easy setup:

```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Deploying SQL Expert LLM Stack"
echo "=================================="

# Configuration
MODEL_NAME="sql-expert"
GGUF_PATH="./models/gguf/sql-expert-q8_0.gguf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if [ -z "$(command -v docker)" ]; then
        print_error "Docker is not installed. Please install Docker Desktop."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed."
        exit 1
    fi
    
    # Check if GGUF model exists
    if [ ! -f "$GGUF_PATH" ]; then
        print_warning "GGUF model not found at $GGUF_PATH"
        print_status "Please ensure your model is converted to GGUF format"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_status "Prerequisites check completed"
}

# Create necessary directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p models/gguf
    mkdir -p logs
    mkdir -p ssl
    mkdir -p grafana/{dashboards,datasources}
    
    print_status "Directories created"
}

# Generate configuration files
generate_configs() {
    print_status "Generating configuration files..."
    
    # Generate API token
    API_TOKEN=$(openssl rand -hex 32)
    
    # Create .env file
    cat > .env << EOF
# API Configuration
API_TOKEN=${API_TOKEN}
MODEL_NAME=${MODEL_NAME}
OLLAMA_HOST=ollama
OLLAMA_PORT=11434

# Database Configuration
REDIS_URL=redis://redis:6379

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
EOF

    print_status "Configuration files generated"
    print_status "API Token: ${API_TOKEN}"
}

# Build and start services
deploy_services() {
    print_status "Building and starting services..."
    
    # Pull base images
    docker-compose pull
    
    # Build custom images
    docker-compose build
    
    # Start services
    docker-compose up -d
    
    print_status "Services started"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be healthy..."
    
    # Wait for Ollama
    print_status "Waiting for Ollama service..."
    timeout 300 bash -c 'until curl -f http://localhost:11434/api/tags &>/dev/null; do sleep 5; done'
    
    # Wait for API
    print_status "Waiting for API service..."
    timeout 120 bash -c 'until curl -f http://localhost:8000/health &>/dev/null; do sleep 5; done'
    
    # Wait for Web interface
    print_status "Waiting for Web interface..."
    timeout 60 bash -c 'until curl -f http://localhost:8501 &>/dev/null; do sleep 5; done'
    
    print_status "All services are healthy"
}

# Display deployment summary
show_summary() {
    print_status "Deployment completed successfully!"
    echo
    echo "📋 Service URLs:"
    echo "  🌐 Web Interface:  http://localhost:8501"
    echo "  🔗 API Docs:       http://localhost:8000/docs"
    echo "  📊 Grafana:        http://localhost:3000 (admin/admin123)"
    echo "  📈 Prometheus:     http://localhost:9090"
    echo
    echo "🔑 API Token: $(grep API_TOKEN .env | cut -d'=' -f2)"
    echo
    echo "📚 Useful commands:"
    echo "  docker-compose logs -f        # View logs"
    echo "  docker-compose ps             # Check status"
    echo "  docker-compose down           # Stop services"
    echo "  docker-compose restart api    # Restart API service"
}

# Main deployment flow
main() {
    check_prerequisites
    setup_directories
    generate_configs
    deploy_services
    wait_for_services
    show_summary
}

# Run deployment
main "$@"
```

## Testing the Deployment

### Viewing Logs

You can view the logs of the running services using the `docker-compose logs` command. To view the logs of all services, run the following command:

```bash
docker-compose logs -f
```

To view the logs of a specific service, you can specify the service name after the `logs` command. For example, to view the logs of the `api` service, run the following command:

```bash
docker-compose logs -f api
```

Create a comprehensive test suite for your deployment:

```python
# test_deployment.py
import requests
import time
import json
import pytest
from typing import Dict, List

class DeploymentTester:
    """Comprehensive deployment testing suite"""
    
    def __init__(self, base_url: str = "http://localhost", api_token: str = None):
        self.base_url = base_url
        self.api_token = api_token or "demo-token-12345"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
    def test_service_health(self) -> Dict[str, bool]:
        """Test health of all services"""
        
        services = {
            "nginx": f"{self.base_url}:80/health",
            "api": f"{self.base_url}:8000/health", 
            "web": f"{self.base_url}:8501",
            "ollama": f"{self.base_url}:11434/api/tags",
            "grafana": f"{self.base_url}:3000/api/health",
            "prometheus": f"{self.base_url}:9090/-/healthy"
        }
        
        health_status = {}
        
        for service, url in services.items():
            try:
                response = requests.get(url, timeout=10)
                health_status[service] = response.status_code == 200
                print(f"{'✅' if health_status[service] else '❌'} {service}: {response.status_code}")
            except Exception as e:
                health_status[service] = False
                print(f"❌ {service}: {e}")
        
        return health_status
    
    def test_api_endpoints(self) -> Dict[str, bool]:
        """Test API endpoints functionality"""
        
        test_results = {}
        
        # Test health endpoint
        try:
            response = requests.get(f"{self.base_url}:8000/health")
            test_results["health"] = response.status_code == 200
        except:
            test_results["health"] = False
        
        # Test metrics endpoint
        try:
            response = requests.get(f"{self.base_url}:8000/metrics", headers=self.headers)
            test_results["metrics"] = response.status_code == 200
        except:
            test_results["metrics"] = False
        
        # Test models endpoint
        try:
            response = requests.get(f"{self.base_url}:8000/models", headers=self.headers)
            test_results["models"] = response.status_code == 200
        except:
            test_results["models"] = False
        
        # Test SQL generation
        try:
            payload = {
                "instruction": "Find all users",
                "table_schema": "users (id, name, email)",
                "max_tokens": 128,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}:8000/generate-sql",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                test_results["sql_generation"] = result.get("success", False)
            else:
                test_results["sql_generation"] = False
                
        except Exception as e:
            test_results["sql_generation"] = False
            print(f"SQL generation test failed: {e}")
        
        return test_results
    
    def test_performance(self) -> Dict[str, float]:
        """Test API performance"""
        
        test_payload = {
            "instruction": "Select all records from users table",
            "table_schema": "users (id, name, email, created_at)",
            "max_tokens": 128,
            "temperature": 0.7
        }
        
        latencies = []
        successes = 0
        
        print("🔬 Running performance test (10 requests)...")
        
        for i in range(10):
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.base_url}:8000/generate-sql",
                    json=test_payload,
                    headers=self.headers,
                    timeout=30
                )
                
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        successes += 1
                
                print(f"  Request {i+1}: {latency:.0f}ms")
                
            except Exception as e:
                print(f"  Request {i+1}: Failed - {e}")
        
        if latencies:
            return {
                "avg_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "success_rate": successes / 10 * 100,
                "total_requests": 10
            }
        else:
            return {"error": "No successful requests"}
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive deployment test"""
        
        print("🧪 Running Comprehensive Deployment Test")
        print("=" * 50)
        
        results = {
            "timestamp": time.time(),
            "service_health": {},
            "api_functionality": {},
            "performance": {}
        }
        
        # Test service health
        print("\n1. Testing Service Health")
        print("-" * 30)
        results["service_health"] = self.test_service_health()
        
        # Test API functionality
        print("\n2. Testing API Functionality")
        print("-" * 30)
        results["api_functionality"] = self.test_api_endpoints()
        
        for endpoint, status in results["api_functionality"].items():
            print(f"{'✅' if status else '❌'} {endpoint}")
        
        # Test performance
        print("\n3. Testing Performance")
        print("-" * 30)
        results["performance"] = self.test_performance()
        
        if "error" not in results["performance"]:
            perf = results["performance"]
            print(f"✅ Average latency: {perf['avg_latency_ms']:.0f}ms")
            print(f"✅ Success rate: {perf['success_rate']:.1f}%")
        
        # Generate summary
        print("\n📊 Test Summary")
        print("=" * 30)
        
        healthy_services = sum(results["service_health"].values())
        total_services = len(results["service_health"])
        
        working_apis = sum(results["api_functionality"].values())
        total_apis = len(results["api_functionality"])
        
        print(f"Service Health: {healthy_services}/{total_services}")
        print(f"API Functionality: {working_apis}/{total_apis}")
        
        if "success_rate" in results["performance"]:
            print(f"Performance: {results['performance']['success_rate']:.1f}% success rate")
        
        # Overall status
        overall_healthy = (
            healthy_services == total_services and 
            working_apis == total_apis and
            results["performance"].get("success_rate", 0) > 80
        )
        
        print(f"\n🎯 Overall Status: {'✅ HEALTHY' if overall_healthy else '❌ ISSUES DETECTED'}")
        
        return results

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SQL Expert deployment")
    parser.add_argument("--base-url", default="http://localhost", help="Base URL for services")
    parser.add_argument("--api-token", default="demo-token-12345", help="API authentication token")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Run tests
    tester = DeploymentTester(args.base_url, args.api_token)
    results = tester.run_comprehensive_test()
    
    # Save results if requested
    if args.save_results:
        filename = f"deployment_test_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")
```

## 📁 Reference Code Repository

All deployment code and configurations are available in the GitHub repository:

**🔗 [fine-tuning-small-llms/part5-deployment](https://github.com/saptak/fine-tuning-small-llms/tree/main/part5-deployment)**

```bash
# Clone the repository and deploy
git clone https://github.com/saptak/fine-tuning-small-llms.git
cd fine-tuning-small-llms

# Deploy the complete stack
./part5-deployment/scripts/deploy.sh

# Or use Docker Compose
docker-compose up -d
```

The Part 5 directory includes:
- Complete Docker Compose stack
- FastAPI service implementation
- Streamlit web interface
- Ollama model conversion scripts
- Nginx configuration and load balancing
- Production deployment scripts

## What's Next?

Congratulations! You've successfully deployed your fine-tuned SQL expert model with a complete production stack including APIs, web interfaces, monitoring, and load balancing.

**[Part 6: Production, Monitoring, and Scaling](/2025/07/25/fine-tuning-small-llms-part6-production/)**

In our final part, you'll learn:
- Advanced monitoring and alerting
- Auto-scaling and load balancing
- Security best practices
- Performance optimization
- Maintenance and updates
- Cost optimization strategies

### Key Achievements from Part 5

✅ **Model Conversion**: Successfully converted to GGUF format for Ollama  
✅ **Production APIs**: Built FastAPI service with authentication and monitoring  
✅ **Web Interfaces**: Created Streamlit dashboard for user interaction  
✅ **Container Orchestration**: Complete Docker Compose stack  
✅ **Load Balancing**: Nginx reverse proxy with rate limiting  
✅ **Monitoring Stack**: Prometheus and Grafana integration  

## Deployment Best Practices

1. **Security First**: Always use proper authentication and HTTPS in production
2. **Monitor Everything**: Set up comprehensive monitoring from day one
3. **Plan for Scale**: Design with horizontal scaling in mind
4. **Test Thoroughly**: Implement automated testing for all components
5. **Document Well**: Maintain clear deployment and operational documentation

## Resources and References

- [Ollama Documentation](https://ollama.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Nginx Configuration Guide](https://nginx.org/en/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

*Continue to [Part 6: Production, Monitoring, and Scaling](/2025/07/25/fine-tuning-small-llms-part6-production/) to complete your production-ready deployment!*
