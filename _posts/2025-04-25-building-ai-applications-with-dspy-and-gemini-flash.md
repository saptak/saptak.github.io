---
author: Saptak
categories:
- ai
- programming
date: 2025-04-25
description: Learn how to build robust AI applications by combining DSPy's declarative
  programming approach with the powerful reasoning capabilities of Gemini Flash 2.5.
header_image_path: /assets/img/blog/headers/2025-04-25-building-ai-applications-with-dspy-and-gemini-flash.jpg
image_credit: Photo by Rami Al-zayat on Unsplash
layout: post
tags: dspy gemini-flash-2.5 llm ai-programming rag
thumbnail_path: /assets/img/blog/thumbnails/2025-04-25-building-ai-applications-with-dspy-and-gemini-flash.jpg
title: Building and Optimizing AI Applications with DSPy and Gemini Flash 2.5
---

# Building and Optimizing AI Applications with DSPy and Gemini Flash 2.5

In this hands-on tutorial, we'll explore how to leverage DSPy (Declarative Self-improving Python) with Google's Gemini Flash 2.5 model to build powerful, reliable AI applications. Instead of spending hours crafting and tweaking prompts, you'll learn how to program language models declaratively and optimize their performance programmatically.

## Table of Contents

1. [Introduction to DSPy and Gemini Flash 2.5](#introduction)
2. [Setting Up Your Environment](#setting-up-environment)
3. [DSPy Core Concepts](#dspy-core-concepts)
4. [Connecting DSPy to Gemini Flash 2.5](#connecting-dspy-to-gemini)
5. [Basic DSPy Modules](#basic-dspy-modules)
   - [Simple Predict Module](#simple-predict)
   - [Chain of Thought](#chain-of-thought)
   - [Program of Thought](#program-of-thought)
   - [ReAct Framework](#react-framework)
   - [TypedPredictor and TypedChainOfThought](#typed-predictor)
6. [Building a Question-Answering System](#question-answering-system)
7. [Implementing RAG Applications](#rag-applications)
   - [Basic RAG System](#basic-rag)
   - [Multi-Hop RAG](#multi-hop-rag)
   - [RAG with Structured Output](#rag-structured-output)
8. [Using DSPy Assertions](#dspy-assertions)
9. [Optimizing DSPy Programs](#optimizing-programs)
   - [BootstrapFewShot](#bootstrap-fewshot)
   - [MIPROv2](#miprov2)
   - [BootstrapFineTune](#bootstrap-finetune)
10. [Leveraging Gemini's Thinking Capabilities](#gemini-thinking)
11. [Building Complex AI Applications](#complex-applications)
    - [Multi-Agent Systems](#multi-agent)
    - [Information Extraction](#information-extraction)
    - [Decision Making Systems](#decision-making)
12. [Conclusion and Next Steps](#conclusion)

## Introduction to DSPy and Gemini Flash 2.5 {#introduction}

DSPy is a framework developed by Stanford NLP that enables developers to build AI systems using a declarative programming approach rather than tedious prompt engineering. It shifts focus from crafting perfect prompts to programming with structured, declarative modules that can be automatically optimized.

Gemini Flash 2.5 is Google's latest fast, cost-efficient thinking model with powerful reasoning capabilities. Released in April 2025, it's Google's first fully hybrid reasoning model, allowing developers to toggle thinking capabilities on or off and set thinking budgets to optimize the balance between quality, cost, and latency.

By combining these technologies, we can build AI applications that are more reliable, maintainable, and efficient than traditional approaches based on prompt engineering.

## Setting Up Your Environment {#setting-up-environment}

Let's start by setting up our development environment with the necessary dependencies:

```python
# Install required packages
!pip install dspy-ai google-generativeai
```

**Code Explanation:** 
This command installs two Python libraries:
- `dspy-ai`: The DSPy framework for programming language models
- `google-generativeai`: Google's official library for interacting with their Gemini models

The exclamation mark (`!`) at the beginning lets you run shell commands directly in a Jupyter notebook or similar environment.

Now, import the necessary libraries:

```python
import dspy
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
```

**Code Explanation:**
This section imports the necessary libraries:
- `dspy`: The main DSPy framework we'll use to program AI models
- `google.generativeai`: Abbreviated as `genai` to make it easier to reference
- `os`: A standard Python library for interacting with the operating system
- `dotenv`: A library that loads environment variables from a `.env` file
- `json`: A standard Python library for working with JSON data (a common format for structured data)

Set up API keys using environment variables:

```python
# Load environment variables
load_dotenv()

# Set up Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
```

**Code Explanation:**
This code:
1. Loads environment variables from a `.env` file in your project directory. This is a secure way to store sensitive information like API keys.
2. Retrieves the Gemini API key from these environment variables
3. Configures the Google Generative AI library with this API key so it can authenticate requests to the Gemini API

## DSPy Core Concepts {#dspy-core-concepts}

Before diving into code, let's understand the key concepts that make DSPy powerful:

### 1. Signatures

Signatures are declarative specifications that define the input and output behavior of a module. They're like function signatures in programming languages but for natural language tasks.

```python
# Example signature for a question-answering task
class QuestionAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

**Code Explanation:**
This code defines a "signature" which is like a contract for what goes in and comes out of a DSPy module:
- We create a class that inherits from `dspy.Signature`
- The docstring (`"""Answer questions with short factoid answers."""`) helps the model understand the task
- `question = dspy.InputField()` defines an input called "question"
- `answer = dspy.OutputField(...)` defines an output called "answer"
- The `desc` parameter provides additional details about the expected output format (in this case, specifying that the answer should be short)

### 2. Modules

Modules are reusable components that implement specific strategies for invoking language models. They encapsulate prompting techniques like chain-of-thought, RAG, and more.

```python
# Example module that uses a signature
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa_predictor = dspy.Predict(QuestionAnswer)

    def forward(self, question):
        return self.qa_predictor(question=question)
```

**Code Explanation:**
This code defines a custom DSPy module for question answering:
- We create a class that inherits from `dspy.Module`
- In the `__init__` method, we initialize a `dspy.Predict` object that uses our `QuestionAnswer` signature
- The `forward` method defines what happens when the module is called - it takes a question as input, passes it to the predictor, and returns the result
- This structure is similar to how neural networks are defined in frameworks like PyTorch

### 3. Optimizers

Optimizers (previously called "teleprompters") improve the prompts or weights in your program based on examples and metrics.

```python
# Example optimizer usage
optimizer = dspy.MIPROv2(metric=answer_accuracy)
optimized_program = optimizer.compile(
    program=my_program,
    trainset=examples[:80],
    valset=examples[80:100]
)
```

**Code Explanation:**
This code shows how to use an optimizer to improve a DSPy program:
- We create an optimizer (`MIPROv2`) and specify a metric to optimize for
- We call the `compile` method with our program and datasets
- The optimizer automatically improves the program by finding better prompts or weights
- It uses the training set to find improvements and the validation set to evaluate them
- The result is an optimized version of our program that performs better on the specified metric

## Connecting DSPy to Gemini Flash 2.5 {#connecting-dspy-to-gemini}

To use Gemini Flash 2.5 with DSPy, we need to create a custom DSPy module that interfaces with the Gemini API. Here's how:

```python
class GeminiFlash(dspy.LM):
    """DSPy module for Gemini Flash 2.5"""

    def __init__(self, model="gemini-2.5-flash-preview-04-17", thinking_budget=0, **kwargs):
        super().__init__()
        self.model = model
        self.thinking_budget = thinking_budget
        self.kwargs = kwargs
```

**Code Explanation:**
This code defines a new class `GeminiFlash` that inherits from `dspy.LM` (Language Model). Let's break it down:
- The class is used to create a custom DSPy module for the Gemini Flash 2.5 model
- `__init__` is a special method in Python classes that runs when you create a new instance
- `model="gemini-2.5-flash-preview-04-17"` sets a default value for the model parameter
- `thinking_budget=0` specifies how much "thinking" the model should do (more on this later)
- `**kwargs` collects any additional parameters you might want to pass
- `super().__init__()` calls the initialization method of the parent class

```python
def basic_request(self, prompt, **kwargs):
    """Send a request to the Gemini API"""
    config = genai.types.GenerateContentConfig(
        temperature=kwargs.get("temperature", 0.0),
        max_output_tokens=kwargs.get("max_tokens", 1024),
        top_p=kwargs.get("top_p", 0.95),
        top_k=kwargs.get("top_k", 0)
    )
```

**Code Explanation:**
This method handles sending requests to the Gemini API:
- `basic_request` is the method DSPy will call when it needs to send a prompt to the model
- `prompt` is the text input that will be sent to the model
- The `config` object configures how the model generates text:
  - `temperature`: Controls randomness (0.0 = most deterministic, 1.0 = most random)
  - `max_output_tokens`: Maximum length of the response
  - `top_p`: Controls diversity by only considering the top probability tokens
  - `top_k`: Similar to top_p, limits the selection to the top k tokens
- The `kwargs.get()` pattern is using any values passed in, or the default value if none is provided

```python
# Add thinking budget if specified
if self.thinking_budget > 0:
    config.thinking_config = genai.types.ThinkingConfig(
        thinking_budget=self.thinking_budget
    )

model = genai.GenerativeModel(self.model)
response = model.generate_content(prompt, config)
return response.text
```

**Code Explanation:**
This code continues the `basic_request` method:
- The first block configures Gemini's unique "thinking" capability:
  - If `thinking_budget` is greater than 0, we add a `ThinkingConfig`
  - This tells Gemini to spend time reasoning internally before responding
  - Higher budgets allow for more thorough reasoning but increase cost and latency
- Next, we create a `GenerativeModel` instance with our chosen model name
- We call `generate_content()` with our prompt and configuration
- Finally, we return just the text part of the response

```python
# Set up Gemini Flash 2.5 as the default LM in DSPy
gemini_flash = GeminiFlash(thinking_budget=0)  # Start with thinking disabled
dspy.settings.configure(lm=gemini_flash)
```

**Code Explanation:**
This code:
1. Creates an instance of our custom `GeminiFlash` class with thinking disabled
2. Configures DSPy to use this as the default language model for all modules
3. This means all DSPy modules will use Gemini Flash 2.5 unless explicitly told to use a different model

Note: As of recent reports, there have been issues with connecting Gemini models directly to DSPy, and users have tried various approaches. One alternative approach is to use the OpenAI-compatible endpoint:

```python
# Alternative approach using OpenAI compatibility layer
lm = dspy.LM(
    "openai/gemini-2.5-flash",
    api_key=api_key,
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_tokens=1000,
    temperature=0.7,
)
dspy.configure(lm=lm)
```

**Code Explanation:**
This is an alternative method that uses Google's OpenAI-compatible endpoint:
- Instead of our custom class, we use the built-in `dspy.LM` class
- We specify the model as "openai/gemini-2.5-flash" (a special format)
- We provide the special API base URL that supports OpenAI-compatible requests
- We set max_tokens (1000) and temperature (0.7) to control output length and randomness
- This approach takes advantage of Google's compatibility layer to simplify integration

## Basic DSPy Modules {#basic-dspy-modules}

DSPy provides several built-in modules for different prompting strategies. Let's explore the most important ones:

### Simple Predict Module {#simple-predict}

The most basic module is `dspy.Predict`, which simply takes a signature and generates outputs:

```python
# Define a simple signature
class Sentiment(dspy.Signature):
    """Determine the sentiment of a text."""
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="either 'positive', 'negative', or 'neutral'")

# Create a predictor
sentiment_predictor = dspy.Predict(Sentiment)

# Test it
result = sentiment_predictor(text="I absolutely loved the new movie!")
print(result.sentiment)  # Output: positive
```

**Code Explanation:**
This example creates a simple sentiment analyzer:
1. First, we define a `Sentiment` signature with a text input and sentiment output
2. The sentiment output has a description specifying the expected values (positive, negative, neutral)
3. We create a predictor using `dspy.Predict` and our signature
4. We call the predictor with a sample text ("I absolutely loved the new movie!")
5. We print the `sentiment` property of the result, which should be "positive"

Behind the scenes, DSPy:
1. Creates a prompt explaining the task using the signature's docstring and field descriptions
2. Sends the prompt and user's text to the Gemini model
3. Parses the response to extract the sentiment value
4. Returns a structured result with the sentiment field

### Chain of Thought {#chain-of-thought}

`dspy.ChainOfThought` elicits step-by-step reasoning before producing an answer, which improves accuracy for complex tasks:

```python
# Define a math problem signature
class MathProblem(dspy.Signature):
    """Solve a math word problem."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="the numerical answer")

# Create a chain-of-thought solver
math_solver = dspy.ChainOfThought(MathProblem)

# Customize the reasoning prompt (optional)
custom_rationale = dspy.OutputField(
    prefix="Reasoning: Let's solve this step by step to",
    desc="${produce the answer}. We need to..."
)
custom_math_solver = dspy.ChainOfThought(MathProblem, rationale_type=custom_rationale)

# Test it
result = math_solver(question="If a train travels at 120 km/h for 2.5 hours, then slows to 90 km/h for 1.5 hours, what is the total distance traveled?")
print(result.rationale)  # Shows the reasoning
print(result.answer)     # Shows the final answer
```

**Code Explanation:**
This code creates a math problem solver using Chain of Thought reasoning:
1. We define a `MathProblem` signature with question input and answer output
2. We create a solver using `dspy.ChainOfThought` with our signature
3. We also show how to customize the reasoning prompt by creating a custom rationale field
   - `prefix` sets the beginning of the reasoning prompt
   - `desc` provides a template for how to continue the reasoning
   - `${produce the answer}` is a template variable that will be filled in
4. We test the solver with a math problem about a train
5. We print both the reasoning process (`rationale`) and the final answer

The Chain of Thought approach is powerful because:
- It encourages the model to think step-by-step before answering
- It makes the reasoning process transparent and inspectable
- It often produces more accurate answers for complex problems
- It's similar to how humans solve difficult problems

### Program of Thought {#program-of-thought}

`dspy.ProgramOfThought` is similar to Chain of Thought but structures the reasoning as executable code:

```python
# Define signature for a computational problem
class Computation(dspy.Signature):
    """Perform a complex calculation."""
    problem = dspy.InputField()
    solution = dspy.OutputField()

# Create a program-of-thought solver
pot_solver = dspy.ProgramOfThought(Computation)

# Test it
result = pot_solver(problem="What is the sum of squares of all integers from 1 to 10?")
print(result.program)  # Shows the generated code
print(result.solution) # Shows the final solution
```

**Code Explanation:**
This code creates a computational problem solver using Program of Thought:
1. We define a `Computation` signature with problem input and solution output
2. We create a solver using `dspy.ProgramOfThought` with our signature
3. We test it with a mathematical problem (sum of squares)
4. We print both the generated program and the final solution

Program of Thought is useful because:
- It structures reasoning as executable code instead of natural language
- The code can be run to verify correctness
- It's particularly effective for mathematical or algorithmic problems
- It helps the model think more precisely through computational steps

### ReAct Framework {#react-framework}

`dspy.ReAct` implements the Reasoning + Acting framework, which allows models to interact with tools:

```python
# Define tools that ReAct can use
tools = [
    {
        "name": "search",
        "description": "Search for information on the web",
        "parameters": {"query": "string"}
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {"expression": "string"}
    }
]

# Define a signature for a task requiring tools
class ProblemSolving(dspy.Signature):
    """Solve a complex problem using tools if needed."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Create a ReAct solver with tools
react_solver = dspy.ReAct(ProblemSolving, tools=tools, max_iters=5)

# Test it
result = react_solver(question="What is the population of France divided by the population of Canada?")
print(result.answer)
```

**Code Explanation:**
This code creates a problem solver that can use tools:
1. We define a list of tools that the model can use:
   - A search tool for finding information
   - A calculator tool for performing calculations
   - Each tool has a name, description, and parameters
2. We define a `ProblemSolving` signature with question input and answer output
3. We create a solver using `dspy.ReAct` with our signature, tools, and a maximum number of iterations
4. We test it with a question that requires both searching for information and performing a calculation
5. We print the final answer

The ReAct framework is powerful because:
- It allows models to use tools to solve problems they couldn't solve on their own
- It alternates between reasoning and acting (using tools)
- It can handle multi-step tasks that require external information or computation
- It makes LLMs more capable by extending their abilities with specialized tools

### TypedPredictor and TypedChainOfThought {#typed-predictor}

For structured outputs, DSPy provides `TypedPredictor` and `TypedChainOfThought`:

```python
from pydantic import BaseModel

# Define structured input and output using Pydantic
class Person(BaseModel):
    name: str
    age: int
    email: str

class ExtractPersonInput(BaseModel):
    text: str

# Create a typed predictor
person_extractor = dspy.TypedPredictor("input:ExtractPersonInput -> output:Person")

# Test it
result = person_extractor(input={"text": "John Doe is 30 years old. You can reach him at john.doe@example.com."})
print(result)  # Output will be structured as a Person object
```

**Code Explanation:**
This code creates an information extractor with typed outputs:
1. We import `BaseModel` from Pydantic, a data validation library
2. We define two models:
   - `Person`: A structured output with name, age, and email fields
   - `ExtractPersonInput`: A simple input with a text field
3. We create a predictor using `dspy.TypedPredictor` with a string signature
   - The signature format is "input:InputType -> output:OutputType"
4. We test it with a text containing person information
5. The result will be a structured Person object with name, age, and email fields

TypedPredictor is useful because:
- It ensures outputs are properly structured and validated
- It works well with Pydantic, a popular data validation library
- It helps create more robust systems by enforcing output types
- It's particularly useful for information extraction tasks

## Building a Question-Answering System {#question-answering-system}

Now, let's build a simple question-answering system that uses Chain of Thought reasoning:

```python
# Define a signature for our QA system
class DetailedQA(dspy.Signature):
    """Answer questions with detailed explanations."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="comprehensive answer with explanation")

# Create a QA module with Chain of Thought
class EnhancedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa_predictor = dspy.ChainOfThought(DetailedQA)

    def forward(self, question):
        # Enhance question with instructions for better answers
        enhanced_question = f"Provide a detailed, accurate answer to this question: {question}"
        return self.qa_predictor(question=enhanced_question)

# Test our QA system
qa_system = EnhancedQA()
response = qa_system("What causes aurora borealis?")
print(response.rationale)  # The reasoning process
print(response.answer)     # The final answer
```

**Code Explanation:**
This code builds a question-answering system with detailed explanations:
1. We define a `DetailedQA` signature with question input and answer output
2. We create a custom `EnhancedQA` module:
   - In `__init__`, we create a `ChainOfThought` predictor using our signature
   - In `forward`, we enhance the question with additional instructions
   - We return the result from the predictor
3. We test it with a question about aurora borealis
4. We print both the reasoning process and the final answer

This approach is effective because:
- It uses Chain of Thought to encourage step-by-step reasoning
- It enhances the question with clear instructions
- It produces detailed, well-explained answers
- It makes the reasoning process transparent for inspection

## Implementing RAG Applications {#rag-applications}

Retrieval-Augmented Generation (RAG) is a powerful technique that combines retrieval of relevant information with language model generation. DSPy makes it easy to implement RAG systems.

### Basic RAG System {#basic-rag}

```python
# First, set up a retriever
from dspy.retrieve.qdrant import QdrantRetriever

# Initialize a vector database (this is a placeholder - replace with your actual DB)
retriever = QdrantRetriever(
    collection_name="my_documents",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Code Explanation:**
This code sets up a document retriever for our RAG system:
1. We import `QdrantRetriever` from DSPy, which interfaces with the Qdrant vector database
2. We initialize it with:
   - `collection_name`: The name of the document collection in the database
   - `embedding_model`: A model that converts text to numerical vectors (embeddings)

A vector database is a specialized database that stores text as numerical vectors, allowing for semantic search (finding documents with similar meaning, not just matching keywords).

```python
# Define a RAG signature
class RAG(dspy.Signature):
    """Generate answers based on retrieved documents."""
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField()

# Create a basic RAG module
class BasicRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(retriever, k=num_passages)
        self.generate = dspy.ChainOfThought(RAG)

    def forward(self, question):
        # Retrieve relevant passages
        passages = self.retrieve(question).passages

        # Concatenate passages into context
        context = "\n\n".join(passages)

        # Generate answer based on context
        response = self.generate(question=question, context=context)

        return response
```

**Code Explanation:**
This code defines our RAG system:
1. We define a `RAG` signature with:
   - `question`: The user's query
   - `context`: The retrieved documents
   - `answer`: The generated response
2. We create a `BasicRAG` module:
   - `num_passages`: How many documents to retrieve (default: 3)
   - `self.retrieve`: A retriever component that fetches documents
   - `self.generate`: A generator that creates answers based on context
3. In the `forward` method:
   - We retrieve relevant passages based on the question
   - We join them into a single context string
   - We generate an answer based on both the question and context
   - We return the response

```python
# Test our RAG system
rag_system = BasicRAG(num_passages=5)
response = rag_system("What are the main principles of quantum computing?")
print(response.answer)
```

**Code Explanation:**
This code tests our RAG system:
1. We create a `BasicRAG` instance with 5 passages per query
2. We call it with a question about quantum computing
3. We print the generated answer

The RAG approach is powerful because:
- It combines the knowledge from a document database with the reasoning abilities of an LLM
- It can answer questions about specific documents or specialized knowledge
- It provides more accurate and up-to-date information than the LLM alone
- It can cite sources for its answers, increasing trustworthiness

### Multi-Hop RAG {#multi-hop-rag}

For complex questions requiring multiple retrieval steps:

```python
# Define a multi-hop RAG module
class MultiHopRAG(dspy.Module):
    def __init__(self, num_passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought("context, question -> query") for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(retriever, k=num_passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(RAG)
        self.max_hops = max_hops
```

**Code Explanation:**
This code defines a more advanced RAG system with multiple "hops" (retrieval steps):
1. We create a `MultiHopRAG` module with:
   - `num_passages_per_hop`: How many documents to retrieve in each step
   - `max_hops`: The maximum number of retrieval steps
2. We initialize:
   - `self.generate_query`: An array of query generators, one for each hop
   - `self.retrieve`: A retriever component for fetching documents
   - `self.generate_answer`: A final answer generator
   - `self.max_hops`: Storage for the maximum number of hops

```python
def forward(self, question):
    context = []
    prev_queries = [question]

    # Perform multiple hops of retrieval
    for hop in range(self.max_hops):
        # Generate a new query based on current context and original question
        query = self.generate_query[hop](context="\n\n".join(context), question=question).query

        # Skip if query is similar to previous queries
        if query in prev_queries:
            continue

        prev_queries.append(query)

        # Retrieve passages
        passages = self.retrieve(query).passages

        # Add to context (avoiding duplicates)
        context = self._deduplicate(context + passages)

    # Generate final answer
    response = self.generate_answer(question=question, context="\n\n".join(context))

    return response
```

**Code Explanation:**
This continues the `MultiHopRAG` module with the `forward` method:
1. We initialize empty lists for context and previous queries
2. For each hop (up to `max_hops`):
   - We generate a new query based on the current context and original question
   - We skip this query if it's identical to a previous one (to avoid loops)
   - We add it to the list of previous queries
   - We retrieve passages using this query
   - We add them to the context, removing duplicates
3. After all hops, we generate a final answer using all collected context
4. We return this response

```python
def _deduplicate(self, passages):
    seen = set()
    unique_passages = []

    for passage in passages:
        if passage not in seen:
            seen.add(passage)
            unique_passages.append(passage)

    return unique_passages
```

**Code Explanation:**
This is a helper method to remove duplicate passages:
1. We create a set to track passages we've already seen
2. We initialize an empty list for unique passages
3. For each passage:
   - If we haven't seen it before, we add it to both the set and the list
4. We return the list of unique passages

Multi-hop RAG is powerful because:
- It can answer complex questions that require connecting multiple pieces of information
- It simulates a multi-step research process
- It can follow chains of reasoning across different documents
- It's particularly effective for questions like "What is the connection between X and Y?"

### RAG with Structured Output {#rag-structured-output}

For extracting structured information from retrieved documents:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

# Define structured output schema
class ResearchPaper(BaseModel):
    title: str
    authors: List[str]
    year: int
    abstract: str
    keywords: Optional[List[str]] = None

class StructuredRAGInput(BaseModel):
    question: str
    context: str
```

**Code Explanation:**
This code defines structured data models for our information extraction:
1. We import necessary modules from Pydantic and Python's typing system
2. We define a `ResearchPaper` model with:
   - `title`: The paper's title (a string)
   - `authors`: A list of author names
   - `year`: The publication year (an integer)
   - `abstract`: The paper's abstract
   - `keywords`: An optional list of keywords
3. We define a `StructuredRAGInput` model with:
   - `question`: The user's query
   - `context`: The retrieved documents

```python
# Create a structured RAG system
class StructuredRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(retriever, k=num_passages)
        self.extract = dspy.TypedPredictor("input:StructuredRAGInput -> output:ResearchPaper")

    def forward(self, question):
        # Retrieve passages
        passages = self.retrieve(question).passages
        context = "\n\n".join(passages)

        # Extract structured information
        input_data = {"question": question, "context": context}
        result = self.extract(input=input_data)

        return result
```

**Code Explanation:**
This code defines a RAG system that outputs structured data:
1. We create a `StructuredRAG` module with:
   - `num_passages`: How many documents to retrieve
2. We initialize:
   - `self.retrieve`: A retriever component
   - `self.extract`: A typed predictor that converts text to structured data
3. In the `forward` method:
   - We retrieve relevant passages
   - We join them into a context string
   - We create an input dictionary with question and context
   - We extract structured information from this input
   - We return the structured result

```python
# Test our structured RAG system
structured_rag = StructuredRAG()
response = structured_rag("Find the research paper about transformer architecture")
print(f"Title: {response.title}")
print(f"Authors: {', '.join(response.authors)}")
print(f"Year: {response.year}")
print(f"Abstract: {response.abstract}")
if response.keywords:
    print(f"Keywords: {', '.join(response.keywords)}")
```

**Code Explanation:**
This code tests our structured RAG system:
1. We create a `StructuredRAG` instance
2. We call it with a query about transformer architecture
3. We print each field of the structured response:
   - The paper title
   - The author list (joined with commas)
   - The publication year
   - The abstract
   - The keywords (if any)

Structured RAG is powerful because:
- It extracts specific information in a structured format
- It can be easily integrated with downstream systems
- It ensures outputs follow a predefined schema
- It's ideal for tasks like information extraction, data population, and building structured databases

## Using DSPy Assertions {#dspy-assertions}

DSPy Assertions allow you to add constraints to ensure outputs meet specific requirements:

```python
# Define a tweet generator with length constraints
class TweetGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("topic -> tweet")

    def forward(self, topic):
        response = self.generate(topic=topic)

        # Add assertion to check tweet length
        dspy.Assert(
            len(response.tweet) <= 280,
            "Tweet must be 280 characters or less."
        )

        return response
```

**Code Explanation:**
This code creates a tweet generator with a length constraint:
1. We define a `TweetGenerator` module with a chain-of-thought generator
2. In the `forward` method:
   - We generate a tweet based on the topic
   - We add an assertion that the tweet must be 280 characters or less
   - If the assertion fails, DSPy will retry with the error message
   - We return the final response (which satisfies the assertion)

Assertions are powerful because:
- They enforce constraints on the output
- They allow the model to self-correct when constraints are violated
- They don't require manually encoding constraints in prompts
- They make outputs more reliable and predictable

```python
# Test our constrained generator
tweet_gen = TweetGenerator()
response = tweet_gen("artificial intelligence")
print(response.tweet)
```

**Code Explanation:**
This code tests our tweet generator:
1. We create a `TweetGenerator` instance
2. We call it with the topic "artificial intelligence"
3. We print the generated tweet (which will be 280 characters or less)

For more complex assertions:

```python
# Define a function to validate JSON format
def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False

# Create a module with multiple assertions
class JSONGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("schema, description -> json_output")

    def forward(self, schema, description):
        response = self.generate(schema=schema, description=description)

        # Add assertion for valid JSON
        dspy.Assert(
            is_valid_json(response.json_output),
            "Output must be valid JSON."
        )

        # Add suggestion for JSON formatting
        dspy.Suggest(
            response.json_output.startswith("{") and response.json_output.endswith("}"),
            "JSON should be enclosed in curly braces."
        )

        return response
```

**Code Explanation:**
This code creates a JSON generator with multiple constraints:
1. We define a helper function `is_valid_json` that checks if a string is valid JSON
2. We create a `JSONGenerator` module with a prediction component
3. In the `forward` method:
   - We generate JSON based on a schema and description
   - We add a strict assertion that the output must be valid JSON
   - We add a softer suggestion that JSON should be enclosed in curly braces
   - We return the response

The difference between assertions and suggestions:
- Assertions are strict constraints that must be satisfied
- Suggestions are softer guidance that the model should try to follow
- If an assertion fails, DSPy will retry with the error message
- If a suggestion fails, DSPy may not retry

```python
# Test our JSON generator
json_gen = JSONGenerator()
schema = '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}'
response = json_gen(schema=schema, description="Generate a person object")
print(response.json_output)
```

**Code Explanation:**
This code tests our JSON generator:
1. We create a `JSONGenerator` instance
2. We call it with:
   - A JSON schema defining a person object with name and age
   - A description of what to generate
3. We print the generated JSON (which will be valid according to our assertions)

## Optimizing DSPy Programs {#optimizing-programs}

One of DSPy's most powerful features is its ability to optimize prompts through examples and metrics:

### BootstrapFewShot {#bootstrap-fewshot}

```python
# Define a simple accuracy metric
def accuracy_metric(example, prediction):
    return 1.0 if example.answer.lower() == prediction.answer.lower() else 0.0

# Create example data
examples = [
    dspy.Example(question="What is the capital of France?", answer="Paris"),
    dspy.Example(question="What is the capital of Japan?", answer="Tokyo"),
    dspy.Example(question="What is the capital of Brazil?", answer="Brasília"),
    # Add more examples...
]
```

**Code Explanation:**
This code prepares for optimization:
1. We define an `accuracy_metric` function that:
   - Takes an example (with the correct answer) and a prediction
   - Returns 1.0 if the answers match (ignoring case), 0.0 otherwise
2. We create a list of example questions and answers
   - Each `dspy.Example` contains both a question and its correct answer
   - These will be used to teach the model the pattern

```python
# Initialize the BootstrapFewShot optimizer
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=accuracy_metric)

# Compile the optimized QA system
optimized_qa = optimizer.compile(
    program=SimpleQA(),
    trainset=examples[:8],
    valset=examples[8:10]
)
```

**Code Explanation:**
This code performs the optimization:
1. We import `BootstrapFewShot` from DSPy's teleprompt module
2. We create an optimizer with our accuracy metric
3. We compile an optimized version of our `SimpleQA` program:
   - We use the first 8 examples as a training set
   - We use the last 2 examples as a validation set
   - The optimizer finds good examples to include in prompts
   - It returns an optimized version of our program

The BootstrapFewShot process:
1. Runs the original program on training examples
2. Collects successful examples (where the program produced correct answers)
3. Uses these examples as few-shot demonstrations in prompts
4. Evaluates the improved program on the validation set
5. Returns the optimized program with better performance

```python
# Test the optimized system
response = optimized_qa("What is the capital of Australia?")
print(response.answer)
```

**Code Explanation:**
This code tests the optimized system:
1. We call the optimized QA system with a new question
2. We print the generated answer
3. The answer should be more accurate now that the system has been optimized

### MIPROv2 {#miprov2}

```python
# Initialize the MIPROv2 optimizer
from dspy.teleprompt import MIPROv2

mipro_optimizer = MIPROv2(metric=accuracy_metric)

# Compile with MIPROv2 for more sophisticated optimization
mipro_qa = mipro_optimizer.compile(
    program=SimpleQA(),
    trainset=examples[:8],
    valset=examples[8:10],
    max_rounds=3
)
```

**Code Explanation:**
This code uses a more advanced optimizer:
1. We import `MIPROv2` from DSPy's teleprompt module
2. We create an optimizer with our accuracy metric
3. We compile an optimized version of our `SimpleQA` program:
   - We use the same training and validation sets as before
   - We specify `max_rounds=3` to run three rounds of optimization
   - The optimizer finds better instructions and demonstrations
   - It returns a highly optimized version of our program

MIPROv2 (Multi-step Instruction Prompting Optimization) is more sophisticated:
1. It bootstraps examples like BootstrapFewShot
2. It also generates and tests different instructions for the model
3. It explores combinations of instructions and examples
4. It iteratively improves the program over multiple rounds
5. It often achieves better performance than simpler optimizers

```python
# Test the optimized system
response = mipro_qa("What is the capital of New Zealand?")
print(response.answer)
```

**Code Explanation:**
This code tests the MIPROv2-optimized system:
1. We call the optimized QA system with a new question
2. We print the generated answer
3. The answer should be even more accurate with this advanced optimization

### BootstrapFineTune {#bootstrap-finetune}

```python
# For smaller models that can be fine-tuned
from dspy.teleprompt import BootstrapFineTune

# Create a larger training set
training_examples = [
    # Add many examples here
]

# Initialize the BootstrapFineTune optimizer
finetune_optimizer = BootstrapFineTune(
    metric=accuracy_metric,
    max_bootstrapped_demos=100
)
```

**Code Explanation:**
This code prepares for fine-tuning optimization:
1. We import `BootstrapFineTune` from DSPy's teleprompt module
2. We would create a larger training set with many examples
3. We create an optimizer with:
   - Our accuracy metric
   - A maximum of 100 bootstrapped demonstrations

The BootstrapFineTune approach:
- Goes beyond prompt optimization to actual model weight updates
- Generates examples automatically using bootstrapping
- Uses these examples to fine-tune the model weights
- Is particularly useful for smaller, locally-hosted models

```python
# Compile with fine-tuning
# Note: This requires a model that supports fine-tuning
finetuned_qa = finetune_optimizer.compile(
    program=SimpleQA(),
    trainset=training_examples[:80],
    valset=training_examples[80:100]
)
```

**Code Explanation:**
This code performs the fine-tuning optimization:
1. We compile a fine-tuned version of our `SimpleQA` program:
   - We use the first 80 examples as a training set
   - We use the last 20 examples as a validation set
   - The optimizer bootstraps additional examples
   - It uses these to fine-tune the model weights
   - It returns an improved program with a fine-tuned model

Note: Fine-tuning requires a model that supports it. Large hosted models like Gemini may not allow fine-tuning, but smaller local models often do.

```python
# Test the fine-tuned system
response = finetuned_qa("What is the capital of South Africa?")
print(response.answer)
```

**Code Explanation:**
This code tests the fine-tuned system:
1. We call the fine-tuned QA system with a new question
2. We print the generated answer
3. The answer should be accurate thanks to the fine-tuned weights

## Leveraging Gemini's Thinking Capabilities {#gemini-thinking}

Gemini Flash 2.5's thinking capabilities can be leveraged to improve complex reasoning tasks. Let's modify our approach to use this feature:

```python
# Create a module that benefits from thinking capabilities
class ThinkingMathSolver(dspy.Module):
    def __init__(self, thinking_budget=2048):
        super().__init__()
        self.thinking_budget = thinking_budget
        self.cot = dspy.ChainOfThought(MathProblem)
```

**Code Explanation:**
This code creates a math solver that uses Gemini's thinking capabilities:
1. We define a `ThinkingMathSolver` module with:
   - A thinking budget parameter (default: 2048)
   - A Chain of Thought component for solving math problems
2. The thinking budget controls how much internal reasoning Gemini will do

```python
def forward(self, question):
    # Create a Gemini instance with thinking enabled
    thinking_gemini = GeminiFlash(thinking_budget=self.thinking_budget)

    # Use the thinking model for this specific task
    with dspy.context(lm=thinking_gemini):
        return self.cot(question=question)
```

**Code Explanation:**
This continues the `ThinkingMathSolver` with the `forward` method:
1. We create a new `GeminiFlash` instance with thinking enabled
2. We use `dspy.context()` to temporarily override the default language model
3. Within this context, we run the Chain of Thought solver
4. This allows us to use thinking for just this specific task

This approach gives you fine control over when to use Gemini's thinking capabilities, allowing you to use it only for complex problems where it's most valuable.

```python
# Test with a complex reasoning problem
thinking_module = ThinkingMathSolver(thinking_budget=2048)
response = thinking_module("A store has a 30% discount on all items. After applying the discount, a customer pays $56 for a shirt. What was the original price of the shirt?")
print(response.rationale)  # Show the reasoning process
print(response.answer)     # Show the final answer
```

**Code Explanation:**
This code tests our thinking-enabled solver:
1. We create a `ThinkingMathSolver` with a budget of 2048 tokens
2. We call it with a discount calculation problem
3. We print both the reasoning process and the final answer

Gemini's thinking capabilities:
- Allow the model to reason internally before responding
- Improve accuracy on complex reasoning tasks
- Can be controlled with a thinking budget
- Have a cost-quality tradeoff (higher budgets cost more)

```python
# Compare performance with different thinking budgets
for budget in [0, 512, 1024, 2048]:
    solver = ThinkingMathSolver(thinking_budget=budget)
    response = solver("If 5 workers can build 5 tables in 5 days, how many days would it take 10 workers to build 10 tables?")
    print(f"Budget: {budget}")
    print(f"Answer: {response.answer}")
    print("-" * 50)
```

**Code Explanation:**
This code experiments with different thinking budgets:
1. We loop through four different budget levels (0, 512, 1024, 2048)
2. For each, we create a solver with that budget
3. We test it on a worker/table problem
4. We print the budget and answer
5. This helps find the right balance between quality and cost

## Building Complex AI Applications {#complex-applications}

DSPy's modular approach allows you to build sophisticated AI applications by combining different modules.

### Multi-Agent Systems {#multi-agent}

```python
# Define specialized agents
class ResearchAgent(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retrieve = dspy.Retrieve(retriever, k=5)
        self.synthesize = dspy.ChainOfThought("question, context -> findings")

    def forward(self, question):
        passages = self.retrieve(question).passages
        context = "\n\n".join(passages)
        return self.synthesize(question=question, context=context)
```

**Code Explanation:**
This code defines a research agent:
1. We create a `ResearchAgent` module that:
   - Takes a retriever in its constructor
   - Retrieves 5 passages for each question
   - Uses Chain of Thought to synthesize findings from the context
2. In the `forward` method:
   - We retrieve relevant passages
   - We join them into a context
   - We synthesize findings from the question and context
   - We return the results

```python
class WritingAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("topic, outline -> article")

    def forward(self, topic, outline):
        return self.generate(topic=topic, outline=outline)

class OutlineAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("topic, research -> outline")

    def forward(self, topic, research):
        return self.generate(topic=topic, research=research)
```

**Code Explanation:**
This code defines two more specialized agents:
1. A `WritingAgent` that:
   - Takes a topic and outline
   - Generates a full article
2. An `OutlineAgent` that:
   - Takes a topic and research findings
   - Generates an outline for an article

Each agent focuses on a specific task and can be composed with others.

```python
# Create a multi-agent system
class ContentCreationSystem(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.research_agent = ResearchAgent(retriever)
        self.outline_agent = OutlineAgent()
        self.writing_agent = WritingAgent()

    def forward(self, topic):
        # Step 1: Research the topic
        research_results = self.research_agent(topic)

        # Step 2: Create an outline
        outline = self.outline_agent(topic=topic, research=research_results.findings)

        # Step 3: Write the article
        article = self.writing_agent(topic=topic, outline=outline.outline)

        return article
```

**Code Explanation:**
This code creates a multi-agent system for content creation:
1. We define a `ContentCreationSystem` that:
   - Takes a retriever in its constructor
   - Creates three specialized agents
2. In the `forward` method, we implement a workflow:
   - Step 1: Research the topic using the research agent
   - Step 2: Create an outline using the outline agent and research results
   - Step 3: Write an article using the writing agent and outline
   - We return the final article

This approach is powerful because:
- It breaks down a complex task into manageable steps
- Each agent specializes in a specific sub-task
- The agents work together in a pipeline
- It mimics how humans collaborate on complex projects

```python
# Test the multi-agent system
content_system = ContentCreationSystem(retriever)
response = content_system("The impact of artificial intelligence on healthcare")
print(response.article)
```

**Code Explanation:**
This code tests our multi-agent system:
1. We create a `ContentCreationSystem` with our retriever
2. We call it with a topic about AI in healthcare
3. We print the generated article

### Information Extraction {#information-extraction}

```python
from pydantic import BaseModel
from typing import List, Dict, Optional

# Define structured output schema
class Company(BaseModel):
    name: str
    industry: str
    founded_year: Optional[int]
    headquarters: Optional[str]
    key_products: Optional[List[str]]
    revenue: Optional[str]
```

**Code Explanation:**
This code defines a structured data model for company information:
1. We import necessary modules from Pydantic and Python's typing system
2. We define a `Company` model with:
   - Required fields: name, industry
   - Optional fields: founded_year, headquarters, key_products, revenue
   - Appropriate types for each field

```python
# Define an information extraction system
class CompanyProfileExtractor(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retrieve = dspy.Retrieve(retriever, k=10)
        self.extract = dspy.TypedChainOfThought("input:StructuredRAGInput -> output:Company")

    def forward(self, company_name):
        # Retrieve information about the company
        passages = self.retrieve(f"information about {company_name}").passages
        context = "\n\n".join(passages)

        # Extract structured information
        input_data = {"question": f"Extract information about {company_name}", "context": context}
        profile = self.extract(input=input_data)

        return profile
```

**Code Explanation:**
This code creates a system for extracting company profiles:
1. We define a `CompanyProfileExtractor` module that:
   - Takes a retriever
   - Retrieves 10 passages about the company
   - Uses TypedChainOfThought to extract structured information
2. In the `forward` method:
   - We retrieve passages about the company
   - We join them into a context
   - We create an input dictionary with question and context
   - We extract a structured company profile
   - We return the profile

This approach is useful for:
- Automatically populating databases with structured information
- Creating company profiles from unstructured text
- Extracting specific fields with proper typing
- Ensuring data consistency across extractions

```python
# Test the information extraction system
extractor = CompanyProfileExtractor(retriever)
company_profile = extractor("Tesla")
print(f"Company: {company_profile.name}")
print(f"Industry: {company_profile.industry}")
print(f"Founded: {company_profile.founded_year}")
print(f"Headquarters: {company_profile.headquarters}")
if company_profile.key_products:
    print(f"Key Products: {', '.join(company_profile.key_products)}")
print(f"Revenue: {company_profile.revenue}")
```

**Code Explanation:**
This code tests our information extraction system:
1. We create a `CompanyProfileExtractor` with our retriever
2. We call it with "Tesla" as the company name
3. We print each field of the extracted profile:
   - Company name
   - Industry
   - Founded year
   - Headquarters
   - Key products (if available)
   - Revenue

### Decision Making Systems {#decision-making}

```python
# Define a decision making system
class InvestmentAdvisor(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retrieve = dspy.Retrieve(retriever, k=7)
        self.analyze = dspy.ChainOfThought("stock, context -> analysis")
        self.decide = dspy.ChainOfThought("analysis, risk_tolerance -> recommendation")

    def forward(self, stock_symbol, risk_tolerance):
        # Retrieve information about the stock
        passages = self.retrieve(f"{stock_symbol} stock analysis financial data").passages
        context = "\n\n".join(passages)

        # Analyze the stock
        analysis = self.analyze(stock=stock_symbol, context=context)

        # Make a recommendation
        recommendation = self.decide(
            analysis=analysis.analysis,
            risk_tolerance=risk_tolerance
        )

        return recommendation
```

**Code Explanation:**
This code creates an investment advisor system:
1. We define an `InvestmentAdvisor` module that:
   - Takes a retriever
   - Retrieves 7 passages about the stock
   - Has two Chain of Thought components:
     - One for analysis
     - One for decision making
2. In the `forward` method:
   - We retrieve passages about the stock
   - We join them into a context
   - We analyze the stock based on the context
   - We make a recommendation based on the analysis and risk tolerance
   - We return the recommendation

This multi-step approach:
- Separates analysis from decision making
- Takes risk tolerance into account
- Uses retrieved information for up-to-date analysis
- Provides transparent reasoning for recommendations

```python
# Test the decision making system
advisor = InvestmentAdvisor(retriever)
recommendation = advisor(stock_symbol="AAPL", risk_tolerance="moderate")
print(recommendation.rationale)  # The reasoning process
print(recommendation.recommendation)  # The final recommendation
```

**Code Explanation:**
This code tests our investment advisor:
1. We create an `InvestmentAdvisor` with our retriever
2. We call it with:
   - "AAPL" as the stock symbol
   - "moderate" as the risk tolerance
3. We print both the reasoning process and the final recommendation

## Conclusion and Next Steps {#conclusion}

In this comprehensive tutorial, we've explored how to combine DSPy's declarative programming approach with Gemini Flash 2.5's thinking capabilities to build powerful AI applications. We've covered:

1. Setting up DSPy with Gemini Flash 2.5
2. Core DSPy concepts: Signatures, Modules, and Optimizers
3. Various DSPy modules like Predict, ChainOfThought, and ReAct
4. Building RAG systems with single and multi-hop reasoning
5. Using DSPy Assertions to enforce constraints
6. Optimizing DSPy programs with different optimizers
7. Leveraging Gemini's thinking feature for complex reasoning
8. Building complex applications like multi-agent systems

This combination provides a powerful framework for building AI applications that are more reliable, maintainable, and efficient than traditional approaches based on prompt engineering.

### Next Steps

- Explore more advanced DSPy modules and optimizers
- Experiment with different thinking budgets in Gemini Flash 2.5
- Integrate with other data sources and vector databases
- Build domain-specific applications for your use cases
- Contribute to the DSPy community to help improve the framework

By programming language models rather than prompting them, you can create more robust and predictable AI systems that improve over time.

## Resources

- [DSPy Documentation](https://dspy.ai/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Stanford NLP GitHub Repository](https://github.com/stanfordnlp/dspy)

Happy coding with DSPy and Gemini Flash 2.5!
