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

Now, import the necessary libraries:

```python
import dspy
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
```

Set up API keys using environment variables:

```python
# Load environment variables
load_dotenv()

# Set up Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
```

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

    def basic_request(self, prompt, **kwargs):
        """Send a request to the Gemini API"""
        config = genai.types.GenerateContentConfig(
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_tokens", 1024),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 0)
        )

        # Add thinking budget if specified
        if self.thinking_budget > 0:
            config.thinking_config = genai.types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt, config)
        return response.text

# Set up Gemini Flash 2.5 as the default LM in DSPy
gemini_flash = GeminiFlash(thinking_budget=0)  # Start with thinking disabled
dspy.settings.configure(lm=gemini_flash)
```

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

# Test our RAG system
rag_system = BasicRAG(num_passages=5)
response = rag_system("What are the main principles of quantum computing?")
print(response.answer)
```

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

    def _deduplicate(self, passages):
        seen = set()
        unique_passages = []

        for passage in passages:
            if passage not in seen:
                seen.add(passage)
                unique_passages.append(passage)

        return unique_passages

# Test our multi-hop RAG system
multihop_rag = MultiHopRAG(num_passages_per_hop=3, max_hops=2)
response = multihop_rag("Which Nobel Prize winner influenced the development of quantum chromodynamics?")
print(response.answer)
```

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

# Test our constrained generator
tweet_gen = TweetGenerator()
response = tweet_gen("artificial intelligence")
print(response.tweet)
```

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

# Test our JSON generator
json_gen = JSONGenerator()
schema = '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}'
response = json_gen(schema=schema, description="Generate a person object")
print(response.json_output)
```

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

# Initialize the BootstrapFewShot optimizer
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=accuracy_metric)

# Compile the optimized QA system
optimized_qa = optimizer.compile(
    program=SimpleQA(),
    trainset=examples[:8],
    valset=examples[8:10]
)

# Test the optimized system
response = optimized_qa("What is the capital of Australia?")
print(response.answer)
```

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

# Test the optimized system
response = mipro_qa("What is the capital of New Zealand?")
print(response.answer)
```

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

# Compile with fine-tuning
# Note: This requires a model that supports fine-tuning
finetuned_qa = finetune_optimizer.compile(
    program=SimpleQA(),
    trainset=training_examples[:80],
    valset=training_examples[80:100]
)

# Test the fine-tuned system
response = finetuned_qa("What is the capital of South Africa?")
print(response.answer)
```

## Leveraging Gemini's Thinking Capabilities {#gemini-thinking}

Gemini Flash 2.5's thinking capabilities can be leveraged to improve complex reasoning tasks. Let's modify our approach to use this feature:

```python
# Create a module that benefits from thinking capabilities
class ThinkingMathSolver(dspy.Module):
    def __init__(self, thinking_budget=2048):
        super().__init__()
        self.thinking_budget = thinking_budget
        self.cot = dspy.ChainOfThought(MathProblem)

    def forward(self, question):
        # Create a Gemini instance with thinking enabled
        thinking_gemini = GeminiFlash(thinking_budget=self.thinking_budget)

        # Use the thinking model for this specific task
        with dspy.context(lm=thinking_gemini):
            return self.cot(question=question)

# Test with a complex reasoning problem
thinking_module = ThinkingMathSolver(thinking_budget=2048)
response = thinking_module("A store has a 30% discount on all items. After applying the discount, a customer pays $56 for a shirt. What was the original price of the shirt?")
print(response.rationale)  # Show the reasoning process
print(response.answer)     # Show the final answer
```

You can experiment with different thinking budgets to find the right balance between reasoning depth and cost:

```python
# Compare performance with different thinking budgets
for budget in [0, 512, 1024, 2048]:
    solver = ThinkingMathSolver(thinking_budget=budget)
    response = solver("If 5 workers can build 5 tables in 5 days, how many days would it take 10 workers to build 10 tables?")
    print(f"Budget: {budget}")
    print(f"Answer: {response.answer}")
    print("-" * 50)
```

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

# Test the multi-agent system
content_system = ContentCreationSystem(retriever)
response = content_system("The impact of artificial intelligence on healthcare")
print(response.article)
```

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

# Test the decision making system
advisor = InvestmentAdvisor(retriever)
recommendation = advisor(stock_symbol="AAPL", risk_tolerance="moderate")
print(recommendation.rationale)  # The reasoning process
print(recommendation.recommendation)  # The final recommendation
```

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
