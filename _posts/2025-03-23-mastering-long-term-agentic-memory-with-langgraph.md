---
categories:
- AI
- Development
- LangGraph
date: 2025-03-23 07:09:33 -0700
description: A deep dive into why, how, and when to use different types of long-term
  memory in AI agents built with Langgraph.
header_image_path: /assets/img/blog/headers/2025-03-23-mastering-long-term-agentic-memory-with-langgraph.jpg
image_credit: Photo by Maximalfocus on Unsplash
layout: post
tags:
- langgraph
- AI agents
- memory systems
- langchain
thumbnail_path: /assets/img/blog/thumbnails/2025-03-23-mastering-long-term-agentic-memory-with-langgraph.jpg
title: Long-Term Agentic Memory with LangGraph
---

# Long-Term Agentic Memory with LangGraph

Imagine having a personal assistant who forgets your preferences, past conversations, and previous instructions each time you interact with them. Not very helpful, right? This is precisely the challenge that long-term memory in AI agents aims to solve. In this comprehensive guide, we'll explore how to implement effective long-term memory in LangGraph-powered agents, focusing on the three primary types of memory: semantic, episodic, and procedural.

```mermaid
flowchart TD
    A[User Input] --> B[Agent with Long-Term Memory]
    B --> C[Response]
    
    subgraph "Memory Systems"
        D[Semantic Memory<br>Facts & Knowledge]
        E[Episodic Memory<br>Past Experiences]
        F[Procedural Memory<br>Task Knowledge]
    end
    
    B <--> D
    B <--> E
    B <--> F
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:1px
    style E fill:#bfb,stroke:#333,stroke-width:1px
    style F fill:#ffb,stroke:#333,stroke-width:1px
```

## Why Long-Term Memory Matters in Agents

AI agents without memory are like goldfish—they forget everything between conversations. This limitation fundamentally restricts what they can accomplish and how helpful they can be. Memory is what enables agents to:

- Maintain context across multiple interactions
- Adapt to user preferences over time
- Learn from past experiences
- Provide personalized responses
- Build upon previous conversations

LangGraph, a powerful framework for building agentic workflows, provides robust support for implementing various memory types that can dramatically enhance your agent's capabilities. Let's dive into understanding these memory types and how to implement them effectively.

## Understanding the Three Types of Memory

Human memory systems have been a source of inspiration for implementing memory in AI agents. The three primary types we'll focus on are:

```mermaid
classDiagram
    class LongTermMemory {
        +retrieve()
        +store()
    }
    class SemanticMemory {
        +facts: dict
        +storeFact(subject, predicate, object)
        +retrieveRelevantFacts(query)
    }
    class EpisodicMemory {
        +episodes: list
        +storeEpisode(observation, thoughts, action, result)
        +retrieveSimilarEpisodes(query)
    }
    class ProceduralMemory {
        +procedures: dict
        +storeSystemPrompt(name, prompt)
        +retrieveSystemPrompt(name)
        +updateSystemPrompt(name, feedback)
    }
    
    LongTermMemory <|-- SemanticMemory
    LongTermMemory <|-- EpisodicMemory
    LongTermMemory <|-- ProceduralMemory
```

### 1. Semantic Memory: Storing Facts and Knowledge

Semantic memory deals with factual knowledge and general information about the world or about specific users. It's the "what" of memory—storing facts that don't necessarily have a temporal context.

**In humans:** Semantic memory includes facts learned in school, general knowledge about the world, and factual information without specific episodic context.

**In AI agents:** Semantic memory typically stores facts about users, domain knowledge, preferences, and other information that should persist across interactions. This type of memory helps personalize responses and maintain user-specific context.

### 2. Episodic Memory: Recalling Experiences and Events

Episodic memory involves storing and recalling specific past events or experiences. It's the "when" and "where" of memory, capturing the context of interactions.

**In humans:** Episodic memory includes autobiographical events like remembering your first day at school or what you did last weekend.

**In AI agents:** Episodic memory often takes the form of few-shot examples derived from past successful interactions. These examples help the agent learn how to handle similar situations in the future. It answers not just "what" but "how" something was solved.

### 3. Procedural Memory: Remembering How to Perform Tasks

Procedural memory relates to knowing how to perform specific tasks or follow certain procedures. It's the "how" of memory.

**In humans:** Procedural memory allows us to perform tasks like riding a bicycle or typing without consciously thinking about each step.

**In AI agents:** Procedural memory is typically implemented through system prompts that define the agent's behavior and operating procedures. This can evolve based on feedback and experience, allowing the agent to improve its instruction-following capabilities over time.

## Implementation Patterns for Long-Term Memory

There are two primary patterns for implementing memory updates in LangGraph:

```mermaid
flowchart TB
    subgraph "Hot Path (Conscious Formation)"
        A1[User Input] --> B1[Agent Processing]
        B1 --> C1[Store in Memory<br>During Interaction]
        C1 --> D1[Response to User]
    end
    
    subgraph "Background (Subconscious Formation)"
        A2[User Input] --> B2[Agent Processing]
        B2 --> D2[Response to User]
        
        D2 -.-> E[Conversation End/Idle]
        E -.-> F[Background Memory<br>Formation Process]
        F -.-> G[Update Memory Store]
    end
    
    style C1 fill:#f96,stroke:#333,stroke-width:2px
    style F fill:#f96,stroke:#333,stroke-width:2px
```

### 1. Hot Path (Conscious Formation)

In this pattern, memory updates happen in real-time during the conversation. The agent actively decides what information to store in memory as part of its interaction flow.

**Advantages:**
- Immediate updates
- Agent has direct control over memory
- Transparent to users

**Disadvantages:**
- Adds latency to user interactions
- Increases complexity of agent's decision-making
- May reduce quality of memory creation due to multitasking

### 2. Background (Subconscious Formation)

This pattern involves updating memory asynchronously after a conversation has concluded or been inactive for some time. The agent can reflect on the interaction without adding latency to the user experience.

**Advantages:**
- No added latency during conversations
- Can process larger amounts of information
- Often produces higher-quality memories through focused reflection

**Disadvantages:**
- Updates aren't immediately available
- Requires additional background processing resources

## Implementing Semantic Memory in LangGraph

Semantic memory is often the most straightforward type to implement and provides immediate value to agents. Here's how to implement it using LangGraph and LangMem:

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant MemoryManager
    participant MemoryStore
    
    User->>Agent: Request with context
    Agent->>MemoryStore: Search for relevant facts
    MemoryStore-->>Agent: Return relevant facts
    Agent->>Agent: Process request with facts
    Agent->>User: Response
    Agent->>MemoryManager: Extract facts from conversation
    MemoryManager->>MemoryStore: Store new facts
```

```python
from langmem import create_memory_manager
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

# Define a schema for the facts we want to extract
class Fact(BaseModel):
    """A simple fact about a user or context."""
    subject: str = Field(..., description="The entity this fact is about")
    predicate: str = Field(..., description="The relationship or attribute")
    object: str = Field(..., description="The value or related entity")

# Set up a memory store
store = InMemoryStore(
    index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
)

# Create a memory manager to extract facts
memory_manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Fact],
    instructions="Extract important facts about the user from the conversation. Focus on preferences, background information, and details that would be useful in future interactions.",
    enable_inserts=True,
)
```

To use this memory in an agent context:

```python
from langchain_core.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool

# Create tools to let the agent manage memory
manage_memory_tool = create_manage_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)

search_memory_tool = create_search_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)

# Add these tools to your agent
tools = [
    # ... other tools ...
    manage_memory_tool,
    search_memory_tool
]
```

## Implementing Episodic Memory in LangGraph

Episodic memory often takes the form of few-shot examples that help the agent learn from past interactions. Here's an implementation:

```mermaid
flowchart LR
    A[User Interaction] --> B{Successful?}
    B -->|Yes| C[Extract as Episode]
    C --> D[Store in Episodic Memory]
    
    E[New Similar Task] --> F[Search Episodic Memory]
    F --> G[Retrieve Relevant Episodes]
    G --> H[Add as Few-Shot Examples]
    H --> I[Improved Response]
    
    style C fill:#bfb,stroke:#333,stroke-width:1px
    style D fill:#bfb,stroke:#333,stroke-width:1px
    style G fill:#bfb,stroke:#333,stroke-width:1px
    style H fill:#bfb,stroke:#333,stroke-width:1px
```

```python
from langmem import create_memory_manager
from pydantic import BaseModel, Field

class Episode(BaseModel):
    """An episode representing a successful interaction pattern."""
    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(..., description="Internal reasoning process that led to success")
    action: str = Field(..., description="What was done and how it was formatted")
    result: str = Field(..., description="Outcome and what made it successful")

# Create episodic memory manager
episodic_manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Episode],
    instructions="Extract examples of successful interactions, capturing the full chain of reasoning. Focus on interactions where the approach was particularly effective.",
    enable_inserts=True,
)
```

To use episodic memory in an agent:

```python
# In a triage function, for example
def triage_router(state, config, store):
    # ... other code ...
    
    # Get relevant episodic examples
    examples = store.search(
        ("email_assistant", config['configurable']['langgraph_user_id'], "examples"),
        query=str({"email": state['email_input']})
    )
    examples_formatted = format_few_shot_examples(examples)
    
    # Include these examples in the prompt
    system_prompt = prompt_template.format(
        # ... other fields ...
        examples=examples_formatted
    )
    
    # ... rest of function ...
```

## Implementing Procedural Memory in LangGraph

Procedural memory typically involves updating the system prompt based on feedback and experience:

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant PromptOptimizer
    participant PromptStore
    
    User->>Agent: Provides feedback on interaction
    Agent->>PromptOptimizer: Send conversation and feedback
    PromptOptimizer->>PromptStore: Retrieve current system prompts
    PromptStore-->>PromptOptimizer: Return current prompts
    PromptOptimizer->>PromptOptimizer: Analyze and improve prompts
    PromptOptimizer->>PromptStore: Store updated prompts
    
    Note over PromptOptimizer,PromptStore: Procedural memory update
    
    User->>Agent: New interaction
    Agent->>PromptStore: Retrieve latest prompts
    PromptStore-->>Agent: Return improved prompts
    Agent->>User: Better response using updated procedures
```

```python
from langmem import create_multi_prompt_optimizer

# Define current prompts
prompts = [
    {
        "name": "main_agent",
        "prompt": store.get(("user_id",), "agent_instructions").value['prompt'],
        "update_instructions": "Keep the instructions concise and focused on key behaviors",
        "when_to_update": "Update this prompt when there is feedback on how the agent should write or schedule"
    },
    # ... other prompts ...
]

# Define user feedback
conversations = [
    (response['messages'], "Always sign your emails 'Best regards, Assistant'")
]

# Create optimizer to update prompts
optimizer = create_multi_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="prompt_memory",
)

# Update prompts based on feedback
updated = optimizer.invoke(
    {"trajectories": conversations, "prompts": prompts}
)

# Store updated prompts
for i, updated_prompt in enumerate(updated):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        store.put(
            ("user_id",),
            f"{name}_instructions",
            {"prompt": updated_prompt['prompt']}
        )
```

## Building a Complete Email Agent with All Three Memory Types

Let's put it all together to create an email assistant with comprehensive memory capabilities:

```mermaid
graph TD
    subgraph "Email Agent Architecture"
        A[START] --> B[Triage Router]
        B -- "Email Request" --> C[Response Agent]
        C -- "Complete" --> D[END]
        
        subgraph "Memory Systems"
            E[Semantic Memory<br>User Facts]
            F[Episodic Memory<br>Previous Email Patterns]
            G[Procedural Memory<br>Email Writing Procedures]
        end
        
        B <--> F
        C <--> E
        C <--> G
        
        H[Tool: Write Email]
        I[Tool: Check Calendar]
        J[Tool: Manage Memory]
        K[Tool: Search Memory]
        
        C --> H
        C --> I
        C --> J
        C --> K
    end
    
    style E fill:#bbf,stroke:#333,stroke-width:1px
    style F fill:#bfb,stroke:#333,stroke-width:1px
    style G fill:#ffb,stroke:#333,stroke-width:1px
```

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool
from typing import TypedDict, Literal, Annotated
from langgraph.graph import add_messages

# Define state
class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

# Create memory store
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# Define tools
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

# Create memory tools
manage_memory_tool = create_manage_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)

search_memory_tool = create_search_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)

tools = [
    write_email,
    check_calendar_availability,
    manage_memory_tool,
    search_memory_tool
]

# Define triage function with episodic memory
def triage_router(state, config, store):
    # ... implementation that uses episodic memory for examples ...
    pass

# Create agent with procedural memory from system prompt
def create_prompt(state, config, store):
    # Get customized instructions from procedural memory
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )
    result = store.get(namespace, "agent_instructions")
    if result is None:
        prompt = default_instructions
    else:
        prompt = result.value['prompt']
    
    return [
        {
            "role": "system",
            "content": system_prompt_template.format(instructions=prompt)
        }
    ] + state['messages']

# Create agent
response_agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=tools,
    prompt=create_prompt,
    store=store
)

# Build graph
email_agent = StateGraph(State)
email_agent.add_node(triage_router)
email_agent.add_node("response_agent", response_agent)
email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile(store=store)
```

## Best Practices for Long-Term Memory in LangGraph

Based on real-world implementations and the latest research, here are some best practices for implementing memory in your agents:

1. **Separate memory types by function**: Keep semantic, episodic, and procedural memory separate to maintain clean boundaries and specific purposes.

2. **Implement proper namespacing**: Use namespaces to organize memories by user, application, or context to prevent cross-contamination.

3. **Balance memory formation**: Consider both hot path and background memory updates depending on your application's latency requirements.

4. **Implement privacy controls**: Ensure memories are properly scoped to respect user privacy and prevent information leakage.

5. **Use semantic search when appropriate**: Configure your memory store with embedding capabilities to enable more relevant memory retrieval.

6. **Handle memory consolidation**: Implement strategies to reconcile new information with existing memories to maintain consistency.

7. **Consider memory expiration**: Implement TTL (time-to-live) for memories that may become outdated or irrelevant over time.

8. **Add human-in-the-loop capability**: Allow users to review, correct, or delete memories to ensure accuracy and build trust.

**LangGraph Memory Store Types**

| Store Type | Description | Use Cases | Persistence |
|------------|-------------|-----------|-------------|
| InMemoryStore | In-memory storage for development and testing | Quick prototyping, non-critical data | Non-persistent |
| Vector Database | External database for semantic search using vector embeddings | Scalable knowledge retrieval, semantic search | Persistent |
| Potential Custom Stores | Developer-defined storage solutions tailored to specific application needs | Specialized data storage or retrieval needs | Depends on implementation |

## Conclusion

Long-term memory transforms AI agents from simple responders to truly helpful assistants that learn, adapt, and personalize over time. By implementing semantic, episodic, and procedural memory in your LangGraph-based agents, you can create experiences that feel more human, more helpful, and more intelligent.

The combination of these memory types mirrors human cognitive processes, allowing agents to remember facts, learn from experiences, and refine their behaviors over time. As you implement these capabilities, you'll find your agents becoming increasingly valuable to users as they build up knowledge and adapt to specific needs and preferences.

LangGraph and the supporting ecosystem of tools like LangMem provide a powerful foundation for building these capabilities, with flexible abstractions that can fit a wide variety of use cases. Whether you're building a personal assistant, a customer service agent, or a specialized tool, investing in robust memory capabilities will pay dividends in user satisfaction and agent effectiveness.

## References

1. LangChain Documentation: [Memory for Agents](https://blog.langchain.dev/memory-for-agents/)
2. DeepLearning.AI Course: [Long-Term Agentic Memory with LangGraph](https://www.deeplearning.ai/short-courses/long-term-agentic-memory-with-langgraph/)
3. LangMem SDK: [Launch Announcement](https://blog.langchain.dev/langmem-sdk-launch/)
4. LangGraph Documentation: [Memory Concepts](https://langchain-ai.github.io/langgraph/concepts/memory/)
5. LangChain Blog: [Launching Long-Term Memory Support](https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/)