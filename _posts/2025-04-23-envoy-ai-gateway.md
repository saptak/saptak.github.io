---
author: Saptak Sen
categories:
- cloud-native
- kubernetes
- ai
- gateway
date: 2025-04-23
header_image_path: /assets/img/blog/headers/2025-04-23-envoy-ai-gateway.jpg
image_credit: Photo by Kvistholt Photography on Unsplash
layout: post
mermaid: true
tags:
- envoy
- gateway-api
- kubernetes
- ai-inference
- cloud-native
thumbnail_path: /assets/img/blog/thumbnails/2025-04-23-envoy-ai-gateway.jpg
title: 'Envoy AI Gateway: The Next Frontier in AI Inference Management'
toc: true
---

## Introduction

With AI inference services becoming as fundamental to enterprise architecture as compute, storage, databases, and networking, organizations are now grappling with a new challenge: how to efficiently and securely manage access to the growing ecosystem of AI models while maintaining governance and control. In this post, I'll explore how the Envoy AI Gateway builds upon the solid foundation of the Kubernetes Gateway API and Envoy proxy to address these challenges.

## AI Industry Context & Challenges

For many industries, AI inference services have become a critical requirement for staying competitive.

The current landscape presents several challenges:

- Over 400,000 models are now available across various platforms
- Multiple hosting providers
- Organizations face internal friction between product teams seeking model access and security/governance teams requiring control

The gateway pattern, which has solved similar enterprise challenges in the past, is emerging as a solution for AI workloads

For most customers, model choice matters significantly, and no single model will dominate the ecosystem - much like what we've seen with databases. Organizations need flexibility without sacrificing security or governance.

## Evolution of Gateway Architecture

To understand the Envoy AI Gateway, it's helpful to trace the evolution of the gateway pattern in Kubernetes:

```mermaid
timeline
    title Gateway API Evolution
    2015 : Service abstraction (v1.0)
    2017 : Envoy Proxy donated to CNCF
    2020 : TwinMedes 3 release
           Ingress controllers matured
    2022 : Envoy Gateway project launched
           Focus on North-South traffic patterns
    2023 : Gateway API became GA (v1.26)
    2025 : Envoy AI Gateway development
           Specialized for inference workloads
```

The Gateway API has evolved to become more declarative and portable than previous solutions, offering sophisticated L7 routing definition capabilities. This maturation provides the perfect foundation for AI-specific extensions.

## Gateway API for AI Implementation

The AI Gateway implementation extends the Gateway API with features specifically designed for inference workloads:

### New Custom Resource Definitions (CRDs)

Two key CRDs have been introduced to handle AI-specific concerns:

1. **Inference Model CRD** - Manages model resources
2. **Inference Pool CRD** - Handles resource allocation

### Specialized Routing Capabilities

The AI Gateway introduces advanced routing mechanisms:

```mermaid
flowchart TB
    Client[Client Request] --> Gateway[AI Gateway Route]
    Gateway --> Processor[External Processor]
    Processor --> Metrics{Endpoint Picker}
    Metrics --> |QDAP Metrics| Model1[Model Endpoint 1]
    Metrics --> |GPU Memory| Model2[Model Endpoint 2]
    Metrics --> |Model Loading| Model3[Model Endpoint 3]
    Metrics --> |Adapter Status| Model4[Model Endpoint 4]
    Model1 --> Response[Response]
    Model2 --> Response
    Model3 --> Response
    Model4 --> Response
```

- Endpoint picker mechanism using metrics like QDAP and GPU memory
- Intelligent routing based on model loading status
- Adapter-aware routing decisions
- Projects like K Gateway and Envoy Gateway now implementing these features

## Envoy AI Gateway Technical Details

### Foundation and Evolution

The Envoy AI Gateway is built on the solid foundation of Envoy proxy, a high-performance proxy written in C++ that started in 2015 and was donated to the CNCF in 2017. It's battle-tested and runs in some of the world's largest environments.

Tetrate saw a need for easier management of Envoy as a gateway solution and built Envoy Gateway on top of Envoy proxy. This wraps the Envoy proxy data plane in a simplified management and configuration layer, making it easier to configure Envoy for North-South traffic.

Envoy AI Gateway extends this further to address the unique challenges of inference workloads, which unlike standard API requests, have:
- Token-based pricing models
- Unique metrics requirements
- Different routing patterns

### Key Components

```mermaid
flowchart LR
    subgraph Control Plane
        Controller[AI Gateway Controller]
        GatewayController[Gateway Controller]
        XDS[XDS APIs]
    end

    subgraph Data Plane
        Proxy[Envoy Proxy]
        Processor[External Processor]
        RateLimiter[Rate Limit Service]
    end

    Controller --> GatewayController
    GatewayController --> XDS
    XDS --> Proxy
    Proxy <--> Processor
    Proxy <--> RateLimiter
```

The core components include:
- **Envoy Proxy**: The high-performance data plane (written in C++)
- **External Processor**: For custom metrics extraction, request/response transformation, and credential injection
- **Rate Limit Service**: Token-based rate limiting optimized for AI workloads
- **AI Gateway Controller**: Manages AI-specific configuration
- **Gateway Controller**: Handles normal routing management

### Custom Resources

Envoy AI Gateway adds several Custom Resource Definitions (CRDs) to handle AI-specific configuration:

1. **AIGatewayRoute** (Application Developer)
   - Defines a unified AI API for a Gateway
   - Allows clients to interact with multiple AI backends using a single schema

2. **AIServiceBackend** (Inference Platform Owner)
   - Represents a single AI service backend that handles traffic with a specific schema

3. **BackendSecurityPolicy** (Inference Platform Owner)
   - Configures authentication and authorization rules for backend access
   - Contains references to API keys or credentials

When these CRDs are processed by the AI Gateway controller, they're rendered in the cluster as:
- An ExtProc Server Deployment (to handle specific processing for the route)
- HTTPRoute and Envoy Extension Policy (to wire it all together)

### Configuration Flow

```mermaid
flowchart TD
    subgraph AIGateway[AI Gateway Controller]
        AIResources[AI Gateway Resources]
        AIController[AI Resource Controller]
    end

    subgraph Gateway[Gateway Controller]
        GWResources[Gateway Resources]
        GWController[Gateway Controller]
    end

    subgraph DataPlane[Data Plane]
        Proxy[Envoy Proxy]
    end

    AIResources --> AIController
    AIController --> GWResources
    GWResources --> GWController
    GWController -->|xDS APIs| Proxy
```

Envoy Gateway and Envoy AI Gateway controllers maintain a separation of concerns:
- The Gateway controller handles normal routing management of the Envoy proxy
- The AI Gateway controller handles the AI-specific configuration (external processor, prompt and credential injection, rate limiting)

When the AI Gateway controller detects changes in AI Gateway Custom Resources, it updates the Envoy Gateway Configuration, which then updates the Envoy proxy configuration via xDS APIs.

### Request Flow

The complete lifecycle of a request through the system:

```mermaid
sequenceDiagram
    participant Client
    participant Proxy as Envoy Proxy
    participant Processor as External Processor
    participant RateLimiter as Rate Limit Service
    participant AIProvider as AI Provider

    Client->>Proxy: Request to unified API
    Proxy->>Proxy: Check route permissions
    Proxy->>RateLimiter: Verify rate limits
    RateLimiter->>Proxy: Rate limit decision
    Proxy->>Processor: Process request
    Processor->>Proxy: Provider routing determination
    Proxy->>Processor: Request transformation
    Processor->>Proxy: Transformed request with credentials
    Proxy->>AIProvider: Forward request to provider
    AIProvider->>Proxy: Response
    Proxy->>Processor: Response transformation
    Processor->>Proxy: Transformed response
    Processor->>RateLimiter: Update token usage metrics
    Proxy->>Client: Final response
```

On the way in:
1. Client calls using the unified API
2. Envoy proxy checks that the client is allowed to call that route
3. Envoy verifies applicable rate limits aren't exceeded
4. If allowed, the system determines which provider to route to
5. The external processor translates the request from the unified API to the upstream API schema
6. Credentials for the upstream are injected before routing to the provider

On the way back:
1. The external processor translates the response back to the unified API schema
2. Token usage metrics are extracted and the rate limit service is updated
3. The response is returned to the client

## Advanced Routing with Inference Extensions

Envoy AI Gateway supports intelligent routing beyond traditional methods through the Gateway API inference extension support:

```mermaid
flowchart LR
    Client[Client Request] --> AIGateway[AI Gateway]
    AIGateway --> EndpointPicker[Endpoint Picker Extension]

    subgraph Metrics
        KVCache[KV-cache Utilization]
        QueueLength[Queue Length]
        ActiveAdapters[Active LoRA Adapters]
    end

    EndpointPicker --> InferencePool[Inference Pool]
    KVCache --> EndpointPicker
    QueueLength --> EndpointPicker
    ActiveAdapters --> EndpointPicker

    InferencePool --> ModelServer1[Model Server 1]
    InferencePool --> ModelServer2[Model Server 2]
    InferencePool --> ModelServer3[Model Server 3]
```

When using inference extensions, routes target inference pools rather than AI backends directly:

- An InferencePool is bundled with an Endpoint Picker extension
- This extension tracks key metrics on each model server:
  - KV-cache utilization
  - Queue length of pending requests
  - Active LoRA adapters
- The endpoint picker then routes incoming inference requests to the optimal model server replica based on these metrics

## AWS Implementation

Deploying the Envoy AI Gateway on AWS leverages several key services:

```mermaid
flowchart TD
    Client[Client] --> NLB[Network Load Balancer]
    NLB --> EnvoyFleet[Envoy Proxy Fleet]
    EnvoyFleet --> |CPU-optimized nodes| EKS[EKS Managed Node Groups]

    EnvoyFleet --> SelfHosted[Self-hosted Models]
    EnvoyFleet --> Bedrock[AWS Bedrock]
    EnvoyFleet --> SageMaker[SageMaker]
    EnvoyFleet --> OnPrem[On-Prem/External]

    EnvoyFleet --> CloudWatch[CloudWatch Metrics]
    EnvoyFleet --> CloudTrail[CloudTrail Logging]
```

The AWS implementation includes:
- Routes through Network Load Balancer to the Envoy proxy fleet
- Proxies are CPU-bound, so compute-optimized nodes are recommended
- Integration with EKS for Horizontal Pod Autoscaling (HPA)
- Optional integration with Istio for certificate management and mTLS
- Support for multiple backends:
  - SageMaker
  - Bedrock
  - Self-hosted models on Trainium/Inferentia instances
  - GPU-optimized nodes
  - External providers (on-prem or SaaS solutions)
- CloudWatch metrics integration for observability
- CloudTrail access logging for security and compliance

## Istio Integration

Envoy AI Gateway can be configured to work seamlessly with Istio:
- Can function as an Ingress Gateway into an Istio service mesh
- Istio can manage all certificates for mTLS
- Organizations get advanced gateway capabilities from Envoy while leveraging Istio's service mesh features

## Community and Contribution

The Envoy AI Gateway is an open-source project with strong industry backing:

- Tetrate contributes more than half of the commits to both Envoy Gateway and Envoy AI Gateway projects
- The project aims to evolve to support complex, real-world workloads including GenAI, edge, and enterprise service mesh scenarios
- The community addresses interesting challenges in networking, performance, and security
- The Gateway and AI Gateway projects are actively seeking contributors who want to help shape their future

## Real-World Adoption

Envoy AI Gateway is being adopted by organizations looking to manage their AI inference at scale. Bloomberg is one notable early adopter, leveraging the gateway for their AI infrastructure needs.

## Conclusion

The Envoy AI Gateway represents a significant step forward in managing AI inference at scale. By building on the proven patterns of the Gateway API and the robust foundation of Envoy proxy, organizations can now address the unique challenges of AI inference workloads without sacrificing security, performance, or governance.

With its performance advantages, flexible routing capabilities, and strong community backing, Envoy AI Gateway is positioned to become a critical component of enterprise AI infrastructure as organizations continue to integrate AI into their applications and services.

As we continue to develop and refine this architecture, we invite feedback from the community and look forward to seeing how these patterns evolve to meet the rapidly changing demands of enterprise AI deployments.
