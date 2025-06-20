# Architecture Overview

This document summarizes the current structure of the **Agent-NN** project. The repository contains a modular multi-agent system that combines LLM components with neural networks for task routing and execution.

## Top Level Components

- **agents/** – implementation of all agent types (chatbot, supervisor, worker agents, OpenHands agents, etc.)
- **llm_models/** – wrappers for different LLM backends and embeddings
- **managers/** – system managers for agent lifecycle, models, knowledge, monitoring and more
- **datastores/** – vector store and agent specific databases
- **training/** – training pipeline for the neural models
- **cli/** – command line interface utilities
- **api/** and **openhands_api/** – FastAPI based services
- **utils/** – helper modules (logging, prompt templates, document management)

## Agent Hierarchy

1. **ChatbotAgent** – user facing agent that forwards requests to the SupervisorAgent.
2. **SupervisorAgent** – decides which WorkerAgent should handle a task using the NNManager.
3. **WorkerAgent** – domain specific executor with its own knowledge base.
4. **Specialized agents** – e.g. OpenHands agents for Docker/Compose operations and software development agents under `agents/software_dev`.

Agents communicate via the `AgentCommunicationHub` and can ingest knowledge through the `DomainKnowledgeManager` and `WorkerAgentDB`.

## Neural Components

- `nn_models/agent_nn.py` – neural network used by WorkerAgents for feature extraction and performance evaluation.
- `managers/nn_manager.py` – selects suitable agents based on embeddings and task requirements.
- Training scripts under `training/` provide the dataset preparation and MLflow tracking.

## Knowledge & Storage

Vector based search is implemented via `datastores/vector_store.py`. Each WorkerAgent uses `WorkerAgentDB` for its documents and retrieval. The project also integrates MLflow for experiment tracking and model registry.

## System Services

FastAPI services reside in `api/` (general API) and `openhands_api/` (execution environment). Monitoring is handled by `openhands_api/monitoring.py` and additional managers under `managers/` provide caching, security, deployment and more.

For a detailed description of modules, see `docs/architecture/Code-Dokumentation.md`.
