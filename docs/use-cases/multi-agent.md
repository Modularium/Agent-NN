# Retrieval-Augmented Generation

With the coordination service a task can be solved by multiple workers. The dispatcher sends the same request to a `retriever` and a `writer` agent. The retriever searches the vector store while the writer uses the documents to craft a final answer via the LLM gateway.

```bash
curl -X POST http://dispatcher/task \
     -d '{"task_type": "demo", "mode": "orchestrated", "description": "Explain gravity"}'
```
