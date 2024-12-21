from datastores.vector_store import VectorStore

class WorkerAgentDB:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.store = VectorStore(collection_name=f"{agent_name}_knowledge")

    def ingest_documents(self, docs):
        # docs: Liste von Texten oder LangChain Document-Objekten
        self.store.add_documents(docs)

    def get_retriever(self):
        return self.store.as_retriever()
