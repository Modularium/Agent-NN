import chromadb

class VectorStore:
    def __init__(self, collection_name):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, docs):
        # Implementation abhängig von Dokumentformat
        for i, doc in enumerate(docs):
            self.collection.add(documents=[doc], ids=[f"doc_{i}"])

    def as_retriever(self):
        # Rückgabe eines Retrievers-Objekts je nach LangChain-Integration
        # Pseudocode: LangChain Chromadb Retriever
        from langchain.vectorstores import Chroma
        vec_store = Chroma(collection_name=self.collection.name)
        return vec_store.as_retriever()
