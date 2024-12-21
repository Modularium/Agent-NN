from langchain.chains import RetrievalQA
from llm_models.specialized_llm import SpecializedLLM
from datastores.worker_agent_db import WorkerAgentDB

class WorkerAgent:
    def __init__(self, name, domain_docs=None):
        self.name = name
        self.db = WorkerAgentDB(name)
        if domain_docs:
            self.db.ingest_documents(domain_docs)
        self.llm = SpecializedLLM(domain=self.name)
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm.get_llm(), retriever=self.db.get_retriever())

    def execute_task(self, task_description: str):
        # Kurzzeitkontext könnte hier gesetzt werden
        response = self.qa_chain.run(task_description)
        return response

    def communicate_with_other_agent(self, other_agent, query):
        # Beispielhafte Inter-Agent Kommunikation
        other_response = other_agent.execute_task(query)
        return other_response
