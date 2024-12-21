from agents.worker_agent import WorkerAgent

class AgentManager:
    def __init__(self):
        self.agents = {}
        # Beispiel: Ein paar Agenten initialisieren
        self.agents["finance_agent"] = WorkerAgent("finance_agent", ["Rechnung 1: ...", "FAQ Finanzen"])
        self.agents["marketing_agent"] = WorkerAgent("marketing_agent", ["Marketing Strategie 2024", "Kampagnenplan Q1"])

    def get_all_agents(self):
        return list(self.agents.keys())

    def get_agent(self, name):
        return self.agents.get(name)

    def create_new_agent(self, task_description):
        # Logik um Domain aus task zu extrahieren
        domain = self._infer_domain(task_description)
        new_agent_name = f"{domain}_agent_{len(self.agents)+1}"
        # Domain-spezifische Doks in der Praxis: extrahieren oder anlernen
        new_agent = WorkerAgent(new_agent_name, ["Domain-Dokument 1", "Domain-Dokument 2"])
        self.agents[new_agent_name] = new_agent
        return new_agent

    def _infer_domain(self, task_description):
        # Sehr einfache Heuristik:
        if "Kunde" in task_description or "Rechnung" in task_description:
            return "finance"
        return "general"
