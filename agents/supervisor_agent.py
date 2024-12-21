from managers.agent_manager import AgentManager
from managers.nn_manager import NNManager

class SupervisorAgent:
    def __init__(self):
        self.agent_manager = AgentManager()
        self.nn_manager = NNManager()

    def execute_task(self, task_description: str):
        # Fragt das NN, welcher Agent am besten geeignet ist
        chosen_agent_name = self.nn_manager.predict_best_agent(task_description, self.agent_manager.get_all_agents())

        if chosen_agent_name is None:
            # Erstelle neuen Agent
            chosen_agent = self.agent_manager.create_new_agent(task_description)
        else:
            chosen_agent = self.agent_manager.get_agent(chosen_agent_name)

        # Führe die Aufgabe aus
        result = chosen_agent.execute_task(task_description)
        return result
