import unittest
from unittest.mock import MagicMock

from services.routing_agent.service import RoutingAgentService


class TestRoutingAgentService(unittest.TestCase):
    def setUp(self):
        self.service = RoutingAgentService(rules_path="services/routing_agent/rules.yaml")

    def test_predict_rule(self):
        ctx = {"task_type": "greeting"}
        self.assertEqual(self.service.predict_agent(ctx), "worker_dev")

    def test_predict_fallback(self):
        ctx = {"task_type": "unknown"}
        self.assertEqual(self.service.predict_agent(ctx), "worker_dev")

    def test_predict_meta(self):
        self.service.meta = MagicMock()
        self.service.meta.predict_agent.return_value = "worker_openhands"
        ctx = {"task_type": "foo"}
        self.assertEqual(self.service.predict_agent(ctx), "worker_openhands")
        self.service.meta.predict_agent.assert_called_once()


class DummyClient:
    def __init__(self, agents):
        self.agents = agents
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def get(self, url):
        self.sent.append(("GET", url))
        return DummyResponse({"agents": self.agents})

    def post(self, url, json=None, timeout=5):
        self.sent.append(("POST", url, json))
        return DummyResponse({"target_worker": "worker_loh"})


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_dispatcher_routing(monkeypatch):
    from services.task_dispatcher.service import TaskDispatcherService

    dummy = DummyClient([
        {"name": "worker_dev", "capabilities": ["chat"]},
        {"name": "worker_loh", "capabilities": ["care"]},
    ])
    monkeypatch.setattr("httpx.Client", lambda: dummy)
    dispatcher = TaskDispatcherService()
    # patch routing call
    dispatcher._route_agent = lambda x: "worker_loh"

    agents = dispatcher._fetch_agents("chat")
    assert agents[0]["name"] == "worker_loh"
