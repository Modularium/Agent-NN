import unittest
from fastapi.testclient import TestClient
from services.session_manager.main import app as session_app

class TestMetricsEndpoint(unittest.TestCase):
    def test_metrics_available(self):
        client = TestClient(session_app)
        resp = client.get('/metrics')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('agentnn_response_seconds', resp.text)

if __name__ == '__main__':
    unittest.main()
