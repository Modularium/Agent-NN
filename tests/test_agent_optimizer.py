import unittest
from unittest.mock import patch, MagicMock
import asyncio
from datetime import datetime

from managers.agent_optimizer import AgentOptimizer

class TestAgentOptimizer(unittest.TestCase):
    def setUp(self):
        self.mlflow_patcher = patch('managers.agent_optimizer.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        self.mock_mlflow.set_experiment.return_value = MagicMock(experiment_id="exp")

        self.embed_patcher = patch('managers.agent_optimizer.OpenAIEmbeddings')
        self.mock_embed_cls = self.embed_patcher.start()
        self.mock_embed = MagicMock()
        self.mock_embed.embed_query.return_value = [0.1, 0.2, 0.3]
        self.mock_embed_cls.return_value = self.mock_embed

        self.chroma_patcher = patch('managers.agent_optimizer.Chroma')
        self.mock_chroma_cls = self.chroma_patcher.start()
        self.mock_chroma = MagicMock()
        self.mock_chroma.similarity_search_with_score.return_value = [(
            MagicMock(metadata={'domain': 'finance'}), 0.9
        )]
        self.mock_chroma.similarity_search.return_value = []
        self.mock_chroma_cls.return_value = self.mock_chroma

        self.optimizer = AgentOptimizer()

    def tearDown(self):
        self.mlflow_patcher.stop()
        self.embed_patcher.stop()
        self.chroma_patcher.stop()

    def test_evaluate_agent_without_metrics(self):
        result = asyncio.run(self.optimizer.evaluate_agent('a1'))
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(result['needs_optimization'])

    def test_get_agent_performance_trend(self):
        now = datetime.now().isoformat()
        self.optimizer.agent_metrics['a1'] = [
            {'response_quality': 0.8, 'task_success': True, 'user_satisfaction': 0.9, 'timestamp': now},
            {'response_quality': 0.6, 'task_success': False, 'user_satisfaction': 0.7, 'timestamp': now},
        ]
        trend = self.optimizer.get_agent_performance_trend('a1', days=1)
        self.assertAlmostEqual(trend['success_rate'][0], 0.5)

    def test_determine_domain(self):
        domain, score = asyncio.run(self.optimizer.determine_domain('desc', ['cap']))
        self.assertEqual(domain, 'finance')
        self.assertGreater(score, 0)

if __name__ == '__main__':
    unittest.main()
