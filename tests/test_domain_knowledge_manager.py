import unittest
from unittest.mock import patch, MagicMock
from managers.domain_knowledge_manager import DomainKnowledgeManager
from langchain.schema import Document

class TestDomainKnowledgeManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Mock OpenAI embeddings
        self.embeddings_patcher = patch('managers.domain_knowledge_manager.OpenAIEmbeddings')
        self.mock_embeddings = self.embeddings_patcher.start()
        
        # Set up mock embeddings instance
        self.mock_embeddings_instance = MagicMock()
        self.mock_embeddings.return_value = self.mock_embeddings_instance
        
        # Initialize manager
        self.manager = DomainKnowledgeManager()
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.embeddings_patcher.stop()
        
    def test_add_domain(self):
        """Test domain creation."""
        # Add domain
        self.manager.add_domain(
            "test_domain",
            "Test domain description",
            initial_docs=["Test document"]
        )
        
        # Check domain was added
        self.assertIn("test_domain", self.manager.get_all_domains())
        
        # Check metadata
        info = self.manager.get_domain_info("test_domain")
        self.assertEqual(info["description"], "Test domain description")
        self.assertEqual(info["document_count"], 1)
        
    def test_add_documents(self):
        """Test document addition."""
        # Add domain
        self.manager.add_domain("test_domain", "Test domain")
        
        # Add documents
        docs = [
            Document(page_content="Test 1"),
            Document(page_content="Test 2")
        ]
        self.manager.add_documents("test_domain", docs)
        
        # Check document count
        info = self.manager.get_domain_info("test_domain")
        self.assertEqual(info["document_count"], 2)
        
    def test_search_domain(self):
        """Test domain search."""
        # Add domain with documents
        self.manager.add_domain(
            "test_domain",
            "Test domain",
            initial_docs=["Test document"]
        )
        
        # Mock search results
        mock_docs = [Document(page_content="Test result")]
        self.manager.vector_stores["test_domain"].similarity_search = MagicMock(
            return_value=mock_docs
        )
        
        # Search domain
        results = self.manager.search_domain("test_domain", "test query")
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Test result")
        
    def test_search_all_domains(self):
        """Test multi-domain search."""
        # Add domains
        self.manager.add_domain(
            "domain1",
            "Domain 1",
            initial_docs=["Test 1"]
        )
        self.manager.add_domain(
            "domain2",
            "Domain 2",
            initial_docs=["Test 2"]
        )
        
        # Mock search results
        mock_docs = [Document(page_content="Test result")]
        for domain in ["domain1", "domain2"]:
            self.manager.vector_stores[domain].similarity_search = MagicMock(
                return_value=mock_docs
            )
            
        # Search all domains
        results = self.manager.search_all_domains("test query")
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertIn("domain1", results)
        self.assertIn("domain2", results)
        
    def test_remove_domain(self):
        """Test domain removal."""
        # Add domain
        self.manager.add_domain("test_domain", "Test domain")
        
        # Remove domain
        self.manager.remove_domain("test_domain")
        
        # Check domain was removed
        self.assertNotIn("test_domain", self.manager.get_all_domains())
        
    def test_clear_domain(self):
        """Test domain clearing."""
        # Add domain with documents
        self.manager.add_domain(
            "test_domain",
            "Test domain",
            initial_docs=["Test document"]
        )
        
        # Clear domain
        self.manager.clear_domain("test_domain")
        
        # Check document count
        info = self.manager.get_domain_info("test_domain")
        self.assertEqual(info["document_count"], 0)
        
    def test_get_domain_stats(self):
        """Test domain statistics."""
        # Add domains
        self.manager.add_domain(
            "domain1",
            "Domain 1",
            initial_docs=["Test 1"]
        )
        self.manager.add_domain(
            "domain2",
            "Domain 2",
            initial_docs=["Test 2"]
        )
        
        # Get stats
        stats = self.manager.get_domain_stats()
        
        # Check stats
        self.assertEqual(len(stats), 2)
        self.assertIn("domain1", stats)
        self.assertIn("domain2", stats)
        self.assertEqual(stats["domain1"]["document_count"], 1)
        self.assertEqual(stats["domain2"]["document_count"], 1)

if __name__ == '__main__':
    unittest.main()