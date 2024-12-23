"""Tests for knowledge base integration with local models."""
import os
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from langchain.schema import Document
from utils.knowledge_base import KnowledgeBaseManager
from llm_models.llm_backend import (
    LLMBackendType,
    LLMBackendManager,
    LMStudioLLM,
    LlamafileLLM
)

class TestKnowledgeBaseLocal(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.kb_dir = os.path.join(self.test_dir, 'knowledge_base')
        self.models_dir = os.path.join(self.test_dir, 'models')
        
        # Initialize managers
        self.kb_manager = KnowledgeBaseManager(self.kb_dir)
        self.llm_manager = LLMBackendManager()
        
        # Create test documents
        self.create_test_documents()
        
    def create_test_documents(self):
        """Create test documents for different domains."""
        # Finance documents
        self.finance_doc = os.path.join(self.test_dir, 'finance.txt')
        with open(self.finance_doc, 'w') as f:
            f.write("""Financial Analysis Report
            ROI: 15%
            Risk Assessment: Medium
            Investment Strategy: Diversified Portfolio""")
            
        # Tech documents
        self.tech_doc = os.path.join(self.test_dir, 'tech.txt')
        with open(self.tech_doc, 'w') as f:
            f.write("""Technical Documentation
            Language: Python
            Framework: LangChain
            Architecture: Microservices""")
            
        # Marketing documents
        self.marketing_doc = os.path.join(self.test_dir, 'marketing.txt')
        with open(self.marketing_doc, 'w') as f:
            f.write("""Marketing Campaign Plan
            Target Audience: Tech Professionals
            Channels: Social Media, Email
            Budget: $50,000""")
            
    def test_document_ingestion_with_local_models(self):
        """Test document ingestion using local models."""
        # Test with LM Studio
        self.llm_manager.set_backend(LLMBackendType.LMSTUDIO)
        
        with patch('requests.post') as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"text": "Test embedding"}]}
            )
            
            # Ingest finance document
            docs = self.kb_manager.ingest_document(
                self.finance_doc,
                "finance",
                {"source": "test"}
            )
            
            self.assertTrue(len(docs) > 0)
            self.assertEqual(docs[0].metadata["domain"], "finance")
            
    def test_document_retrieval_with_local_models(self):
        """Test document retrieval using local models."""
        # Ingest test documents
        self.kb_manager.ingest_document(self.finance_doc, "finance")
        self.kb_manager.ingest_document(self.tech_doc, "tech")
        self.kb_manager.ingest_document(self.marketing_doc, "marketing")
        
        # Test with Llamafile
        self.llm_manager.set_backend(LLMBackendType.LLAMAFILE)
        
        # Mock Llamafile server
        with patch.object(LlamafileLLM, '_call') as mock_call:
            mock_call.return_value = "Test response"
            
            # Search documents
            docs = self.kb_manager.search_documents("ROI calculation")
            self.assertTrue(any("ROI" in doc.page_content for doc in docs))
            
    def test_domain_specific_retrieval(self):
        """Test domain-specific document retrieval with local models."""
        # Ingest documents
        self.kb_manager.ingest_document(self.finance_doc, "finance")
        self.kb_manager.ingest_document(self.tech_doc, "tech")
        
        # Get domain documents
        finance_docs = self.kb_manager.get_domain_documents("finance")
        tech_docs = self.kb_manager.get_domain_documents("tech")
        
        # Verify domain separation
        self.assertTrue(all("finance" in doc.metadata["domain"] 
                          for doc in finance_docs))
        self.assertTrue(all("tech" in doc.metadata["domain"] 
                          for doc in tech_docs))
        
    def test_model_switching_during_retrieval(self):
        """Test switching between models during document retrieval."""
        # Ingest with OpenAI
        self.llm_manager.set_backend(LLMBackendType.OPENAI)
        self.kb_manager.ingest_document(self.finance_doc, "finance")
        
        # Retrieve with LM Studio
        self.llm_manager.set_backend(LLMBackendType.LMSTUDIO)
        with patch('requests.post') as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"text": "Test response"}]}
            )
            
            docs = self.kb_manager.search_documents("investment")
            self.assertTrue(len(docs) > 0)
            
        # Retrieve with Llamafile
        self.llm_manager.set_backend(LLMBackendType.LLAMAFILE)
        with patch.object(LlamafileLLM, '_call') as mock_call:
            mock_call.return_value = "Test response"
            
            docs = self.kb_manager.search_documents("investment")
            self.assertTrue(len(docs) > 0)
            
    def test_concurrent_model_access(self):
        """Test concurrent access to knowledge base with different models."""
        # Ingest documents
        self.kb_manager.ingest_document(self.finance_doc, "finance")
        self.kb_manager.ingest_document(self.tech_doc, "tech")
        
        # Simulate concurrent access
        def search_with_model(backend_type: LLMBackendType, query: str):
            self.llm_manager.set_backend(backend_type)
            return self.kb_manager.search_documents(query)
            
        # Mock responses for different backends
        with patch('requests.post') as mock_post, \
             patch.object(LlamafileLLM, '_call') as mock_llamafile:
                
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"text": "LM Studio response"}]}
            )
            mock_llamafile.return_value = "Llamafile response"
            
            # Run concurrent searches
            lmstudio_docs = search_with_model(
                LLMBackendType.LMSTUDIO,
                "python"
            )
            llamafile_docs = search_with_model(
                LLMBackendType.LLAMAFILE,
                "investment"
            )
            
            self.assertTrue(len(lmstudio_docs) > 0)
            self.assertTrue(len(llamafile_docs) > 0)
            
    def test_error_handling_with_local_models(self):
        """Test error handling when local models fail."""
        # Test LM Studio server failure
        self.llm_manager.set_backend(LLMBackendType.LMSTUDIO)
        with patch('requests.post', side_effect=ConnectionError()):
            with self.assertRaises(Exception):
                self.kb_manager.search_documents("test")
                
        # Test Llamafile server failure
        self.llm_manager.set_backend(LLMBackendType.LLAMAFILE)
        with patch.object(LlamafileLLM, 'start_server',
                         side_effect=Exception("Server start failed")):
            with self.assertRaises(Exception):
                self.kb_manager.search_documents("test")
                
    def test_model_performance_metrics(self):
        """Test collection of performance metrics for different models."""
        # This would test response times, memory usage, etc.
        # For now, just a placeholder
        pass
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()