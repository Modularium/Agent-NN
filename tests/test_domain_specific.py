"""Tests for domain-specific functionality."""
import os
import unittest
import asyncio
from unittest.mock import patch, MagicMock
import json
import tempfile
from pathlib import Path
from langchain.schema import Document

from utils.prompts import (
    get_domain_template,
    get_task_template,
    create_combined_prompt,
    get_system_prompt
)
from utils.document_manager import DocumentManager
from agents.worker_agent import WorkerAgent
from agents.agent_communication import AgentCommunicationHub
from agents.domain_knowledge import DomainKnowledgeManager

class TestDomainSpecific(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test directories
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.docs_dir = cls.test_dir / "documents"
        cls.docs_dir.mkdir()
        
        # Create test documents
        cls.create_test_documents()
        
    @classmethod
    def create_test_documents(cls):
        """Create test documents for different domains."""
        # Finance document
        finance_doc = cls.docs_dir / "finance_report.txt"
        finance_doc.write_text("""
        Financial Analysis Report
        
        Key Metrics:
        - Revenue: $1M
        - Profit Margin: 15%
        - ROI: 22%
        
        Risk Assessment:
        - Market Risk: Medium
        - Credit Risk: Low
        """)
        
        # Tech document
        tech_doc = cls.docs_dir / "tech_spec.txt"
        tech_doc.write_text("""
        Technical Specification
        
        System Architecture:
        - Microservices
        - Kubernetes
        - Message Queue
        
        Security:
        - OAuth2
        - Encryption
        - Firewalls
        """)
        
        # Marketing document
        marketing_doc = cls.docs_dir / "marketing_plan.txt"
        marketing_doc.write_text("""
        Marketing Campaign Plan
        
        Target Audience:
        - Age: 25-45
        - Income: $50k+
        - Interests: Tech
        
        Channels:
        - Social Media
        - Email
        - Content Marketing
        """)
        
    def setUp(self):
        """Set up each test."""
        self.doc_manager = DocumentManager(str(self.docs_dir))
        self.comm_hub = AgentCommunicationHub()
        self.knowledge_manager = DomainKnowledgeManager()
        
    def test_domain_templates(self):
        """Test domain-specific prompt templates."""
        # Test finance template
        finance_template = get_domain_template("finance")
        self.assertIn("financial expert", finance_template.template)
        self.assertIn("Risk factors", finance_template.template)
        
        # Test tech template
        tech_template = get_task_template("tech", "code_review")
        self.assertIn("code quality", tech_template.template)
        self.assertIn("security issues", tech_template.template)
        
        # Test combined template
        combined = create_combined_prompt(
            "marketing",
            "campaign_analysis",
            {"budget": "$50,000"}
        )
        self.assertIn("marketing strategy expert", combined.template)
        self.assertIn("campaign analysis", combined.template.lower())
        self.assertIn("budget: $50,000", combined.template.lower())
        
    def test_document_ingestion(self):
        """Test document ingestion for different domains."""
        # Test finance document
        finance_docs = self.doc_manager.ingest_file(
            str(self.docs_dir / "finance_report.txt"),
            "finance",
            {"category": "analysis"}
        )
        self.assertTrue(len(finance_docs) > 0)
        self.assertEqual(finance_docs[0].metadata["domain"], "finance")
        self.assertEqual(finance_docs[0].metadata["category"], "analysis")
        
        # Test tech document
        tech_docs = self.doc_manager.ingest_file(
            str(self.docs_dir / "tech_spec.txt"),
            "tech",
            {"type": "specification"}
        )
        self.assertTrue(len(tech_docs) > 0)
        self.assertEqual(tech_docs[0].metadata["domain"], "tech")
        
        # Test marketing document
        marketing_docs = self.doc_manager.ingest_file(
            str(self.docs_dir / "marketing_plan.txt"),
            "marketing"
        )
        self.assertTrue(len(marketing_docs) > 0)
        self.assertEqual(marketing_docs[0].metadata["domain"], "marketing")
        
    def test_document_search(self):
        """Test document search functionality."""
        # Ingest all documents
        for doc in self.docs_dir.glob("*.txt"):
            domain = doc.stem.split("_")[0]
            self.doc_manager.ingest_file(str(doc), domain)
            
        # Search in finance domain
        finance_results = self.doc_manager.search_documents(
            "ROI",
            domain="finance"
        )
        self.assertTrue(len(finance_results) > 0)
        self.assertIn("ROI", finance_results[0].page_content)
        
        # Search in tech domain
        tech_results = self.doc_manager.search_documents(
            "kubernetes",
            domain="tech"
        )
        self.assertTrue(len(tech_results) > 0)
        self.assertIn("Kubernetes", tech_results[0].page_content)
        
        # Cross-domain search
        results = self.doc_manager.search_documents("risk")
        self.assertTrue(len(results) > 0)
        
    def test_domain_statistics(self):
        """Test domain statistics."""
        # Ingest documents
        for doc in self.docs_dir.glob("*.txt"):
            domain = doc.stem.split("_")[0]
            self.doc_manager.ingest_file(str(doc), domain)
            
        # Get finance stats
        finance_stats = self.doc_manager.get_domain_statistics("finance")
        self.assertGreater(finance_stats["total_documents"], 0)
        self.assertGreater(finance_stats["total_tokens"], 0)
        
        # Get tech stats
        tech_stats = self.doc_manager.get_domain_statistics("tech")
        self.assertGreater(tech_stats["total_documents"], 0)
        self.assertIn(".txt", tech_stats["file_types"])
        
    async def test_worker_agent_specialization(self):
        """Test specialized worker agent functionality."""
        # Create finance agent
        finance_agent = WorkerAgent(
            "finance",
            communication_hub=self.comm_hub,
            knowledge_manager=self.knowledge_manager
        )
        
        # Ingest domain knowledge
        finance_doc = self.docs_dir / "finance_report.txt"
        finance_agent.ingest_knowledge([str(finance_doc)])
        
        # Test domain-specific query
        response = await finance_agent.execute_task(
            "What is the ROI mentioned in the report?"
        )
        self.assertIn("22%", response)
        
        # Test with context
        response = await finance_agent.execute_task(
            "Is this a good investment?",
            context="Consider the ROI of 22% and medium market risk."
        )
        self.assertIn("ROI", response)
        self.assertIn("risk", response.lower())
        
    async def test_cross_domain_communication(self):
        """Test communication between agents of different domains."""
        # Create agents
        finance_agent = WorkerAgent(
            "finance",
            communication_hub=self.comm_hub,
            knowledge_manager=self.knowledge_manager
        )
        marketing_agent = WorkerAgent(
            "marketing",
            communication_hub=self.comm_hub,
            knowledge_manager=self.knowledge_manager
        )
        
        # Ingest domain knowledge
        finance_agent.ingest_knowledge([str(self.docs_dir / "finance_report.txt")])
        marketing_agent.ingest_knowledge([str(self.docs_dir / "marketing_plan.txt")])
        
        # Start message processing
        asyncio.create_task(finance_agent.process_messages())
        asyncio.create_task(marketing_agent.process_messages())
        
        # Test cross-domain query
        response = await finance_agent.communicate_with_other_agent(
            "marketing",
            "What is the target audience's income level?"
        )
        
        self.assertIn("response", response)
        self.assertIn("50k", response["response"])
        
    def test_error_handling(self):
        """Test error handling in domain-specific operations."""
        # Test invalid domain
        with self.assertRaises(ValueError):
            get_domain_template("invalid_domain")
            
        # Test invalid file type
        with self.assertRaises(ValueError):
            self.doc_manager.ingest_file(
                "test.invalid",
                "finance"
            )
            
        # Test invalid task type
        with self.assertRaises(ValueError):
            get_task_template("tech", "invalid_task")
            
    def test_document_export(self):
        """Test document export functionality."""
        # Ingest documents
        for doc in self.docs_dir.glob("*.txt"):
            domain = doc.stem.split("_")[0]
            self.doc_manager.ingest_file(str(doc), domain)
            
        # Export finance data
        finance_export = self.doc_manager.export_domain_data(
            "finance",
            "json"
        )
        self.assertTrue(os.path.exists(finance_export))
        
        # Verify exported data
        with open(finance_export) as f:
            data = json.load(f)
            self.assertTrue(len(data) > 0)
            self.assertEqual(data[0]["domain"], "finance")
            
    def tearDown(self):
        """Clean up after each test."""
        # Clean up any test files
        pass
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(cls.test_dir)

def async_test(coro):
    """Decorator for async test methods."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

# Apply async_test decorator to test methods
for name in dir(TestDomainSpecific):
    if name.startswith('test_'):
        attr = getattr(TestDomainSpecific, name)
        if asyncio.iscoroutinefunction(attr):
            setattr(TestDomainSpecific, name, async_test(attr))

if __name__ == '__main__':
    unittest.main()