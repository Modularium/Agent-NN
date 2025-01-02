import unittest
import os
import shutil
from langchain.schema import Document
from datastores.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.test_collection = "test_collection"
        self.test_docs = [
            Document(page_content="This is a test document about AI.", metadata={"source": "test1.txt"}),
            Document(page_content="Another document about machine learning.", metadata={"source": "test2.txt"}),
            Document(page_content="A document about something completely different.", metadata={"source": "test3.txt"})
        ]
        self.vector_store = VectorStore(self.test_collection)

    def tearDown(self):
        # Clean up test data
        if os.path.exists("data/vectorstore"):
            shutil.rmtree("data/vectorstore")

    def test_add_and_search(self):
        """Test adding documents and searching them"""
        # Add documents
        self.vector_store.add_documents(self.test_docs)
        
        # Search for similar documents
        results = self.vector_store.similarity_search("AI and machine learning", k=2)
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertTrue(any("AI" in doc.page_content for doc in results))
        self.assertTrue(any("machine learning" in doc.page_content for doc in results))
        
        # The document about "something completely different" should not be in top 2
        self.assertFalse(any("completely different" in doc.page_content for doc in results))

    def test_retriever_interface(self):
        """Test the retriever interface"""
        # Add documents
        self.vector_store.add_documents(self.test_docs)
        
        # Get retriever
        retriever = self.vector_store.as_retriever(k=2)
        
        # Search using retriever
        results = retriever.invoke("AI and machine learning")
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertTrue(any("AI" in doc.page_content for doc in results))
        self.assertTrue(any("machine learning" in doc.page_content for doc in results))

    def test_delete_collection(self):
        """Test deleting a collection"""
        # Add documents
        self.vector_store.add_documents(self.test_docs)
        
        # Delete collection
        self.vector_store.delete_collection()
        
        # Search should return empty results
        results = self.vector_store.similarity_search("AI", k=1)
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()