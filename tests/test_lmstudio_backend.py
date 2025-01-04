"""Tests for LM Studio backend."""
import pytest
from llm_models.lmstudio_backend import LMStudioLLM, LMStudioEmbeddings, load_llm_config

def test_load_llm_config():
    """Test loading LLM configuration."""
    config = load_llm_config()
    assert isinstance(config, dict)
    assert "base_url" in config
    assert "chat_model" in config
    assert "embedding_model" in config

@pytest.mark.asyncio
async def test_lmstudio_llm():
    """Test LM Studio LLM."""
    config = load_llm_config()
    llm = LMStudioLLM(
        base_url=config["base_url"],
        model=config["chat_model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )
    
    # Test simple completion
    response = llm.invoke("What is 2+2?")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test more complex prompt
    response = llm.invoke("""You are a helpful AI assistant. Please help me solve this math problem:
    If a train travels at 60 mph for 2 hours, how far does it travel?""")
    assert isinstance(response, str)
    assert len(response) > 0
    assert "120" in response  # Should mention the answer (60 * 2 = 120 miles)

def test_lmstudio_embeddings():
    """Test LM Studio embeddings."""
    config = load_llm_config()
    embeddings = LMStudioEmbeddings(
        base_url=config["base_url"],
        model=config["embedding_model"]
    )
    
    # Test single text embedding
    text = "This is a test sentence."
    embedding = embeddings.embed_query(text)
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
    
    # Test multiple texts embeddings
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence."
    ]
    embeddings_list = embeddings.embed_documents(texts)
    assert isinstance(embeddings_list, list)
    assert len(embeddings_list) == len(texts)
    assert all(isinstance(x, list) for x in embeddings_list)
    assert all(isinstance(y, float) for x in embeddings_list for y in x)