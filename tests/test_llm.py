import pytest
from unittest.mock import MagicMock, patch
from utils.llm import LLMService

@pytest.fixture
def service():
    return LLMService()

def test_singleton_initialization(service):
    assert service._config["provider"] == "openai"
    assert service._max_concurrent == 200

def test_configure(service):
    service.configure(temperature=0.7, max_tokens=100)
    assert service._config["temperature"] == 0.7
    assert service._config["max_tokens"] == 100

def test_get_model(service):
    assert service.get_model("default") == "gpt-5-nano"
    service.configure(model="gpt-4o")
    assert service.get_model("default") == "gpt-4o"

@patch("utils.llm.openai")
def test_openai_sync_call(mock_openai, service):
    # Setup mock
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Hello"
    mock_resp.usage.prompt_tokens = 10
    mock_resp.usage.completion_tokens = 5
    mock_client.chat.completions.create.return_value = mock_resp
    
    # Execute
    result = service.call_sync("Hi", provider="openai")
    
    # Verify
    assert result == "Hello"
    mock_client.chat.completions.create.assert_called_once()
