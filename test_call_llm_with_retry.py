import pytest
from unittest.mock import patch, MagicMock

from openai import APITimeoutError, RateLimitError, APIError

import weather_agent_loop as wal


@pytest.fixture
def mock_messages():
    return [{"role": "user", "content": "test"}]


@pytest.fixture
def mock_tools():
    return [{"type": "function", "function": {"name": "test"}}]


class TestCallLlmWithRetry:
    """call_llm_with_retry 函数测试"""

    def test_success_first_attempt(self, mock_messages, mock_tools):
        """成功调用，第一次就返回"""
        mock_response = MagicMock()
        with patch.object(wal.client.chat.completions, "create", return_value=mock_response):
            response, error = wal.call_llm_with_retry(mock_messages, mock_tools)
        
        assert response == mock_response
        assert error is None

    @patch("weather_agent_loop.time.sleep")
    def test_timeout_retry_then_success(self, mock_sleep, mock_messages, mock_tools):
        """超时后重试成功"""
        mock_response = MagicMock()
        with patch.object(
            wal.client.chat.completions, "create",
            side_effect=[APITimeoutError(request=MagicMock()), mock_response]
        ):
            response, error = wal.call_llm_with_retry(mock_messages, mock_tools)
        
        assert response == mock_response
        assert error is None
        mock_sleep.assert_called_once_with(1)  # 2 ** 0 = 1

    @patch("weather_agent_loop.time.sleep")
    def test_timeout_all_retries_exhausted(self, mock_sleep, mock_messages, mock_tools):
        """超时，重试次数用尽"""
        with patch.object(
            wal.client.chat.completions, "create",
            side_effect=APITimeoutError(request=MagicMock())
        ):
            response, error = wal.call_llm_with_retry(mock_messages, mock_tools)
        
        assert response is None
        assert error == "API 请求超时，请稍后重试"

    @patch("weather_agent_loop.time.sleep")
    def test_rate_limit_retry_then_success(self, mock_sleep, mock_messages, mock_tools):
        """限流后重试成功"""
        mock_response = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        with patch.object(
            wal.client.chat.completions, "create",
            side_effect=[RateLimitError("rate limit", response=mock_resp, body=None), mock_response]
        ):
            response, error = wal.call_llm_with_retry(mock_messages, mock_tools)
        
        assert response == mock_response
        assert error is None
        mock_sleep.assert_called_once_with(4)  # 2 ** (0 + 2) = 4

    @patch("weather_agent_loop.time.sleep")
    def test_rate_limit_all_retries_exhausted(self, mock_sleep, mock_messages, mock_tools):
        """限流，重试次数用尽"""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        with patch.object(
            wal.client.chat.completions, "create",
            side_effect=RateLimitError("rate limit", response=mock_resp, body=None)
        ):
            response, error = wal.call_llm_with_retry(mock_messages, mock_tools)
        
        assert response is None
        assert error == "服务繁忙，请稍后重试"

    def test_api_error_no_retry(self, mock_messages, mock_tools):
        """API 错误，不重试直接返回"""
        class MockAPIError(APIError):
            def __init__(self):
                self.message = "server error"
        
        with patch.object(
            wal.client.chat.completions, "create",
            side_effect=MockAPIError()
        ):
            response, error = wal.call_llm_with_retry(mock_messages, mock_tools)
        
        assert response is None
        assert "API 错误" in error

    def test_unknown_exception_no_retry(self, mock_messages, mock_tools):
        """未知异常，不重试直接返回"""
        with patch.object(
            wal.client.chat.completions, "create",
            side_effect=ValueError("unexpected")
        ):
            response, error = wal.call_llm_with_retry(mock_messages, mock_tools)
        
        assert response is None
        assert "未知错误" in error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
