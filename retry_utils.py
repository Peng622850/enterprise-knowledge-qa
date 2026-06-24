# retry_utils.py
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

logger = logging.getLogger(__name__)

def llm_retry(func):
    """
    装饰器：LLM 调用自动重试
    - 最多重试 3 次
    - 指数退避：1s -> 2s -> 4s
    - 只重试网络/超时/限速错误
    """
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        )),
        before_sleep=lambda retry_state: logger.warning(
            f"LLM 调用失败，第 {retry_state.attempt_number} 次重试..."
        ),
        reraise=True,
    )(func)