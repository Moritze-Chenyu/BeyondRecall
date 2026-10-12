"""Public API configuration: reads environment variables, contains no credentials.

Supports the historical launchers' variable names. A provider may require
different model names or parameters; configure those explicitly before running.
"""
import os


def setting(name, default=""):
    return os.getenv(name) or default

ANSWER_API_KEY = setting('BR_API_KEY', '')
ANSWER_BASE_URL = setting('BR_API_BASE_URL', '')
ANSWER_MODEL = setting('BR_ANSWER_MODEL', 'Qwen/Qwen3-14B')
ANSWER_MAX_TOKENS = int(setting('BR_ANSWER_MAX_TOKENS', '1024'))
MEM0_API_KEY = setting('BR_FRAME_MEMORY_KEY', ANSWER_API_KEY)
MEM0_BASE_URL = setting('BR_FRAME_MEMORY_URL', ANSWER_BASE_URL)
MEM0_MODEL = setting('BR_MEMORY_MODEL', 'Qwen/Qwen3-32B')
MEM0_EVAL_EMBED_MODEL = setting('BR_EMBED_MODEL', 'BAAI/bge-small-en-v1.5')
FULL_CONTEXT_CAPTION_MODEL = ANSWER_MODEL
FULL_CONTEXT_MULTIMODAL_API_KEY = setting('BR_MULTIMODAL_API_KEY', ANSWER_API_KEY)
FULL_CONTEXT_MULTIMODAL_BASE_URL = setting('BR_MULTIMODAL_BASE_URL', ANSWER_BASE_URL)
FULL_CONTEXT_MULTIMODAL_MODEL = setting('BR_MULTIMODAL_MODEL', 'Qwen/Qwen3.5-9B')
FULL_CONTEXT_MULTIMODAL_ENABLE_THINKING = setting('BR_ENABLE_THINKING', 'false').lower() == 'true'
VISION_API_KEY = setting('BR_CAPTION_KEY', FULL_CONTEXT_MULTIMODAL_API_KEY)
VISION_BASE_URL = setting('BR_CAPTION_BASE_URL', FULL_CONTEXT_MULTIMODAL_BASE_URL)
VISION_MODEL = setting('BR_CAPTION_MODEL', FULL_CONTEXT_MULTIMODAL_MODEL)
FULL_CONTEXT_JUDGE_API_KEY = setting('BR_JUDGE_API_KEY', ANSWER_API_KEY)
FULL_CONTEXT_JUDGE_BASE_URL = setting('BR_JUDGE_BASE_URL', ANSWER_BASE_URL)
FULL_CONTEXT_JUDGE_MODEL = setting('BR_JUDGE_MODEL', ANSWER_MODEL)
