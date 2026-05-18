# Download the user data released with the paper, place it under ./user_data_{en/zh},
# and then run the data preparation or evaluation scripts.
INPUT_DATA_PATH = "./raw_user_data/zh"
OUTPUT_DIR = "./output"
RESULTS_DIR = "./results"

# Set to a local path or "Qwen/Qwen3-8B"
QWEN_TOKENIZER_MODEL = "Qwen/Qwen3-8B"

# Judge model for evaluating simulated customer-service conversations in E-commerce scenarios
TEACHER_MODEL_NAME = "claude-sonnet-4-5-20250929"

MODELS_TO_EVALUATE = [
    {
        "name": "gpt-5",
        "type": "openai_compatible",
        "api_key": "YOUR_API_KEY",
        "endpoints": [
            {"url": "https://api.openai.com/v1", "max_workers": 5},
        ],
        "model": "gpt-5",
        "temperature": 0.1,
        "use_logprobs": False,
    },
    {
        "name": "Qwen3-235B",
        "type": "openai_compatible",
        "api_key": "sk-dummy",
        "endpoints": [
            {"url": "http://your-server-1:30000/v1", "max_workers": 20},
            {"url": "http://your-server-2:30000/v1", "max_workers": 20},
            {"url": "http://your-server-3:30000/v1", "max_workers": 20},
            {"url": "http://your-server-4:30000/v1", "max_workers": 20},
        ],
        "model": "Qwen3-235B",
        "temperature": 0.1,
        "use_logprobs": True,
        "enable_thinking": False,
    },
    {
        "name": "claude-sonnet-4-5-20250929",
        "type": "openai_compatible",
        "api_key": "YOUR_API_KEY",
        "endpoints": [
            {"url": "https://api.xxx.com/v1", "max_workers": 5},
        ],
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 0.1,
    },
]
