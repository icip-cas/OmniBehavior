import json
import re
import time
import os
import math
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Union
import requests
from openai import OpenAI


def stable_hash(key: str) -> int:
    """Stable MD5-based hash, consistent across runs."""
    return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)


def remove_think_tags(text: str) -> str:
    """Strip <think>/<thinking> tags from model output."""
    if not text:
        return ""

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*?</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</think>', '', text, flags=re.DOTALL)

    # Handle truncated output with no closing tag
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*$', '', text, flags=re.DOTALL)

    return text.strip()


def get_endpoints(model_config: Dict) -> List[Dict]:
    """Extract endpoint list from model config."""
    default_workers = model_config.get("max_workers", 10)
    result = []
    for ep in model_config.get("endpoints", []):
        if isinstance(ep, dict):
            result.append({
                "url": ep["url"],
                "max_workers": ep.get("max_workers", default_workers)
            })
        elif isinstance(ep, str):
            result.append({"url": ep, "max_workers": default_workers})
    return result


def assign_users_to_endpoints(user_ids: List[str], endpoints: List[Dict]) -> Dict[str, List[str]]:
    """Assign users to endpoints by stable hash weighted by max_workers. Same user always maps to the same endpoint for prefix cache reuse."""
    assignment = {ep["url"]: [] for ep in endpoints}

    if not endpoints or not user_ids:
        return assignment

    total_workers = sum(ep["max_workers"] for ep in endpoints)

    # Build cumulative weight list using integers to avoid float precision issues
    cumulative_workers = []
    cumsum = 0
    for ep in endpoints:
        cumsum += ep["max_workers"]
        cumulative_workers.append(cumsum)
    
    for user_id in user_ids:
        h = stable_hash(user_id) % total_workers
        for i, threshold in enumerate(cumulative_workers):
            if h < threshold:
                assignment[endpoints[i]["url"]].append(user_id)
                break
        else:
            assignment[endpoints[-1]["url"]].append(user_id)

    return assignment


class DynamicTaskQueue:
    """
    Work-stealing task queue for multi-endpoint load balancing.

    Each endpoint has a local queue (initially assigned by user hash to exploit prefix caching).
    When a local queue is empty, tasks are stolen from the busiest other endpoint.
    """

    def __init__(self, endpoints: List[Dict], tasks: List[Dict], user_tasks: Dict[str, List[Dict]]):
        self.endpoints = endpoints
        self.user_tasks = user_tasks
        self.lock = threading.Lock()

        user_ids = list(user_tasks.keys())
        initial_assignment = assign_users_to_endpoints(user_ids, endpoints)

        from collections import deque
        self.endpoint_queues: Dict[str, deque] = {}
        for ep in endpoints:
            url = ep["url"]
            users = initial_assignment.get(url, [])
            local_tasks = []
            for uid in users:
                local_tasks.extend(user_tasks.get(uid, []))
            self.endpoint_queues[url] = deque(local_tasks)

        self.total_tasks = len(tasks)
        self.completed_count = 0
        self.stolen_count = 0
        self.endpoint_stats = {ep["url"]: {"local": 0, "stolen": 0} for ep in endpoints}

        print(f"\nLoad balancer initialized:")
        for ep in endpoints:
            url = ep["url"]
            queue_size = len(self.endpoint_queues[url])
            print(f"   [{url}] initial tasks: {queue_size}, workers: {ep['max_workers']}")
    
    def get_task(self, endpoint_url: str) -> Optional[Dict]:
        """Get a task for the given endpoint, stealing from the busiest peer if local queue is empty."""
        with self.lock:
            local_queue = self.endpoint_queues.get(endpoint_url)
            if local_queue and len(local_queue) > 0:
                task = local_queue.popleft()
                self.endpoint_stats[endpoint_url]["local"] += 1
                return task

            # Local queue empty — steal from the endpoint with the most pending tasks
            max_queue_url = None
            max_queue_size = 0
            for ep in self.endpoints:
                url = ep["url"]
                if url != endpoint_url:
                    queue = self.endpoint_queues[url]
                    if len(queue) > max_queue_size:
                        max_queue_size = len(queue)
                        max_queue_url = url

            if max_queue_url and max_queue_size > 0:
                task = self.endpoint_queues[max_queue_url].popleft()
                self.stolen_count += 1
                self.endpoint_stats[endpoint_url]["stolen"] += 1
                return task

            return None
    
    def mark_completed(self):
        """Mark one task as completed."""
        with self.lock:
            self.completed_count += 1
    
    def get_remaining_count(self) -> int:
        """Return total tasks still in all queues."""
        with self.lock:
            total_in_queues = sum(len(q) for q in self.endpoint_queues.values())
            return total_in_queues
    
    def is_all_done(self) -> bool:
        """Return True if all queues are empty."""
        with self.lock:
            return all(len(q) == 0 for q in self.endpoint_queues.values())
    
    def get_stats(self) -> Dict:
        """Return queue and completion statistics."""
        with self.lock:
            return {
                "total_tasks": self.total_tasks,
                "completed": self.completed_count,
                "stolen_count": self.stolen_count,
                "stolen_rate": self.stolen_count / self.total_tasks * 100 if self.total_tasks > 0 else 0,
                "endpoint_stats": dict(self.endpoint_stats),
                "remaining_per_endpoint": {url: len(q) for url, q in self.endpoint_queues.items()}
            }


class ModelCaller:
    """Unified model calling interface."""

    def __init__(self, model_config: Dict, base_url_override: str = None, error_log_dir: str = None):
        if not model_config:
            raise ValueError("model_config is required")
        
        self.config = model_config
        self.model_type = model_config.get("type", "openai_compatible")
        self.model_name = model_config.get("name", model_config.get("model", "unknown"))
        self.use_logprobs = model_config.get("use_logprobs", False)
        self.max_tokens = model_config.get("max_tokens")
        
        # check if thinking mode is enabled
        self.enable_thinking = model_config.get("enable_thinking", False)
                
        # thinking mode: disable logprobs (answer follows the think block)
        if self.enable_thinking:
            self.use_logprobs = False
            
        self.use_cache_control = model_config.get("use_cache_control", False)
        self.error_log_dir = error_log_dir

        if base_url_override:
            self.base_url = base_url_override
        else:
            endpoints = get_endpoints(model_config)
            self.base_url = endpoints[0]["url"] if endpoints else None
        
        self._error_log_lock = threading.Lock()

    def set_error_log_dir(self, error_log_dir: str):
        """Set the error log directory."""
        self.error_log_dir = error_log_dir

    def _log_error(self, prompt: str, raw_output: str, error_msg: str):
        """Append a failed-prompt entry to the error log."""
        if not self.error_log_dir:
            return
            
        try:
            log_file = os.path.join(self.error_log_dir, "failed_prompts.jsonl")
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.model_name,
                "error": error_msg,
                "raw_output": raw_output,
                "prompt": prompt
            }
            with self._error_log_lock:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"FAILED to write error log: {e}")

    
    def call(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call the model; return response text or None on failure."""
        for attempt in range(max_retries):
            try:
                if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                    response_data = self._call_openai_compatible(prompt)
                    response = response_data["content"]
                elif self.model_type == "anthropic":
                    response = self._call_anthropic(prompt)
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")

                return response

            except Exception as e:
                print(f"[{self.model_name}] call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    return None
        
        return None
    
    def _call_openai_compatible(self, prompt: str, enable_logprobs: bool = False, max_tokens: int = None) -> Union[str, Dict]:
        """Call OpenAI-compatible API; returns {"content", "logprobs", "usage"}."""
        client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.base_url,
            timeout=36000
        )
        
        if self.use_cache_control:
            message_content = [
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        else:
            message_content = prompt
        
        request_params = {
            "model": self.config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            "temperature": self.config.get("temperature", 0.1),
        }
        
        if enable_logprobs:
            request_params["logprobs"] = True
            request_params["top_logprobs"] = 5
        
        model_name_lower = self.model_name.lower()
        is_reasoning_model = self.config.get("is_reasoning_model", False) or any(
            name in model_name_lower for name in ["gpt-5", "gpt5", "o1", "o3", "reasoning"]
        )
        
        if max_tokens is not None:
            if is_reasoning_model:
                request_params["max_completion_tokens"] = max_tokens
            else:
                request_params["max_tokens"] = max_tokens

        if is_reasoning_model:
            request_params["reasoning_effort"] = self.config.get("reasoning_effort", "minimal")
        
        extra_body = self._build_extra_body()
        if extra_body:
            request_params["extra_body"] = extra_body
        
        response = client.chat.completions.create(**request_params)

        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and choice.message:
                content = choice.message.content or ""
            else:
                content = ""
        else:
            content = ""
        
        usage_data = {}
        if response.usage:
            try:
                if hasattr(response.usage, 'model_dump'):
                    usage_data = response.usage.model_dump()
                elif hasattr(response.usage, '__dict__'):
                    usage_data = dict(response.usage.__dict__)
                else:
                    usage_data = {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0),
                    }
            except Exception:
                usage_data = {}
        
        cached_tokens = 0
        prompt_tokens_details = usage_data.get("prompt_tokens_details") or {}
        if isinstance(prompt_tokens_details, dict) and prompt_tokens_details:
            cached_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0
        usage_data["cached_tokens"] = cached_tokens
        
        if enable_logprobs:
            logprobs_data = None
            if response.choices and len(response.choices) > 0 and response.choices[0].logprobs:
                try:
                    if hasattr(response.choices[0].logprobs, 'model_dump'):
                        logprobs_data = response.choices[0].logprobs.model_dump()
                    elif hasattr(response.choices[0].logprobs, '__dict__'):
                        logprobs_data = dict(response.choices[0].logprobs.__dict__)
                    else:
                        logprobs_data = response.choices[0].logprobs
                except Exception:
                    logprobs_data = None
            return {
                "content": content,
                "logprobs": logprobs_data,
                "usage": usage_data
            }
        
        return {"content": content, "usage": usage_data}

    def _build_extra_body(self) -> Dict:
        """Build request extra_body from model config."""
        if "enable_thinking" not in self.config:
            return {}

        return {
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
            }
        }
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config['api_key'],
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens or 1024,
            "temperature": self.config.get("temperature", 0.7),
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result["content"][0]["text"]
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Return True if the error looks like a rate limit."""
        error_str = str(error).lower()
        rate_limit_keywords = [
            "rate limit", "ratelimit", "rate_limit", "RateLimitError",
            "429", "too many requests", "quota exceeded",
            "throttl", "limit exceeded", "请求过于频繁"
        ]
        return any(keyword in error_str for keyword in rate_limit_keywords)
    
    def call_binary_classification(self, prompt: str, max_retries: int = 6) -> Dict:
        """Binary (Yes/No) prediction; uses logprobs softmax for open-source models, direct mapping for closed-source."""
        result = {
            "success": False,
            "prediction": None,
            "predicted_label": None,
            "raw_output": "",
            "method": "logprobs" if self.use_logprobs else "direct_mapping",
            "retry_count": 0,
            "error": None,
            "logprob_yes": None,
            "logprob_no": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }

        for attempt in range(max_retries):
            result["retry_count"] = attempt

            try:
                if self.use_logprobs:
                    prediction_result = self._call_binary_with_logprobs(prompt)
                    if prediction_result["success"]:
                        result.update(prediction_result)
                        return result
                    else:
                        result["raw_output"] = prediction_result.get("raw_output", "")
                        result["error"] = prediction_result.get("error", "Unknown error")
                else:
                    prediction_result = self._call_binary_direct_mapping(prompt)
                    if prediction_result["success"]:
                        result.update(prediction_result)
                        return result
                    else:
                        result["raw_output"] = prediction_result.get("raw_output", "")
                        result["error"] = prediction_result.get("error", "Unknown error")

                if attempt < max_retries - 1:
                    if attempt == 0 or (attempt + 1) % 3 == 0:
                        raw_output_preview = result.get("raw_output", "")[:50]
                        print(f"[{self.model_name}] invalid binary output (attempt {attempt + 1}/{max_retries}): '{raw_output_preview}...'")

            except Exception as e:
                result["error"] = str(e)
                
                if attempt < max_retries - 1:
                    if self._is_rate_limit_error(e):
                        wait_time = 10
                        print(f"[{self.model_name}] rate limit hit, retrying in {wait_time}s: {e}")
                    else:
                        wait_time = 1
                        print(f"[{self.model_name}] binary call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    time.sleep(wait_time)
        
        result["retry_count"] = max_retries
        error_info = result.get("error", "unknown error")
        raw_output_preview = result.get("raw_output", "")[:100] if result.get("raw_output") else "no output"
        print(f"[{self.model_name}] binary call exhausted {max_retries} retries — reason: {error_info}, output: {raw_output_preview}")
        return result

    def call_continuous_prediction(self, prompt: str, max_retries: int = 5) -> Dict:
        """Continuous value prediction (e.g. watch duration); parses numeric output."""
        result = {
            "success": False,
            "prediction": None,
            "raw_output": "",
            "method": "continuous",
            "retry_count": 0,
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        
        for attempt in range(max_retries):
            result["retry_count"] = attempt

            try:
                if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                    max_t = None if self.enable_thinking else 200
                    response_data = self._call_openai_compatible(prompt, enable_logprobs=False, max_tokens=max_t)
                elif self.model_type == "anthropic":
                    response_data = self._call_anthropic(prompt)
                else:
                    result["error"] = f"Unsupported model type: {self.model_type}"
                    return result

                content = response_data["content"]
                usage_data = response_data.get("usage", {})
                result["raw_output"] = content
                result["prompt_tokens"] = usage_data.get("prompt_tokens", 0)
                result["completion_tokens"] = usage_data.get("completion_tokens", 0)
                result["cached_tokens"] = usage_data.get("cached_tokens", 0)

                content_without_think = remove_think_tags(content)
                number = self._extract_number_from_text(content_without_think)

                if number is not None:
                    result["success"] = True
                    result["prediction"] = number
                    return result
                else:
                    content_preview = content[:100] + '...' if len(content) > 100 else content
                    result["error"] = f"no number found in output: '{content_preview}'"

                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    if attempt == 0 or (attempt + 1) % 3 == 0:
                        raw_output_preview = result.get("raw_output", "")[:50]
                        print(f"[{self.model_name}] invalid continuous output (attempt {attempt + 1}/{max_retries}): '{raw_output_preview}...'")
                    time.sleep(wait_time)

            except Exception as e:
                result["error"] = str(e)

                if attempt < max_retries - 1:
                    if self._is_rate_limit_error(e):
                        wait_time = 10
                        print(f"[{self.model_name}] rate limit hit, retrying in {wait_time}s: {e}")
                    else:
                        wait_time = 1
                        print(f"[{self.model_name}] continuous call failed (attempt {attempt + 1}/{max_retries}): {e}")

                    time.sleep(wait_time)

        result["retry_count"] = max_retries
        error_info = result.get("error", "unknown error")
        raw_output_preview = result.get("raw_output", "")[:100] if result.get("raw_output") else "no output"
        print(f"[{self.model_name}] continuous call exhausted {max_retries} retries — reason: {error_info}, output: {raw_output_preview}")
        return result
    
    def call_text_prediction(self, prompt: str, max_retries: int = 5) -> Dict:
        """Text prediction (e.g. search keywords); returns cleaned model output."""
        result = {
            "success": False,
            "prediction": None,
            "raw_output": "",
            "method": "text",
            "retry_count": 0,
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        
        for attempt in range(max_retries):
            result["retry_count"] = attempt
            
            try:
                if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                    max_t = None if self.enable_thinking else 200
                    response_data = self._call_openai_compatible(prompt, enable_logprobs=False, max_tokens=max_t)
                elif self.model_type == "anthropic":
                    response_data = self._call_anthropic(prompt)
                else:
                    result["error"] = f"Unsupported model type: {self.model_type}"
                    return result

                content = response_data["content"]
                usage_data = response_data.get("usage", {})
                result["raw_output"] = content
                result["prompt_tokens"] = usage_data.get("prompt_tokens", 0)
                result["completion_tokens"] = usage_data.get("completion_tokens", 0)
                result["cached_tokens"] = usage_data.get("cached_tokens", 0)

                content_without_think = remove_think_tags(content)
                cleaned_text = self._clean_text_output(content_without_think)

                if cleaned_text:
                    result["success"] = True
                    result["prediction"] = cleaned_text
                    return result
                else:
                    content_preview = content[:100] + '...' if len(content) > 100 else content
                    result["error"] = f"empty or invalid output: '{content_preview}'"

                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    if attempt == 0 or (attempt + 1) % 3 == 0:
                        raw_output_preview = result.get("raw_output", "")[:50]
                        print(f"[{self.model_name}] invalid text output (attempt {attempt + 1}/{max_retries}): '{raw_output_preview}...'")
                    time.sleep(wait_time)

            except Exception as e:
                result["error"] = str(e)

                if attempt < max_retries - 1:
                    if self._is_rate_limit_error(e):
                        wait_time = 10
                        print(f"[{self.model_name}] rate limit hit, retrying in {wait_time}s: {e}")
                    else:
                        wait_time = 1
                        print(f"[{self.model_name}] text call failed (attempt {attempt + 1}/{max_retries}): {e}")

                    time.sleep(wait_time)

        result["retry_count"] = max_retries
        error_info = result.get("error", "unknown error")
        raw_output_preview = result.get("raw_output", "")[:100] if result.get("raw_output") else "no output"
        print(f"[{self.model_name}] text call exhausted {max_retries} retries — reason: {error_info}, output: {raw_output_preview}")
        return result
    
    def _clean_text_output(self, text: str) -> str:
        """Strip quotes and common prefixes from text output."""
        if not text:
            return ""
        
        text = text.strip()
        
        quotes = ['"', "'", '"', '"', ''', ''', '「', '」', '『', '』']
        for quote in quotes:
            if text.startswith(quote) and text.endswith(quote):
                text = text[1:-1].strip()
                break
            elif text.startswith(quote):
                text = text[1:].strip()
            elif text.endswith(quote):
                text = text[:-1].strip()
        
        prefixes_to_remove = [
            "搜索关键词：", "搜索关键词:", "关键词：", "关键词:",
            "搜索：", "搜索:", "Search:", "Keyword:", "Keywords:",
        ]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
        
        return text
    
    def _extract_number_from_text(self, text: str) -> Optional[float]:
        """Extract a number from text; supports seconds/minutes units."""
        if not text:
            return None
        
        text = text.strip()

        try:
            return float(text)
        except ValueError:
            pass

        patterns = [
            r'(\d+\.?\d*)\s*秒',        # "42秒" or "42.5秒"
            r'(\d+\.?\d*)\s*分钟',       # "3分钟" or "3.5分钟" (convert to seconds)
            r'(\d+\.?\d*)\s*s(?:econds?)?',  # "42s" or "42 seconds"
            r'(\d+\.?\d*)\s*min(?:utes?)?',  # "3min" or "3 minutes" (convert to seconds)
            r'^(\d+\.?\d*)$',            # plain number
            r'(\d+\.?\d*)',              # any number (last-resort fallback)
        ]

        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = float(match.group(1))
                if i == 1 or i == 3:  # convert minutes to seconds
                    number *= 60
                return number
        
        return None
    
    def _call_binary_with_logprobs(self, prompt: str) -> Dict:
        """Binary prediction using logprobs softmax (open-source models)."""
        result = {
            "success": False,
            "prediction": None,
            "predicted_label": None,
            "raw_output": "",
            "logprob_yes": None,
            "logprob_no": None,
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }

        try:
            if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                max_t = 4096 if getattr(self, "enable_thinking", False) else 200
                response_data = self._call_openai_compatible(prompt, enable_logprobs=True, max_tokens=max_t)
            else:
                # fall back to direct mapping for model types without logprobs
                result["error"] = f"model type {self.model_type} does not support logprobs"
                return result
            
            content = response_data["content"]
            logprobs_data = response_data["logprobs"]
            usage_data = response_data.get("usage", {})
            result["raw_output"] = content
            result["prompt_tokens"] = usage_data.get("prompt_tokens", 0)
            result["completion_tokens"] = usage_data.get("completion_tokens", 0)
            result["cached_tokens"] = usage_data.get("cached_tokens", 0)

            # Strip think tags before checking Yes/No (compatibility fallback — logprobs disabled when thinking is on)
            content = remove_think_tags(content)

            content_stripped = content.strip()
            import re as _re
            content_cleaned = _re.sub(r'[*_~`.]', '', content_stripped).strip()
            if content_cleaned.lower() not in ["yes", "no"]:
                result["error"] = f"output is not Yes/No: '{content_cleaned}'"
                return result
            content_stripped = "Yes" if content_cleaned.lower() == "yes" else "No"

            if logprobs_data and "content" in logprobs_data and len(logprobs_data["content"]) > 0:
                top_logprobs = logprobs_data["content"][0].get("top_logprobs", [])
                
                logprob_yes = None
                logprob_no = None
                
                for item in top_logprobs:
                    token = item.get("token", "")
                    logprob = item.get("logprob", None)
                    
                    if token == "Yes":
                        logprob_yes = logprob
                    elif token == "No":
                        logprob_no = logprob
                    
                    if logprob_yes is not None and logprob_no is not None:
                        break
                
                result["logprob_yes"] = logprob_yes
                result["logprob_no"] = logprob_no
                result["predicted_label"] = 1 if content_stripped == "Yes" else 0

                if logprob_yes is not None and logprob_no is not None:
                    exp_yes = math.exp(logprob_yes)
                    exp_no = math.exp(logprob_no)
                    p_yes = exp_yes / (exp_yes + exp_no)
                    result["success"] = True
                    result["prediction"] = p_yes
                else:
                    # Fall back to direct mapping when both logprobs not found in top_logprobs
                    result["error"] = "Yes/No logprob not found in top_logprobs"
                    result["success"] = True
                    result["prediction"] = 1.0 if content_stripped == "Yes" else 0.0
            else:
                # logprobs data incomplete — fall back to direct mapping
                result["error"] = "incomplete logprobs data"
                result["predicted_label"] = 1 if content_stripped == "Yes" else 0
                result["success"] = True
                result["prediction"] = 1.0 if content_stripped == "Yes" else 0.0
                    
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _call_binary_direct_mapping(self, prompt: str) -> Dict:
        """Binary prediction via direct Yes/No mapping (closed-source models)."""
        result = {
            "success": False,
            "prediction": None,
            "predicted_label": None,
            "raw_output": "",
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }

        try:
            if self.model_type in ["openai", "deepseek", "qwen", "openai_compatible"]:
                max_t = 4096 if getattr(self, "enable_thinking", False) else 50
                response_data = self._call_openai_compatible(prompt, enable_logprobs=False, max_tokens=max_t)
                response = response_data["content"]
                usage_data = response_data.get("usage", {})
                result["prompt_tokens"] = usage_data.get("prompt_tokens", 0) or 0
                result["completion_tokens"] = usage_data.get("completion_tokens", 0) or 0
                result["cached_tokens"] = usage_data.get("cached_tokens", 0) or 0
            elif self.model_type == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                result["error"] = f"Unsupported model type: {self.model_type}"
                return result

            result["raw_output"] = response
            response = remove_think_tags(response)
            content_cleaned = self._clean_text_output(response)
            content_compact = "".join(content_cleaned.split()).lower()

            if content_compact == "yes":
                result["success"] = True
                result["prediction"] = 1.0
                result["predicted_label"] = 1
                return result
            elif content_compact == "no":
                result["success"] = True
                result["prediction"] = 0.0
                result["predicted_label"] = 0
                return result

            if content_cleaned.lower().startswith("yes"):
                result["success"] = True
                result["prediction"] = 1.0
                result["predicted_label"] = 1
                return result
            elif content_cleaned.lower().startswith("no"):
                result["success"] = True
                result["prediction"] = 0.0
                result["predicted_label"] = 0
                return result

            # Search only the first 20 chars to avoid matching incidental yes/no in longer text
            head_part = content_cleaned[:20].lower()
            if "yes" in head_part and "no" not in head_part:
                result["success"] = True
                result["prediction"] = 1.0
                result["predicted_label"] = 1
                return result
            elif "no" in head_part and "yes" not in head_part:
                result["success"] = True
                result["prediction"] = 0.0
                result["predicted_label"] = 0
                return result

            import re
            if re.search(r'\b(yes|YES|Yes)\b', content_cleaned):
                 result["success"] = True
                 result["prediction"] = 1.0
                 result["predicted_label"] = 1
            elif re.search(r'\b(no|NO|No)\b', content_cleaned):
                 result["success"] = True
                 result["prediction"] = 0.0
                 result["predicted_label"] = 0
            else:
                error_msg = f"output is not Yes/No: '{content_cleaned[:50]}...'"
                result["error"] = error_msg
                self._log_error(prompt, response, error_msg)
                
        except Exception as e:
            result["error"] = str(e)
            self._log_error(prompt, response if 'response' in locals() else "NO_RESPONSE", str(e))

        return result


def parse_model_response(response_text: str, num_questions: int, debug: bool = False) -> Dict:
    """Parse model JSON response into {answer_1: val, ..., answer_N: val}. Returns None for unparseable keys."""
    if debug:
        print(f"\n[DEBUG] raw response:\n{response_text[:500]}...\n")
        
    response_text = remove_think_tags(response_text)

    # Try to extract JSON: markdown-wrapped, plain code block, or bare object
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    
    if json_match:
        try:
            json_str = json_match.group(1) if json_match.lastindex and json_match.lastindex >= 1 else json_match.group(0)
            
            if debug:
                print(f"[DEBUG] extracted JSON:\n{json_str}\n")
            
            json_str = re.sub(r'//.*?\n', '\n', json_str)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

            result = json.loads(json_str)

            parsed = {}
            for i in range(1, num_questions + 1):
                key = f"answer_{i}"
                if key in result:
                    value = result[key]
                    if isinstance(value, str):
                        parsed[key] = value
                    else:
                        try:
                            parsed[key] = float(value)
                        except (ValueError, TypeError):
                            parsed[key] = str(value)
                else:
                    if debug:
                        print(f"[DEBUG] Warning: missing {key}")
                    parsed[key] = None
            
            if debug:
                print(f"[DEBUG] parsed successfully: {parsed}\n")
            
            return parsed
            
        except json.JSONDecodeError as e:
            if debug:
                print(f"[DEBUG] JSON parse failed: {e}\n")
    
    # Fallback: regex extraction
    if debug:
        print("[DEBUG] falling back to regex extraction\n")

    parsed = {}
    for i in range(1, num_questions + 1):
        pattern_str = rf'"?answer_{i}"?\s*:\s*"([^"]+)"'
        match = re.search(pattern_str, response_text)
        if match:
            parsed[f"answer_{i}"] = match.group(1)
            if debug:
                print(f"[DEBUG] answer_{i} = {match.group(1)} (string)")
        else:
            pattern_num = rf'"?answer_{i}"?\s*:\s*([0-9.]+)'
            match = re.search(pattern_num, response_text)
            if match:
                try:
                    parsed[f"answer_{i}"] = float(match.group(1))
                    if debug:
                        print(f"[DEBUG] answer_{i} = {match.group(1)} (number)")
                except ValueError:
                    parsed[f"answer_{i}"] = None
                    if debug:
                        print(f"[DEBUG] answer_{i} = None (conversion failed)")
            else:
                parsed[f"answer_{i}"] = None
                if debug:
                    print(f"[DEBUG] answer_{i} = None (not found)")
    
    return parsed
