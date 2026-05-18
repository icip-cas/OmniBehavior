import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from action_types import get_action_type, get_record_action_type, normalize_action_type_set
from data.data_processor import load_user_data, filter_users_by_month
from models.prompt_builder import build_single_binary_prompt, build_single_continuous_prompt, build_single_text_prompt, get_binary_questions_for_action, get_all_questions_for_action, get_actual_used_history
from models.model_caller import ModelCaller, get_endpoints, assign_users_to_endpoints, stable_hash, DynamicTaskQueue, remove_think_tags
from metrics import calculate_all_binary_metrics, calculate_all_continuous_metrics, calculate_micro_macro_f1
from datetime import datetime
import pytz

import numpy as np


DEFAULT_EXPERIMENT_DATA_PATH = "./dataset/experiment_data.json"


def round_floats(obj, decimals=2):
    """Recursively round floats and convert NaN/Inf to None for JSON compatibility."""
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), decimals)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    else:
        return obj


def convert_metrics_to_percentage(metrics: Dict) -> Dict:
    """Convert 0-1 range metrics (F1, AUC, etc.) to percentage (0-100). Excludes loss/error metrics."""
    import copy
    metrics_copy = copy.deepcopy(metrics)
    
    # metric name keywords that should be multiplied by 100
    percentage_keywords = [
        'F1', 'Precision', 'Recall', 'AUC', 'Accuracy', 'ECE',
        'success_rate', 'positive_rate', 'Micro_', 'Macro_', 'R²'
    ]
    
    # metrics to exclude even if their value is in 0-1 range
    exclude_keywords = [
        'LogLoss', 'Loss', 'MAE', 'MSE', 'RMSE', 'mean_', 'count', 'TP', 'FP', 'FN', 'TN', 'relative_accuracy', 'symmetry_consistency'
    ]
    
    def should_convert(key: str) -> bool:
        """Return True if this metric should be converted to percentage."""
        if any(exclude in key for exclude in exclude_keywords):
            return False
        return any(keyword in key for keyword in percentage_keywords)
    
    def convert_dict(d: Dict) -> Dict:
        """Recursively convert percentage values in dict."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = convert_dict(value)
            elif isinstance(value, (float, np.floating)) and should_convert(key):
                if not np.isnan(value) and not np.isinf(value) and 0 <= value <= 1:
                    result[key] = float(value * 100)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    if 'overall' in metrics_copy:
        metrics_copy['overall'] = convert_dict(metrics_copy['overall'])
    
    if 'binary_metrics' in metrics_copy:
        for field in metrics_copy['binary_metrics']:
            metrics_copy['binary_metrics'][field] = convert_dict(metrics_copy['binary_metrics'][field])
    
    if 'continuous_metrics' in metrics_copy:
        for field in metrics_copy['continuous_metrics']:
            metrics_copy['continuous_metrics'][field] = convert_dict(metrics_copy['continuous_metrics'][field])
    
    return metrics_copy


FINAL_SCORE_BINARY_FIELDS = {
    "video_binary_score": [
        "video_completed",
        "video_liked",
        "video_commented",
        "video_shared",
        "video_collected",
        "video_followed",
    ],
    "live_binary_score": [
        "live_liked",
        "live_commented",
        "live_sent_gift",
        "live_followed",
        "live_shared",
        "live_clicked_cart",
    ],
    "ad_binary_score": [
        "ad_liked",
        "ad_commented",
        "ad_activated",
        "ad_form_submitted",
    ],
    "shop_binary_score": [
        "shop_added_to_cart",
        "shop_order_success",
    ],
}


SCORE_SUMMARY_COMPONENT_KEYS = [
    ("video_binary", "video_binary_score"),
    ("video_continuous", "video_continuous_score"),
    ("live_binary", "live_binary_score"),
    ("ads_binary", "ad_binary_score"),
    ("e_commerce_binary", "shop_binary_score"),
    ("e_commerce_textual", "shop_text_score"),
]


def _safe_mean(values: List[float]) -> Optional[float]:
    valid_values = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(value) or np.isinf(value):
            continue
        valid_values.append(value)
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def calculate_final_score_components(
    binary_metrics: Dict,
    continuous_metrics: Dict,
    llm_judge_metrics: Dict,
) -> Dict:
    """
    Compute the final model score on a 0-100 scale.

    Binary scene scores are the mean F1 of that scene's behavior fields.
    Video continuous score is 100 - NMAE.
    Shop text score is the LLM judge average score scaled from 1-10 to 0-100.
    The final score is the mean of all component scores; if any component is missing it is None.
    """
    components = {}
    missing_components = []

    for component_name, fields in FINAL_SCORE_BINARY_FIELDS.items():
        f1_values = [
            binary_metrics[field].get("F1")
            for field in fields
            if field in binary_metrics
        ]
        mean_f1 = _safe_mean(f1_values)
        if mean_f1 is None:
            components[component_name] = None
            missing_components.append(component_name)
        else:
            components[component_name] = mean_f1 * 100

    video_metrics = continuous_metrics.get("video_watch_seconds", {})
    video_nmae = video_metrics.get("NMAE")
    if video_nmae is None:
        components["video_continuous_score"] = None
        missing_components.append("video_continuous_score")
    else:
        try:
            video_nmae = float(video_nmae)
            if np.isnan(video_nmae) or np.isinf(video_nmae):
                raise ValueError
            components["video_continuous_score"] = max(0.0, min(100.0, 100.0 - video_nmae))
        except (TypeError, ValueError):
            components["video_continuous_score"] = None
            missing_components.append("video_continuous_score")

    judge_avg = llm_judge_metrics.get("average_score")
    if judge_avg is None:
        components["shop_text_score"] = None
        missing_components.append("shop_text_score")
    else:
        try:
            judge_avg = float(judge_avg)
            if np.isnan(judge_avg) or np.isinf(judge_avg):
                raise ValueError
            components["shop_text_score"] = max(0.0, min(100.0, judge_avg * 10))
        except (TypeError, ValueError):
            components["shop_text_score"] = None
            missing_components.append("shop_text_score")

    final_score = None if missing_components else _safe_mean(components.values())

    return {
        "final_score": final_score,
        "components": components,
        "available_component_count": sum(1 for v in components.values() if v is not None),
        "missing_components": missing_components,
        "scale": "0-100",
        "notes": {
            "binary_scene_scores": "mean F1 of configured behavior fields, scaled to 0-100",
            "video_continuous_score": "100 - NMAE",
            "shop_text_score": "LLM judge average_score scaled from 1-10 to 0-100",
        },
    }


def build_score_summary(model_name: str, final_score: Dict) -> Dict:
    """Build one table-row-shaped score summary for console and JSON reports."""
    components = final_score.get("components", {})
    score_summary = {"model": model_name}
    for output_key, component_key in SCORE_SUMMARY_COMPONENT_KEYS:
        score_summary[output_key] = components.get(component_key)
    score_summary["overall_score"] = final_score.get("final_score")
    return score_summary


# ---------------------------------------------------------------------------
# LLM Judge functions for e-commerce service dialogue evaluation
# ---------------------------------------------------------------------------


def _llm_judge_check_user_history(record: Dict) -> bool:
    """Check if prior user turns exist in the dialogue before the target message."""
    questions = record.get("questions", [])
    target_text = None
    for q in questions:
        if q.get("type") == "text":
            target_text = q.get("true_value")
            break

    if not target_text:
        return False

    scene_info = record.get("scene_info", {})
    dialogue_content = []
    for action in scene_info.get("action", []):
        if action.get("type") == "dialogue":
            dialogue_content = action.get("content", [])
            break

    if not dialogue_content:
        return False

    last_match_idx = -1
    for i, msg in enumerate(dialogue_content):
        if msg.get("role") == "user" and msg.get("content") == target_text:
            last_match_idx = i

    if last_match_idx == -1:
        return False

    return any(
        m.get("role") == "user"
        for m in dialogue_content[:last_match_idx]
    )


def _llm_judge_extract_inputs(record: Dict) -> Optional[Dict]:
    """Extract user_profile, context, ground_truth, model_output from a record."""
    import re as _re
    if get_record_action_type(record) != "电商客服对话":
        return None

    prompts = record.get("prompts", [])
    if not prompts:
        return None
    full_prompt = prompts[0]

    m = _re.search(r"## 输入一：用户画像\s*\n(.*?)\n## 输入二：", full_prompt, _re.DOTALL)
    if not m:
        m = _re.search(r"## 输入一：用户画像\n(.*?)\n## 输入二：", full_prompt, _re.DOTALL)
    user_profile = m.group(1).strip() if m else ""

    m = _re.search(r"(## 输入三：.*?)\n+请你站在这个用户的角度", full_prompt, _re.DOTALL)
    context = m.group(1).strip() if m else ""

    questions = record.get("questions", [])
    if not questions:
        return None

    ground_truth = ""
    model_output = ""
    for q in questions:
        if q.get("type") == "text":
            ground_truth = q.get("true_value", "")
            model_output = q.get("predicted_value", "")
            break

    if not ground_truth or not model_output:
        if not model_output and record.get("model_responses"):
            model_output = record["model_responses"][0]
        if not ground_truth or not model_output:
            return None

    return {
        "user_profile": user_profile,
        "context": context,
        "ground_truth": ground_truth,
        "model_output": model_output,
    }


def _llm_judge_construct_prompt(inputs: Dict) -> str:
    """Construct the LLM judge evaluation prompt."""
    return f"""# Role
你是一个高精度的"用户模拟评估系统"。你的任务是量化评估 [Model Output] 在模拟特定人类用户时，与 [Ground Truth] 的拟合程度。

# Input Data
1. **User Profile (用户画像)**: {inputs['user_profile']}
2. **Current Context (当前状态)**: {inputs['context']}
3. **Ground Truth (真实用户回复)**: {inputs['ground_truth']}
4. **Model Output (模拟器预测)**: {inputs['model_output']}

# Evaluation Metrics (评分维度 1-10分)

请严格基于以下标准进行打分：

1. **Intent Fidelity (意图还原度)**
   - **核心问题**: 模型是否做出了与真实用户相同的*决策*？
   - **评分标准**:
     - **10分**: 决策完全一致。
     - **5分**: 大方向一致，但侧重点不同。
     - **1分**: 意图完全背离。

2. **Persona & Tone Mimicry (人设与语气模仿)**
   - **核心问题**: 这句话听起来像是 *这个特定用户* 说的吗？
   - **评分标准**:
     - **10分**: 完美捕捉用户的对话风格、情绪状态以及潜在的性格特征。
     - **1分**: 听起来像一个通用的 AI 助手，或者完全不符合用户画像。

3. **Knowledge Boundary (认知边界一致性)**
   - **核心问题**: 模型是否严格遵守了用户的"所知"与"所不知"？
   - **评分标准**:
     - **10分**: 模型仅利用了Context中用户已知的信息。
     - **1分**: 模型泄露了用户不可能知道的信息，或出现了严重的逻辑幻觉。

4. **Semantic Alignment (语义对齐度)**
   - **核心问题**: 预测文本与真实文本在信息含量和表达意义上是否重合？
   - **评分标准**:
     - **10分**: 意思完全一样，可以互换使用而不改变对话流向。
     - **1分**: 意思完全不同，会导致对话走向不同的分支。

# Constraints
1. **DO NOT** output any reasoning, explanation, or conversational filler.
2. **DO NOT** use markdown code blocks (```json ... ```).
3. **ONLY** return a valid, raw JSON object.

# Output Format (JSON)

{{
  "scores": {{
    "intent_fidelity": <1-10>,
    "persona_mimicry": <1-10>,
    "knowledge_boundary": <1-10>,
    "semantic_alignment": <1-10>
  }}
}}
"""


def _llm_judge_parse_response(response: str) -> Optional[Dict]:
    """Parse and validate the JSON scores from the judge response."""
    import re as _re, json as _json
    if not response:
        return None
    try:
        m = _re.search(r"\{.*\}", response, _re.DOTALL)
        data = _json.loads(m.group(0) if m else response)
        scores = data.get("scores")
        if not scores:
            return None
        required_keys = ["intent_fidelity", "persona_mimicry", "knowledge_boundary", "semantic_alignment"]
        if any(k not in scores for k in required_keys):
            return None
        validated = {}
        for k in required_keys:
            val = int(float(scores[k]))
            if not (1 <= val <= 10):
                return None
            validated[k] = val
        return validated
    except Exception:
        return None


def _llm_judge_evaluate_record(record: Dict, judge: "ModelCaller", max_retries: int) -> Dict:
    """Evaluate a single record with the LLM judge. Thread-safe."""
    if get_record_action_type(record) != "电商客服对话":
        return {"status": "not_target", "record": record}

    inputs = _llm_judge_extract_inputs(record)
    if not inputs:
        return {"status": "not_target", "record": record}

    if not _llm_judge_check_user_history(record):
        record["llm_judge_skipped"] = True
        record["llm_judge_reason"] = "No user history found in context"
        return {"status": "skipped", "record": record}

    judge_prompt = _llm_judge_construct_prompt(inputs)
    scores = None
    last_response = None

    for attempt in range(max_retries):
        response = judge.call(judge_prompt)
        last_response = response
        scores = _llm_judge_parse_response(response)
        if scores:
            break
        time.sleep(1)

    if scores:
        avg_score = round(sum(scores.values()) / len(scores), 5)
        record["llm_judge_scores"] = scores
        record["llm_judge_avg_score"] = avg_score
        return {"status": "evaluated", "record": record}
    else:
        record["llm_judge_error"] = "Failed to get valid scores after retries"
        if last_response:
            record["llm_judge_raw_response"] = last_response
        return {"status": "error", "record": record}


def run_llm_judge_evaluation(
    results: List[Dict],
    judge_model_name: str = None,
    max_retries: int = 3,
    max_workers: int = None,
) -> List[Dict]:
    """Run LLM judge evaluation on 电商客服对话 records and annotate results in-place."""
    from config import MODELS_TO_EVALUATE as _MODELS
    from config import TEACHER_MODEL_NAME

    if not judge_model_name:
        judge_model_name = TEACHER_MODEL_NAME

    if not judge_model_name:
        print("LLM Judge skipped: no teacher model configured.")
        return results

    judge_config = next((m for m in _MODELS if m["name"] == judge_model_name), None)
    if not judge_config:
        print(f"LLM Judge skipped: model '{judge_model_name}' not found in config")
        return results

    target_indices = [i for i, r in enumerate(results) if get_record_action_type(r) == "电商客服对话"]
    if not target_indices:
        return results

    if max_workers is None:
        max_workers = _resolve_model_workers(judge_config)

    print(f"\nLLM Judge evaluation: {len(target_indices)} e-commerce service dialogue records, concurrency {max_workers}")
    judge = ModelCaller(judge_config)
    evaluated = skipped = errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_llm_judge_evaluate_record, results[i], judge, max_retries): i
            for i in target_indices
        }
        with tqdm(total=len(target_indices), desc="LLM Judge") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    results[idx] = res["record"]
                    if res["status"] == "evaluated":
                        evaluated += 1
                    elif res["status"] == "skipped":
                        skipped += 1
                    elif res["status"] == "error":
                        errors += 1
                except Exception as e:
                    print(f"\nLLM Judge record {idx} error: {e}")
                    errors += 1
                finally:
                    pbar.update(1)

    print(f"LLM Judge done: evaluated {evaluated}, skipped (no history) {skipped}, errors {errors}")
    return results


def _resolve_model_workers(model_config: Dict) -> int:
    """Resolve workers from top-level config or endpoint configs."""
    if model_config.get("max_workers") is not None:
        return model_config["max_workers"]

    endpoints = get_endpoints(model_config)
    if endpoints:
        return sum(ep["max_workers"] for ep in endpoints)

    return 1


def _describe_model_workers(model_config: Dict) -> str:
    """Return a human-readable worker summary for model or endpoint config."""
    if model_config.get("max_workers") is not None:
        return str(model_config["max_workers"])

    endpoints = get_endpoints(model_config)
    if endpoints:
        if len(endpoints) == 1:
            return f"{endpoints[0]['max_workers']} (from endpoint config)"
        total_workers = sum(ep["max_workers"] for ep in endpoints)
        return f"{total_workers} (across {len(endpoints)} endpoints)"

    return "1 (default)"


class UserSimulationEvaluator:
    """User simulation evaluator."""

    def __init__(
        self,
        model_config: Dict,
        experiment_name: str = None,
        run_timestamp: str = None,
        run_date: str = None,
        max_history_tokens: int = None,
        max_history_days: int = None,
        skip_model_eval: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            model_config: model configuration dict
            experiment_name: experiment name
            run_timestamp: run timestamp for unified save path (second precision)
            run_date: run date for date directory
            max_history_tokens: max token limit for history; None uses dataset metadata/default
            max_history_days: keep only history within the last N days; None means no limit
        """
        self.model_config = model_config
        self.endpoints = get_endpoints(model_config)
        
        if len(self.endpoints) <= 1:
            self.model_caller = ModelCaller(model_config=model_config)
        else:
            self.model_caller = None  # created per-endpoint in evaluate_all
        
        self.results = []
        self.experiment_name = experiment_name or "default"
        self.model_name = model_config.get("name", "unknown")
        self.max_history_tokens = max_history_tokens
        self.max_history_days = max_history_days
        self.skip_model_eval = skip_model_eval
        self.verbose = verbose
        self.run_summary = {}
        self.run_summary_path = None
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        now_shanghai = datetime.now(shanghai_tz)
        
        self.run_date = run_date or now_shanghai.strftime("%Y-%m-%d")
        self.run_timestamp = run_timestamp or now_shanghai.strftime("%Y-%m-%d_%H-%M-%S")
        
        if len(self.endpoints) > 1:
            total_workers = sum(ep["max_workers"] for ep in self.endpoints)
            print(f"Multi-endpoint mode: {len(self.endpoints)} endpoints, total concurrency {total_workers}:")
            for i, ep in enumerate(self.endpoints):
                print(f"   [{i+1}] {ep['url']} (workers: {ep['max_workers']})")

    def _save_run_summary(self):
        """Persist a compact run summary after each pipeline phase."""
        if not self.run_summary_path:
            return
        os.makedirs(os.path.dirname(self.run_summary_path), exist_ok=True)
        with open(self.run_summary_path, 'w', encoding='utf-8') as f:
            json.dump(round_floats(self.run_summary, decimals=4), f, ensure_ascii=False, indent=2)

    def _summarize_cost_stats(self, stats: Dict) -> Dict:
        summary = {
            "token_source": stats.get("token_source"),
            "total_chars": stats.get("total_chars", 0),
            "prompt_chars": stats.get("total_prompt_chars", 0),
            "response_chars": stats.get("total_response_chars", 0),
            "successful_calls": stats.get("successful_calls", 0),
            "total_questions": stats.get("total_questions", 0),
            "failed_questions": stats.get("failed_questions", 0),
        }
        if stats.get("token_source") == "api_actual":
            summary.update({
                "prompt_tokens": stats.get("total_prompt_tokens", 0),
                "completion_tokens": stats.get("total_completion_tokens", 0),
                "total_tokens": stats.get("total_tokens", 0),
                "cached_tokens": stats.get("total_cached_tokens", 0),
                "cache_hit_rate": stats.get("cache_hit_rate", 0),
                "avg_prompt_tokens_per_question": stats.get("avg_prompt_tokens_per_question", 0),
                "avg_prompt_tokens_per_action": stats.get("avg_prompt_tokens_per_action", 0),
                "prompt_tokens_stats": stats.get("prompt_tokens_stats", {}),
            })
        else:
            summary.update({
                "estimated_tokens": stats.get("estimated_tokens", 0),
                "cached_tokens": stats.get("total_cached_tokens", 0),
            })
        return summary

    def record_llm_judge_summary(self, results: List[Dict]):
        status = {
            "evaluated": sum(1 for r in results if "llm_judge_scores" in r),
            "skipped": sum(1 for r in results if r.get("llm_judge_skipped")),
            "errors": sum(1 for r in results if r.get("llm_judge_error")),
            "target_records": sum(1 for r in results if get_record_action_type(r) == "电商客服对话"),
        }
        averages = {}
        score_keys = ["intent_fidelity", "persona_mimicry", "knowledge_boundary", "semantic_alignment"]
        for key in score_keys:
            values = [r["llm_judge_scores"][key] for r in results if key in r.get("llm_judge_scores", {})]
            averages[key] = sum(values) / len(values) if values else None
        avg_scores = [r.get("llm_judge_avg_score") for r in results if r.get("llm_judge_avg_score") is not None]
        averages["average_score"] = sum(avg_scores) / len(avg_scores) if avg_scores else None
        self.run_summary["llm_judge"] = {
            "status": status,
            "average_scores": averages,
        }
        self._save_run_summary()
        
    def evaluate_single_action(
        self,
        user_id: str,
        user_profile: str,
        action_history: List[Dict],
        test_action: Dict,
        model_caller: ModelCaller = None
    ) -> Dict:
        """
        Evaluate a single action prediction (binary mode: one call per question, Yes/No output).

        Args:
            model_caller: optional ModelCaller to use (multi-endpoint mode)

        Returns:
            {
                "user_id": str,
                "action_type": str,
                "timestamp": str,
                "questions": List[Dict],
                "prompts": List[str],
                "total_prompt_length": int,
                "model_responses": List[str],
                "total_response_length": int,
                "success": bool,
                "filtered": bool,
                "binary_mode": bool,
                "failed_questions": List[Dict],
            }
        """
        caller = model_caller or self.model_caller
        all_questions = get_all_questions_for_action(test_action)

        # check if the action should be filtered before trying to get questions
        from models.prompt_builder import should_filter_action
        if should_filter_action(test_action):
            return {
                "user_id": user_id,
                "action_type": get_action_type(test_action, "unknown"),
                "timestamp": test_action.get("timestamp", "unknown"),
                "scene_info": None,
                "prompts": [],
                "total_prompt_length": 0,
                "model_responses": [],
                "total_response_length": 0,
                "success": False,
                "filtered": True,
                "questions": [],
                "binary_mode": True,
                "failed_questions": [],
            }
        
        if not all_questions:
            return {
                "user_id": user_id,
                "action_type": get_action_type(test_action, "unknown"),
                "timestamp": test_action.get("timestamp", "unknown"),
                "scene_info": None,
                "prompts": [],
                "total_prompt_length": 0,
                "model_responses": [],
                "total_response_length": 0,
                "success": False,
                "filtered": False,
                "questions": [],
                "binary_mode": True,
                "failed_questions": [
                    {
                        "error": "no_prediction_questions",
                        "action_type": get_action_type(test_action, "unknown"),
                    }
                ],
            }
        
        scene_info = {
            "type": get_action_type(test_action, "unknown"),
            "timestamp": test_action.get("timestamp", "unknown"),
            "context": test_action.get("context", {}),
            "action": test_action.get("action", []),
        }
        
        history_info = get_actual_used_history(
            action_history, 
            max_history_tokens=self.max_history_tokens,
            max_history_days=self.max_history_days,
            reference_timestamp=test_action.get("timestamp")
        )
        actual_used_actions = history_info["actual_used_actions"]
        
        history_scene_counts = {}
        history_action_counts = {}
        history_timestamps = []
        
        for action in actual_used_actions:
            scene_type = get_action_type(action, "Unknown")
            history_scene_counts[scene_type] = history_scene_counts.get(scene_type, 0) + 1
            
            timestamp = action.get("timestamp", "")
            if timestamp:
                history_timestamps.append(timestamp)

            actions_list = action.get("action", [])
            if isinstance(actions_list, list):
                for act in actions_list:
                    act_type = act.get("type", "unknown")
                    key = f"{scene_type}_{act_type}"
                    history_action_counts[key] = history_action_counts.get(key, 0) + 1
        
        history_time_stats = self._calculate_history_time_stats(history_timestamps)
        
        result = {
            "user_id": user_id,
            "action_type": get_action_type(test_action, "unknown"),
            "timestamp": test_action.get("timestamp", "unknown"),
            "scene_info": scene_info,
            "prompts": [],
            "total_prompt_length": 0,
            "model_responses": [],
            "total_response_length": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "success": False,
            "filtered": False,
            "questions": [],
            "binary_mode": True,
            "failed_questions": [],
            "history_stats": {
                "original_count": history_info["original_count"],
                "filtered_count": history_info["filtered_count"],
                "actual_used_count": history_info["actual_used_count"],
                "actual_used_tokens": history_info["actual_used_tokens"],
                "scene_distribution": history_scene_counts,
                "action_distribution": history_action_counts,
                "earliest_timestamp": history_time_stats.get("earliest_timestamp"),
                "latest_timestamp": history_time_stats.get("latest_timestamp"),
                "time_span_days": history_time_stats.get("time_span_days"),
                "avg_actions_per_day": history_time_stats.get("avg_actions_per_day"),
            },
        }
        
        successful_predictions = 0

        for question_info in all_questions:
            question_type = question_info.get("type", "binary")

            if question_type == "binary":
                prompt_data = build_single_binary_prompt(
                    user_profile,
                    action_history,
                    test_action,
                    question_info,
                    max_history_tokens=self.max_history_tokens,
                    max_history_days=self.max_history_days
                )
            elif question_type == "continuous":
                prompt_data = build_single_continuous_prompt(
                    user_profile,
                    action_history,
                    test_action,
                    question_info,
                    max_history_tokens=self.max_history_tokens,
                    max_history_days=self.max_history_days
                )
            elif question_type == "text":
                prompt_data = build_single_text_prompt(
                    user_profile,
                    action_history,
                    test_action,
                    question_info,
                    max_history_tokens=self.max_history_tokens,
                    max_history_days=self.max_history_days
                )
            else:
                continue
            if prompt_data is None:
                continue
            
            prompt = prompt_data["prompt"]
            result["prompts"].append(prompt)
            result["total_prompt_length"] += len(prompt)
            
            if self.skip_model_eval:
                prediction_result = {
                    "success": True,
                    "prediction": question_info.get("true_value"),
                    "predicted_label": ("YES" if question_info.get("true_value") == 1 else "NO") if question_type == "binary" else str(question_info.get("true_value")),
                    "method": "mock_for_sft",
                    "raw_output": "SFT Export Mode Skipped Eval",
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": 0,
                    "cached_tokens": 0
                }
            else:
                if question_type == "binary":
                    prediction_result = caller.call_binary_classification(prompt)
                elif question_type == "continuous":
                    prediction_result = caller.call_continuous_prediction(prompt)
                elif question_type == "text":
                    prediction_result = caller.call_text_prediction(prompt)

            raw_output = prediction_result.get("raw_output", "")
            cleaned_output = remove_think_tags(raw_output)
            result["model_responses"].append(cleaned_output)
            result["total_response_length"] += len(cleaned_output)

            result["total_prompt_tokens"] += prediction_result.get("prompt_tokens", 0)
            result["total_completion_tokens"] += prediction_result.get("completion_tokens", 0)
            result["total_cached_tokens"] += prediction_result.get("cached_tokens", 0)

            question_result = {
                "type": question_info["type"],
                "field": question_info["field"],
                "true_value": question_info["true_value"],
                "predicted_value": prediction_result.get("prediction"),
                "predicted_label": prediction_result.get("predicted_label"),
                "prediction_method": prediction_result.get("method"),
                "raw_output": raw_output,
                "retry_count": prediction_result.get("retry_count", 0),
                "logprob_yes": prediction_result.get("logprob_yes"),
                "logprob_no": prediction_result.get("logprob_no"),
                "video_duration": question_info.get("video_duration"),
                "prompt_tokens": prediction_result.get("prompt_tokens", 0),
                "completion_tokens": prediction_result.get("completion_tokens", 0),
                "cached_tokens": prediction_result.get("cached_tokens", 0),
                "history_action_count": history_info["actual_used_count"],
                "history_token_count": history_info["actual_used_tokens"],
                "history_earliest_time": history_time_stats.get("earliest_timestamp"),
                "history_latest_time": history_time_stats.get("latest_timestamp"),
                "history_time_span_days": history_time_stats.get("time_span_days"),
                "history_avg_actions_per_day": history_time_stats.get("avg_actions_per_day"),
                "has_prior_user_speech": question_info.get("has_prior_user_speech"),
            }
            
            if prediction_result.get("success"):
                successful_predictions += 1
                result["questions"].append(question_result)
            else:
                question_result["error"] = prediction_result.get("error")
                result["failed_questions"].append(question_result)

        result["success"] = successful_predictions > 0
        
        return result
    
    def _calculate_history_time_stats(self, timestamps: List[str]) -> Dict:
        """Calculate time-distribution statistics over a list of history timestamps."""
        if not timestamps:
            return {
                "earliest_timestamp": None,
                "latest_timestamp": None,
                "time_span_days": None,
                "avg_actions_per_day": None,
            }
        
        from datetime import datetime
        
        parsed_times = []
        for ts in timestamps:
            if not ts:
                continue
            try:
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"]:
                    try:
                        parsed_times.append(datetime.strptime(ts, fmt))
                        break
                    except ValueError:
                        continue
            except Exception:
                continue
        
        if not parsed_times:
            return {
                "earliest_timestamp": timestamps[0] if timestamps else None,
                "latest_timestamp": timestamps[-1] if timestamps else None,
                "time_span_days": None,
                "avg_actions_per_day": None,
            }
        
        parsed_times.sort()
        earliest = parsed_times[0]
        latest = parsed_times[-1]

        time_span = (latest - earliest).total_seconds() / (24 * 3600)
        time_span_days = round(time_span, 2)

        if time_span_days > 0:
            avg_actions_per_day = round(len(parsed_times) / time_span_days, 2)
        else:
            avg_actions_per_day = float(len(parsed_times))
        
        return {
            "earliest_timestamp": earliest.strftime("%Y-%m-%d %H:%M:%S"),
            "latest_timestamp": latest.strftime("%Y-%m-%d %H:%M:%S"),
            "time_span_days": time_span_days,
            "avg_actions_per_day": avg_actions_per_day,
        }
    
    def _run_multi_endpoint_evaluation(self, tasks: List[Dict], all_results: List[Dict],
                                         total_actions: int, update_stats_fn, report_progress_fn,
                                         save_fn=None, error_log_dir=None):
        """
        Multi-endpoint parallel evaluation with dynamic load balancing (work stealing).
        Tasks are initially assigned by user hash (for prefix-cache affinity), then
        idle endpoints steal from others to maximise utilisation.

        Args:
            save_fn: optional streaming save callback for periodic checkpoints
            error_log_dir: directory for error logs
        """
        from collections import defaultdict
        
        user_tasks = defaultdict(list)
        for task in tasks:
            user_tasks[task["user_id"]].append(task)

        task_queue = DynamicTaskQueue(self.endpoints, tasks, dict(user_tasks))

        lock = threading.Lock()
        results_lock = threading.Lock()
        completed = [0]
        endpoint_first_success = {ep["url"]: False for ep in self.endpoints}
        last_progress_milestone = [0]
        endpoint_active_workers = {ep["url"]: 0 for ep in self.endpoints}
        
        def _print_model_output_sample(result: Dict, prefix: str):
            """Print a brief model output sample for progress logging."""
            try:
                from tqdm import tqdm
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"{prefix}")
                tqdm.write(f"{'='*60}")

                if result.get("model_responses"):
                    first_response = result["model_responses"][0] if result["model_responses"] else "N/A"
                    tqdm.write(f"   user_id: {result.get('user_id', 'N/A')}")
                    tqdm.write(f"   action_type: {result.get('action_type', 'N/A')}")
                    tqdm.write(f"   model output: {first_response[:100]}{'...' if len(str(first_response)) > 100 else ''}")
                    if result.get("questions"):
                        q = result["questions"][0]
                        pred_val = q.get('predicted_value', 'N/A')
                        true_val = q.get('true_value', 'N/A')
                        q_type = q.get('type', 'unknown')
                        tqdm.write(f"   question type: {q_type}, predicted: {pred_val}, true: {true_val}")
                else:
                    tqdm.write(f"   result: {str(result)[:200]}...")
                tqdm.write(f"{'='*60}\n")
            except ImportError:
                print(f"\n[{prefix}] model output: {str(result)[:200]}...\n", flush=True)
        
        def worker_loop(endpoint_url: str, caller: ModelCaller, worker_id: int):
            """Worker loop: pull tasks until the global queue is empty."""
            results = []

            while True:
                task = task_queue.get_task(endpoint_url)

                if task is None:
                    break

                try:
                    with lock:
                        endpoint_active_workers[endpoint_url] += 1

                    result = self.evaluate_single_action(
                        task["user_id"],
                        task["user_profile"],
                        task["action_history"],
                        task["test_action"],
                        model_caller=caller
                    )
                    results.append(result)

                    with lock:
                        completed[0] += 1
                        endpoint_active_workers[endpoint_url] -= 1
                        update_stats_fn(result)
                        task_queue.mark_completed()

                        if self.verbose and not endpoint_first_success[endpoint_url] and result.get("success"):
                            endpoint_first_success[endpoint_url] = True
                            _print_model_output_sample(result, f"✅ endpoint [{endpoint_url}] first call succeeded")

                        current_progress = int((completed[0] / total_actions) * 100)
                        current_milestone = (current_progress // 10) * 10
                        if self.verbose and current_milestone > last_progress_milestone[0]:
                            last_progress_milestone[0] = current_milestone
                            _print_model_output_sample(result, f"🚀 progress milestone: {current_milestone}%")
                        
                        # periodic save
                        if save_fn:
                            save_fn(results)
                
                except Exception as e:
                    print(f"\nTask error: {e}")
                    import traceback
                    traceback.print_exc()
                    with lock:
                        completed[0] += 1
                        endpoint_active_workers[endpoint_url] -= 1
                
                if len(results) >= 10:
                    with results_lock:
                        all_results.extend(results)
                        if save_fn:
                            save_fn(all_results)
                    results = []

            if results:
                with results_lock:
                    all_results.extend(results)
                    if save_fn:
                        save_fn(all_results)
            
            return results
        
        def run_endpoint(endpoint_url: str, endpoint_max_workers: int):
            """Run the worker thread pool for a single endpoint."""
            caller = ModelCaller(self.model_config, base_url_override=endpoint_url, error_log_dir=error_log_dir)

            all_endpoint_results = []

            with ThreadPoolExecutor(max_workers=endpoint_max_workers) as executor:
                futures = []
                for worker_id in range(endpoint_max_workers):
                    future = executor.submit(worker_loop, endpoint_url, caller, worker_id)
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        results = future.result()
                        all_endpoint_results.extend(results)
                    except Exception as e:
                        print(f"\n[{endpoint_url}] worker error: {e}")
            
            return all_endpoint_results
        
        print(f"\nStarting {len(self.endpoints)} endpoints in parallel")

        with tqdm(total=total_actions, desc="Total progress") as pbar:
            stop_progress = [False]
            
            def update_progress():
                last = 0
                last_stats_print = 0
                while not stop_progress[0] and last < total_actions:
                    with lock:
                        current = completed[0]
                    if current > last:
                        pbar.update(current - last)
                        report_progress_fn(current, total_actions)
                        last = current
                    
                    if self.verbose and current > 0 and current - last_stats_print >= 500:
                        last_stats_print = current
                        stats = task_queue.get_stats()
                        remaining = stats["remaining_per_endpoint"]
                        stolen = stats["stolen_count"]
                        stolen_rate = stats["stolen_rate"]

                        status_parts = []
                        for ep in self.endpoints:
                            url = ep["url"]
                            short_url = url.split("//")[-1].split("/")[0]
                            r = remaining.get(url, 0)
                            status_parts.append(f"{short_url}:{r}")
                        
                        try:
                            from tqdm import tqdm
                            tqdm.write(f"\n⚖️  Load balance [{current}/{total_actions}]: stolen={stolen} ({stolen_rate:.1f}%)")
                            tqdm.write(f"   Remaining: {', '.join(status_parts)}")
                        except ImportError:
                            pass
                    
                    time.sleep(0.1)
            
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()
            
            with ThreadPoolExecutor(max_workers=len(self.endpoints)) as endpoint_executor:
                endpoint_futures = []
                for ep in self.endpoints:
                    future = endpoint_executor.submit(run_endpoint, ep["url"], ep["max_workers"])
                    endpoint_futures.append(future)

                for future in as_completed(endpoint_futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"\nEndpoint error: {e}")

            stop_progress[0] = True
            progress_thread.join(timeout=1)

            if save_fn:
                save_fn(all_results)

        stats = task_queue.get_stats()
        print(f"\nLoad balancing stats:")
        print(f"   Total tasks: {stats['total_tasks']}")
        print(f"   Completed: {stats['completed']}")
        print(f"   Stolen tasks: {stats['stolen_count']} ({stats['stolen_rate']:.1f}%)")
        print(f"\n   Per-endpoint stats:")
        for ep in self.endpoints:
            url = ep["url"]
            ep_stats = stats["endpoint_stats"].get(url, {})
            local = ep_stats.get("local", 0)
            stolen = ep_stats.get("stolen", 0)
            total_processed = local + stolen
            print(f"   [{url}]")
            print(f"      local: {local}, stolen: {stolen}, total: {total_processed}")

            progress_thread.join(timeout=1)
    
    def evaluate_all(self, eval_data: List[Dict], save_intermediate: bool = True, max_workers: int = None) -> List[Dict]:
        """
        Evaluate all test actions for all users (rolling-prediction mode).

        History construction:
        - base history: actions from the historical time window
        - incremental history: actions in the test window before the current test action
        - full history = base + incremental; oldest entries dropped first if token limit exceeded

        Args:
            eval_data: output of load_fixed_experiment_data()
            save_intermediate: whether to save intermediate results
            max_workers: max concurrency; None uses config value, 1 runs sequentially

        Returns:
            list of all evaluation results
        """
        print(f"\nStarting evaluation for {len(eval_data)} users...")
        if len(self.endpoints) > 1:
            total_workers = sum(ep["max_workers"] for ep in self.endpoints)
            print(f"Concurrency: multi-endpoint mode, total {total_workers}")
        else:
            print(f"Concurrency: {max_workers}")
        print(f"Mode: rolling prediction (base history + incremental history before each test action)")
        if self.max_history_days and self.max_history_days > 0:
            print(f"History day filter: only history within {self.max_history_days} days before each test action")
        
        all_results = []
        total_actions = sum(len(d["test_actions"]) for d in eval_data)
        
        # prepare all tasks (build history context for each task upfront)
        tasks = []
        for user_data in eval_data:
            user_id = user_data["user_id"]
            user_profile = user_data["user_profile"]
            base_history = user_data["base_history"]
            test_time_all_actions = user_data["test_time_all_actions"]
            test_actions = user_data["test_actions"]
            
            for i, test_item in enumerate(test_actions):
                test_action = dict(test_item["action"])
                test_action["user_id"] = user_id
                test_time_index = test_item["test_time_index"]

                incremental_history = test_time_all_actions[:test_time_index]
                
                task_history = base_history + incremental_history
                
                if self.max_history_days is not None and self.max_history_days > 0:
                    from datetime import datetime, timedelta

                    reference_timestamp = test_action.get("timestamp")
                    if reference_timestamp:
                        try:
                            if len(reference_timestamp) == 10:  # YYYY-MM-DD
                                ref_datetime = datetime.strptime(reference_timestamp, "%Y-%m-%d")
                            else:  # YYYY-MM-DD HH:MM:SS
                                ref_datetime = datetime.strptime(reference_timestamp[:19], "%Y-%m-%d %H:%M:%S")

                            cutoff_datetime = ref_datetime - timedelta(days=self.max_history_days)
                            cutoff_str = cutoff_datetime.strftime("%Y-%m-%d %H:%M:%S")

                            days_filtered_history = []
                            for action in task_history:
                                action_timestamp = action.get("timestamp", "")
                                if action_timestamp and action_timestamp >= cutoff_str:
                                    days_filtered_history.append(action)

                            task_history = days_filtered_history
                        except ValueError:
                            pass
                
                tasks.append({
                    "user_id": user_id,
                    "user_profile": user_profile,
                    "action_history": task_history,
                    "test_action": test_action,
                    "task_index": i
                })

        question_count_by_type = {}
        no_question_count_by_type = {}
        total_questions_to_predict = 0

        for task in tasks:
            action_type = get_action_type(task["test_action"], "unknown")
            question_count = len(get_all_questions_for_action(task["test_action"]))
            if question_count:
                question_count_by_type[action_type] = question_count_by_type.get(action_type, 0) + question_count
                total_questions_to_predict += question_count
            else:
                no_question_count_by_type[action_type] = no_question_count_by_type.get(action_type, 0) + 1

        print(f"Total questions to predict: {total_questions_to_predict}")
        if no_question_count_by_type:
            print("⚠️  The following scene types generated no prediction questions (model will not be called):")
            for action_type, count in sorted(no_question_count_by_type.items(), key=lambda x: -x[1]):
                print(f"   - {action_type}: {count} actions")
        if self.verbose and question_count_by_type:
            print("Question distribution by scene:")
            for action_type, count in sorted(question_count_by_type.items(), key=lambda x: -x[1]):
                print(f"   - {action_type}: {count} questions")
        if tasks and total_questions_to_predict == 0:
            supported_types = "、".join(["视频浏览", "商城购物", "广告推荐", "直播间", "电商客服对话"])
            observed_types = "、".join(sorted(no_question_count_by_type.keys())) or "无"
            raise ValueError(
                "所有待评估行为都没有生成预测问题，因此不会发起任何模型调用。"
                f"当前数据中的场景类型: {observed_types}。"
                f"问题生成器支持的场景类型: {supported_types}。"
                "请检查数据集 action.type 是否与支持的中文场景名一致。"
            )
        
        cumulative_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        last_reported_percentage = 0
        last_save_count = [0]

        SAVE_INTERVAL = 500

        timestamp_output_dir = None
        intermediate_output_path = None
        if save_intermediate:
            date_dir = self.run_date
            timestamp = self.run_timestamp
            timestamp_output_dir = os.path.join(OUTPUT_DIR, date_dir, timestamp)
            # Handle model names with slashes (e.g. "pa/Claude...") by replacing with underscore
            safe_experiment_name = self.experiment_name.replace('/', '_')
            intermediate_output_path = os.path.join(timestamp_output_dir, f"{safe_experiment_name}_prediction_results.json")

            error_log_dir = os.path.join(timestamp_output_dir, "error_logs")
            if not os.path.exists(error_log_dir):
                os.makedirs(error_log_dir, exist_ok=True)

            if self.model_caller:
                self.model_caller.set_error_log_dir(error_log_dir)

            print(f"Intermediate results will be saved to: {intermediate_output_path}")
            print(f"Error logs will be saved to: {error_log_dir}")
            print(f"Auto-saving every {SAVE_INTERVAL} completed tasks")
        else:
            error_log_dir = None

        def update_cumulative_stats(result):
            """Accumulate token usage statistics."""
            if result.get("binary_mode"):
                cumulative_stats["prompt_tokens"] += result.get("total_prompt_tokens", 0)
                cumulative_stats["completion_tokens"] += result.get("total_completion_tokens", 0)
                cumulative_stats["cached_tokens"] += result.get("total_cached_tokens", 0)

        def save_intermediate_results(results_list, force=False):
            """Stream-save intermediate results (no-op when save_intermediate=False)."""
            if not save_intermediate or intermediate_output_path is None:
                return

            current_count = len(results_list)
            if force or (current_count - last_save_count[0] >= SAVE_INTERVAL):
                if not os.path.exists(timestamp_output_dir):
                    os.makedirs(timestamp_output_dir, exist_ok=True)
                    if self.verbose:
                        print(f"✅ Created output directory: {timestamp_output_dir}")
                try:
                    with open(intermediate_output_path, 'w', encoding='utf-8') as f:
                        json.dump(results_list, f, ensure_ascii=False, indent=2)
                    last_save_count[0] = current_count
                    if self.verbose:
                        print(f"\n💾 Saved intermediate results: {current_count}/{total_actions} tasks done")
                except Exception as e:
                    print(f"\n⚠️ Failed to save intermediate results: {e}")
        
        def report_progress(completed_count, total_count):
            """Report progress at every 10% milestone."""
            if not self.verbose:
                return
            nonlocal last_reported_percentage
            current_percentage = int((completed_count / total_count) * 10) * 10

            if current_percentage > last_reported_percentage and current_percentage > 0:
                last_reported_percentage = current_percentage
                total_tokens = cumulative_stats["prompt_tokens"] + cumulative_stats["completion_tokens"]
                print(f"\nProgress {current_percentage}% ({completed_count}/{total_count}) Token stats:")
                print(f"   Prompt Tokens: {cumulative_stats['prompt_tokens']:,}")
                print(f"   Completion Tokens: {cumulative_stats['completion_tokens']:,}")
                print(f"   Total Tokens: {total_tokens:,}")
                if cumulative_stats["cached_tokens"] > 0:
                    print(f"   Cached Tokens: {cumulative_stats['cached_tokens']:,}")
                    if cumulative_stats["prompt_tokens"] > 0:
                        cache_rate = cumulative_stats["cached_tokens"] / cumulative_stats["prompt_tokens"] * 100
                        print(f"   Cache hit rate: {cache_rate:.2f}%")
                else:
                    print(f"   Cached Tokens: no data (API did not return cache info)")
        
        if len(self.endpoints) > 1:
            self._run_multi_endpoint_evaluation(tasks, all_results, total_actions, 
                                                 update_cumulative_stats, report_progress,
                                                 save_fn=save_intermediate_results,
                                                 error_log_dir=error_log_dir)
        elif max_workers == 1:
            with tqdm(total=total_actions, desc="Evaluation progress") as pbar:
                for task in tasks:
                    result = self.evaluate_single_action(
                        task["user_id"],
                        task["user_profile"],
                        task["action_history"],
                        task["test_action"]
                    )
                    all_results.append(result)
                    update_cumulative_stats(result)
                    pbar.update(1)
                    report_progress(len(all_results), total_actions)
                    save_intermediate_results(all_results)
        else:
            lock = threading.Lock()

            def evaluate_task(task):
                return self.evaluate_single_action(
                    task["user_id"],
                    task["user_profile"],
                    task["action_history"],
                    task["test_action"]
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                with tqdm(total=total_actions, desc="Evaluation progress") as pbar:
                    future_to_task = {executor.submit(evaluate_task, task): task for task in tasks}

                    for future in as_completed(future_to_task):
                        try:
                            result = future.result()
                            with lock:
                                all_results.append(result)
                                update_cumulative_stats(result)
                                pbar.update(1)
                                report_progress(len(all_results), total_actions)
                                save_intermediate_results(all_results)
                        except Exception as e:
                            print(f"\nEvaluation task error: {e}")
                            with lock:
                                pbar.update(1)

        filtered_count = sum(1 for r in all_results if r.get("filtered", False))
        valid_count = len(all_results) - filtered_count

        if filtered_count > 0:
            print(f"\n⚠️  Note: {filtered_count} samples filtered (play_duration is 0)")
            print(f"   Valid samples: {valid_count} / {len(all_results)}")

        if save_intermediate:
            save_intermediate_results(all_results, force=True)
            output_path = intermediate_output_path
            
            stats = self._calculate_cost_stats(all_results)
            safe_experiment_name = self.experiment_name.replace('/', '_')
            stats_path = os.path.join(timestamp_output_dir, f"{safe_experiment_name}_cost_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            history_distribution = self._aggregate_history_stats(all_results)
            history_stats_path = os.path.join(timestamp_output_dir, f"{safe_experiment_name}_actual_history_distribution.json")
            with open(history_stats_path, 'w', encoding='utf-8') as f:
                json.dump(history_distribution, f, ensure_ascii=False, indent=2)

            self.run_summary_path = os.path.join(timestamp_output_dir, f"{safe_experiment_name}_run_summary.json")
            self.run_summary = {
                "model": self.model_name,
                "experiment_name": self.experiment_name,
                "run_date": self.run_date,
                "run_timestamp": self.run_timestamp,
                "max_history_tokens": self.max_history_tokens,
                "max_history_days": self.max_history_days,
                "files": {
                    "prediction_results": output_path,
                    "cost_stats": stats_path,
                    "actual_history_distribution": history_stats_path,
                    "run_summary": self.run_summary_path,
                },
                "prediction_results": {
                    "total_actions": len(all_results),
                    "valid_actions": valid_count,
                    "filtered_actions": filtered_count,
                    "successful_actions": sum(1 for r in all_results if r.get("success")),
                },
                "token_usage": self._summarize_cost_stats(stats),
                "history_distribution": history_distribution.get("metadata", {}),
            }
            self._save_run_summary()

            token_summary = self.run_summary["token_usage"]
            if stats.get("token_source") == "api_actual":
                print(
                    f"Saved predictions: {output_path}\n"
                    f"Saved run summary: {self.run_summary_path}\n"
                    f"Token usage: {token_summary.get('total_tokens', 0):,} total "
                    f"({token_summary.get('prompt_tokens', 0):,} prompt, "
                    f"{token_summary.get('completion_tokens', 0):,} completion), "
                    f"cache hit {token_summary.get('cache_hit_rate', 0):.2%}"
                )
            else:
                print(
                    f"Saved predictions: {output_path}\n"
                    f"Saved run summary: {self.run_summary_path}\n"
                    f"Estimated token usage: {token_summary.get('estimated_tokens', 0):,}"
                )
        
        return all_results
    
    def _calculate_cost_stats(self, results: List[Dict]) -> Dict:
        """Calculate cost/token statistics, supporting both old and new result formats."""
        total_prompt_chars = 0
        total_response_chars = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        successful_calls = 0
        total_questions = 0
        failed_questions = 0
        total_retries = 0
        
        all_prompt_tokens = []
        
        for result in results:
            if result.get("success"):
                successful_calls += 1
                
                if result.get("binary_mode"):
                    total_prompt_chars += result.get("total_prompt_length", 0)
                    total_response_chars += result.get("total_response_length", 0)
                    total_prompt_tokens += result.get("total_prompt_tokens", 0)
                    total_completion_tokens += result.get("total_completion_tokens", 0)
                    total_cached_tokens += result.get("total_cached_tokens", 0)
                    total_questions += len(result.get("questions", []))
                    failed_questions += len(result.get("failed_questions", []))
                    
                    for q in result.get("questions", []):
                        total_retries += q.get("retry_count", 0)
                        prompt_tokens = q.get("prompt_tokens", 0)
                        if prompt_tokens > 0:
                            all_prompt_tokens.append(prompt_tokens)
                    for q in result.get("failed_questions", []):
                        total_retries += q.get("retry_count", 0)
                        prompt_tokens = q.get("prompt_tokens", 0)
                        if prompt_tokens > 0:
                            all_prompt_tokens.append(prompt_tokens)
                else:
                    total_prompt_chars += result.get("prompt_length", 0)
                    total_response_chars += result.get("response_length", 0)
                    total_questions += len(result.get("questions", []))
        
        total_chars = total_prompt_chars + total_response_chars
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        # rough estimate: ~2.5 chars/token when no API token data available
        estimated_tokens = int(total_chars / 2.5) if total_tokens == 0 else None
        
        stats = {
            "model_name": self.model_name,
            "successful_calls": successful_calls,
            "total_prompt_chars": total_prompt_chars,
            "total_response_chars": total_response_chars,
            "total_chars": total_chars,
            "avg_prompt_chars": int(total_prompt_chars / successful_calls) if successful_calls > 0 else 0,
            "avg_response_chars": int(total_response_chars / successful_calls) if successful_calls > 0 else 0,
            "total_questions_predicted": total_questions,
            "failed_questions": failed_questions,
            "total_retries": total_retries,
        }
        
        if total_tokens > 0:
            stats["total_prompt_tokens"] = total_prompt_tokens
            stats["total_completion_tokens"] = total_completion_tokens
            stats["total_tokens"] = total_tokens
            stats["total_cached_tokens"] = total_cached_tokens
            stats["avg_prompt_tokens_per_question"] = int(total_prompt_tokens / total_questions) if total_questions > 0 else 0
            stats["avg_completion_tokens_per_question"] = int(total_completion_tokens / total_questions) if total_questions > 0 else 0
            stats["avg_prompt_tokens_per_action"] = int(total_prompt_tokens / successful_calls) if successful_calls > 0 else 0
            stats["cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0
            stats["token_source"] = "api_actual"
        else:
            stats["estimated_tokens"] = estimated_tokens
            stats["total_cached_tokens"] = total_cached_tokens
            stats["token_source"] = "estimated"

        if all_prompt_tokens:
            stats["prompt_tokens_stats"] = {
                "count": len(all_prompt_tokens),
                "avg": round(sum(all_prompt_tokens) / len(all_prompt_tokens), 2),
                "min": min(all_prompt_tokens),
                "max": max(all_prompt_tokens),
                "median": sorted(all_prompt_tokens)[len(all_prompt_tokens) // 2],
            }
        else:
            stats["prompt_tokens_stats"] = {
                "count": 0,
                "avg": 0,
                "min": 0,
                "max": 0,
                "median": 0,
            }
        
        return stats
    
    def _aggregate_history_stats(self, results: List[Dict]) -> Dict:
        """Aggregate actual history-usage statistics across all prediction results."""
        total_scene_counts = {}
        total_action_counts = {}
        total_samples = 0
        total_original_count = 0
        total_filtered_count = 0
        total_actual_used_count = 0
        total_actual_used_tokens = 0
        
        total_prompt_tokens = 0
        total_prompt_chars = 0
        user_stats = {}
        
        for result in results:
            if result.get("filtered", False):
                continue
            
            history_stats = result.get("history_stats", {})
            if not history_stats:
                continue
            
            total_samples += 1
            total_original_count += history_stats.get("original_count", 0)
            total_filtered_count += history_stats.get("filtered_count", 0)
            total_actual_used_count += history_stats.get("actual_used_count", 0)
            total_actual_used_tokens += history_stats.get("actual_used_tokens", 0)
            
            prompt_tokens = result.get("total_prompt_tokens", 0)
            prompt_chars = result.get("total_prompt_length", 0)
            total_prompt_tokens += prompt_tokens
            total_prompt_chars += prompt_chars

            user_id = result.get("user_id", "unknown")
            if user_id not in user_stats:
                user_stats[user_id] = {
                    "history_count": [],
                    "history_tokens": [],
                    "prompt_tokens": [],
                    "prompt_chars": []
                }
            user_stats[user_id]["history_count"].append(history_stats.get("actual_used_count", 0))
            user_stats[user_id]["history_tokens"].append(history_stats.get("actual_used_tokens", 0))
            user_stats[user_id]["prompt_tokens"].append(prompt_tokens)
            user_stats[user_id]["prompt_chars"].append(prompt_chars)
            
            for scene, count in history_stats.get("scene_distribution", {}).items():
                total_scene_counts[scene] = total_scene_counts.get(scene, 0) + count

            for action, count in history_stats.get("action_distribution", {}).items():
                total_action_counts[action] = total_action_counts.get(action, 0) + count

        total_scene_actions = sum(total_scene_counts.values())
        scene_distribution = {}
        for scene, count in sorted(total_scene_counts.items(), key=lambda x: -x[1]):
            scene_distribution[scene] = {
                "count": count,
                "percentage": round(count / total_scene_actions * 100, 2) if total_scene_actions > 0 else 0
            }

        total_actions = sum(total_action_counts.values())
        action_distribution = {}
        for action, count in sorted(total_action_counts.items(), key=lambda x: -x[1]):
            action_distribution[action] = {
                "count": count,
                "percentage": round(count / total_actions * 100, 2) if total_actions > 0 else 0
            }

        per_scene_actions = {}
        for action_key, count in total_action_counts.items():
            if "_" in action_key:
                scene = action_key.rsplit("_", 1)[0]
                for known_scene in total_scene_counts.keys():
                    if action_key.startswith(known_scene + "_"):
                        scene = known_scene
                        break
                if scene not in per_scene_actions:
                    per_scene_actions[scene] = {}
                per_scene_actions[scene][action_key] = count
        
        num_users = len(user_stats)
        user_avg_history_count = []
        user_avg_history_tokens = []
        user_avg_prompt_tokens = []
        user_avg_prompt_chars = []
        
        for user_id, stats in user_stats.items():
            if stats["history_count"]:
                user_avg_history_count.append(sum(stats["history_count"]) / len(stats["history_count"]))
                user_avg_history_tokens.append(sum(stats["history_tokens"]) / len(stats["history_tokens"]))
                user_avg_prompt_tokens.append(sum(stats["prompt_tokens"]) / len(stats["prompt_tokens"]))
                user_avg_prompt_chars.append(sum(stats["prompt_chars"]) / len(stats["prompt_chars"]))
        
        return {
            "metadata": {
                "total_samples": total_samples,
                "total_users": num_users,
                "total_original_history_count": total_original_count,
                "total_filtered_history_count": total_filtered_count,
                "total_actual_used_history_count": total_actual_used_count,
                "total_actual_used_tokens": total_actual_used_tokens,
                "avg_history_per_sample": round(total_actual_used_count / total_samples, 2) if total_samples > 0 else 0,
                "avg_history_tokens_per_sample": round(total_actual_used_tokens / total_samples, 2) if total_samples > 0 else 0,
                "total_prompt_tokens": total_prompt_tokens,
                "avg_prompt_tokens_per_sample": round(total_prompt_tokens / total_samples, 2) if total_samples > 0 else 0,
                "avg_history_per_user": round(sum(user_avg_history_count) / len(user_avg_history_count), 2) if user_avg_history_count else 0,
                "avg_history_tokens_per_user": round(sum(user_avg_history_tokens) / len(user_avg_history_tokens), 2) if user_avg_history_tokens else 0,
                "avg_prompt_tokens_per_user": round(sum(user_avg_prompt_tokens) / len(user_avg_prompt_tokens), 2) if user_avg_prompt_tokens else 0,
            },
            "scene_distribution": scene_distribution,
            "action_distribution": action_distribution,
            "per_scene_actions": per_scene_actions,
        }
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate all evaluation metrics; returns binary_metrics, continuous_metrics, llm_judge_metrics, overall."""
        if self.verbose:
            print("\nCalculating evaluation metrics...")

        binary_data = {}
        continuous_data = {}
        text_data = {}

        successful_api_calls = 0
        total_questions = 0
        failed_parsing_count = 0
        filtered_samples = 0
        failed_binary_questions = 0
        total_retries = 0

        logprobs_predictions = 0
        direct_mapping_predictions = 0
        
        for result in results:
            if result.get("filtered", False):
                filtered_samples += 1
                continue
            
            if result["success"]:
                successful_api_calls += 1
                
                if result.get("binary_mode"):
                    failed_binary_questions += len(result.get("failed_questions", []))
                    for q in result.get("failed_questions", []):
                        total_retries += q.get("retry_count", 0)
                
                for q in result["questions"]:
                    total_questions += 1
                    field = q["field"]
                    q_type = q["type"]
                    true_val = q["true_value"]
                    pred_val = q["predicted_value"]
                    
                    if q.get("prediction_method") == "logprobs":
                        logprobs_predictions += 1
                    elif q.get("prediction_method") == "direct_mapping":
                        direct_mapping_predictions += 1

                    total_retries += q.get("retry_count", 0)

                    if pred_val is None:
                        failed_parsing_count += 1
                        continue

                    if q_type == "binary":
                        if field in {"video_downloaded", "ad_converted"}:
                            continue
                        if field not in binary_data:
                            binary_data[field] = {"y_true": [], "y_pred_labels": [], "y_pred_probs": []}
                        try:
                            predicted_label = q.get("predicted_label")
                            if predicted_label is None:
                                # fallback for old data without predicted_label
                                predicted_label = 1 if float(pred_val) >= 0.5 else 0

                            binary_data[field]["y_true"].append(int(true_val))
                            binary_data[field]["y_pred_labels"].append(int(predicted_label))
                            binary_data[field]["y_pred_probs"].append(float(pred_val))
                        except (ValueError, TypeError):
                            failed_parsing_count += 1

                    elif q_type == "continuous":
                        if field not in continuous_data:
                            continuous_data[field] = {"y_true": [], "y_pred": [], "normalizers": []}
                        try:
                            continuous_data[field]["y_true"].append(float(true_val))
                            continuous_data[field]["y_pred"].append(float(pred_val))
                            normalizer = q.get("video_duration")
                            # fallback for old data: extract duration from scene_info.context
                            if normalizer is None and field == "video_watch_seconds":
                                scene_info = result.get("scene_info", {})
                                context = scene_info.get("context", {})
                                duration_raw = context.get("duration")
                                if duration_raw is not None:
                                    try:
                                        if isinstance(duration_raw, (int, float)):
                                            normalizer = float(duration_raw) if duration_raw > 0 else None
                                        elif isinstance(duration_raw, str):
                                            val = float(duration_raw.replace("秒", "").strip()) if duration_raw else 0
                                            normalizer = val if val > 0 else None
                                    except (ValueError, TypeError):
                                        normalizer = None
                            if normalizer is not None:
                                try:
                                    continuous_data[field]["normalizers"].append(float(normalizer))
                                except (ValueError, TypeError):
                                    continuous_data[field]["normalizers"].append(None)
                            else:
                                continuous_data[field]["normalizers"].append(None)
                        except (ValueError, TypeError):
                            failed_parsing_count += 1
                    
                    elif q_type == "text":
                        if field not in text_data:
                            text_data[field] = {"references": [], "hypotheses": []}
                        text_data[field]["references"].append(str(true_val))
                        text_data[field]["hypotheses"].append(str(pred_val))
        
        binary_metrics = {}
        for field, data in binary_data.items():
            if self.verbose:
                print(f"  Binary metrics: {field} (n={len(data['y_true'])})")
            if self.verbose:
                binary_metrics[field] = calculate_all_binary_metrics(
                    data["y_true"],
                    data["y_pred_labels"],
                    data["y_pred_probs"],
                    field_name=field
                )
            else:
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    binary_metrics[field] = calculate_all_binary_metrics(
                        data["y_true"],
                        data["y_pred_labels"],
                        data["y_pred_probs"],
                        field_name=field
                    )

        continuous_metrics = {}
        for field, data in continuous_data.items():
            if self.verbose:
                print(f"  Continuous metrics: {field} (n={len(data['y_true'])})")
            normalizers = data.get("normalizers", [])
            has_valid_normalizers = any(n is not None and n > 0 for n in normalizers)
            if self.verbose:
                continuous_metrics[field] = calculate_all_continuous_metrics(
                    data["y_true"], data["y_pred"],
                    normalizers=normalizers if has_valid_normalizers else None
                )
            else:
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    continuous_metrics[field] = calculate_all_continuous_metrics(
                        data["y_true"], data["y_pred"],
                        normalizers=normalizers if has_valid_normalizers else None
                    )

        micro_macro_f1 = calculate_micro_macro_f1(binary_metrics)
        if self.verbose:
            print(f"  Micro F1: {micro_macro_f1['Micro_F1']:.4f}, Macro F1: {micro_macro_f1['Macro_F1']:.4f}")

        valid_samples = len(results) - filtered_samples
        overall = {
            "total_predictions": len(results),
            "filtered_samples": filtered_samples,
            "valid_samples": valid_samples,
            "successful_api_calls": successful_api_calls,
            "api_success_rate": successful_api_calls / valid_samples if valid_samples > 0 else 0,
            "total_questions": total_questions,
            "failed_parsing_questions": failed_parsing_count,
            "failed_binary_questions": failed_binary_questions,
            "total_retries": total_retries,
            "parsing_success_rate": 1 - (failed_parsing_count / total_questions) if total_questions else 0,
            "logprobs_predictions": logprobs_predictions,
            "direct_mapping_predictions": direct_mapping_predictions,
            "Micro_F1": micro_macro_f1["Micro_F1"],
            "Macro_F1": micro_macro_f1["Macro_F1"],
            "Micro_Precision": micro_macro_f1["Micro_Precision"],
            "Micro_Recall": micro_macro_f1["Micro_Recall"],
            "Total_TP": micro_macro_f1["Total_TP"],
            "Total_FP": micro_macro_f1["Total_FP"],
            "Total_FN": micro_macro_f1["Total_FN"],
        }
        
        llm_judge_data = {
            "intent_fidelity": [],
            "persona_mimicry": [],
            "knowledge_boundary": [],
            "semantic_alignment": [],
            "average_score": []
        }
        
        has_llm_judge_scores = False
        
        for result in results:
            if "llm_judge_scores" in result:
                has_llm_judge_scores = True
                scores = result["llm_judge_scores"]
                llm_judge_data["intent_fidelity"].append(scores.get("intent_fidelity", 0))
                llm_judge_data["persona_mimicry"].append(scores.get("persona_mimicry", 0))
                llm_judge_data["knowledge_boundary"].append(scores.get("knowledge_boundary", 0))
                llm_judge_data["semantic_alignment"].append(scores.get("semantic_alignment", 0))
                llm_judge_data["average_score"].append(result.get("llm_judge_avg_score", 0))

        llm_judge_metrics = {}
        if has_llm_judge_scores:
            if self.verbose:
                print(f"  LLM Judge metrics (n={len(llm_judge_data['average_score'])})")
            for key, values in llm_judge_data.items():
                if values:
                    llm_judge_metrics[key] = round(sum(values) / len(values), 5)
                else:
                    llm_judge_metrics[key] = 0.0

        final_score = calculate_final_score_components(
            binary_metrics,
            continuous_metrics,
            llm_judge_metrics,
        )
        score_summary = build_score_summary(self.model_name, final_score)
        overall["final_score"] = final_score["final_score"]
        overall["final_score_available_component_count"] = final_score["available_component_count"]
        overall["final_score_missing_components"] = final_score["missing_components"]

        self.run_summary["metrics"] = {
            "score_summary": score_summary,
            "overall": overall,
            "llm_judge_metrics": llm_judge_metrics,
            "binary_field_count": len(binary_metrics),
            "continuous_field_count": len(continuous_metrics),
        }
        self._save_run_summary()

        score_summary_print = round_floats(score_summary, decimals=2)
        print(json.dumps({"score_summary": score_summary_print}, ensure_ascii=False))

        return {
            "binary_metrics": binary_metrics,
            "continuous_metrics": continuous_metrics,
            "llm_judge_metrics": llm_judge_metrics,
            "final_score": final_score,
            "score_summary": score_summary,
            "overall": overall,
        }
    
    def save_metrics_report(self, metrics: Dict, output_path: str):
        """Save the metrics report as JSON only (dated subdirectory, floats rounded to 2 dp)."""
        date_dir = self.run_date
        timestamp = self.run_timestamp

        output_dir = os.path.dirname(output_path)
        filename = os.path.basename(output_path)
        if not filename.endswith(".json"):
            filename = os.path.splitext(filename)[0] + ".json"
        timestamp_output_dir = os.path.join(output_dir, date_dir, timestamp)
        output_path = os.path.join(timestamp_output_dir, filename)

        os.makedirs(timestamp_output_dir, exist_ok=True)

        metrics_percentage = convert_metrics_to_percentage(metrics)
        metrics_rounded = round_floats(metrics_percentage, decimals=2)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_rounded, f, ensure_ascii=False, indent=2)

        if self.run_summary:
            self.run_summary.setdefault("files", {})["metrics_report"] = output_path
            self.run_summary["metrics_report_json"] = output_path
            self._save_run_summary()

        print(json.dumps({"metrics_report_json": output_path}, ensure_ascii=False))


SKIP_SCENE_TYPES = normalize_action_type_set(set())


def load_fixed_experiment_data(data_path: str) -> Tuple[List[Dict], Dict]:
    """
    Load data from a fixed experiment data file.

    The data structure supports rolling prediction with separated time ranges:
    - base_history: historical actions (e.g. September)
    - test_time_all_actions: all actions in the test window (for incremental history)
    - test_actions: sampled actions to predict (includes test_time_index)

    Actions whose type is in SKIP_SCENE_TYPES are filtered out automatically.

    Returns:
        Tuple of (eval_data, metadata)
    """
    print(f"Loading from fixed dataset: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        experiment_data = json.load(f)

    metadata = experiment_data.get("metadata", {})
    users = experiment_data.get("users", [])

    print(f"  Metadata:")
    print(f"    Users: {metadata.get('num_users')}")
    print(f"    Test length: {metadata.get('test_length')}")
    print(f"    Total test actions: {metadata.get('total_test_actions')}")
    print(f"    Created at: {metadata.get('created_at')}")
    
    max_history_tokens = metadata.get('max_history_tokens')
    if max_history_tokens:
        print(f"    Max history tokens: {max_history_tokens}")

    if 'history_token_statistics' in metadata:
        token_stats = metadata['history_token_statistics']
        print(f"    Base history token stats:")
        print(f"      Avg per user: {token_stats.get('avg_tokens_per_user', 0):,.0f}")
        print(f"      Min: {token_stats.get('min_tokens', 0):,}")
        print(f"      Max: {token_stats.get('max_tokens', 0):,}")

    if 'time_distribution' in metadata:
        time_dist = metadata['time_distribution']
        print(f"    Time ranges:")
        if 'history' in time_dist:
            print(f"      Base history: {time_dist['history'].get('earliest', 'N/A')} ~ {time_dist['history'].get('latest', 'N/A')}")
        if 'test' in time_dist:
            print(f"      Test set: {time_dist['test'].get('earliest', 'N/A')} ~ {time_dist['test'].get('latest', 'N/A')}")

    total_skipped_actions = 0
    skipped_actions_by_type = {}
    eval_data = []
    total_original_actions = 0
    total_valid_actions = 0

    for user in users:
        filtered_test_actions = []
        for t in user["test_actions"]:
            action = t["action"]
            action_type = get_action_type(action)
            total_original_actions += 1

            if action_type in SKIP_SCENE_TYPES:
                total_skipped_actions += 1
                skipped_actions_by_type[action_type] = skipped_actions_by_type.get(action_type, 0) + 1
            else:
                filtered_test_actions.append({
                    "action": action,
                    "test_time_index": t["test_time_index"]
                })
                total_valid_actions += 1

        if filtered_test_actions:
            eval_data.append({
                "user_id": user["user_id"],
                "user_profile": user["user_profile"],
                "base_history": user["base_history"],
                "test_time_all_actions": user["test_time_all_actions"],
                "test_actions": filtered_test_actions
            })

    if total_skipped_actions > 0:
        print(f"\n⚠️  Filtered {total_skipped_actions} actions from skipped scene types:")
        print(f"    Skipped types: {SKIP_SCENE_TYPES}")
        for scene_type, count in skipped_actions_by_type.items():
            print(f"      {scene_type}: {count} actions")
        print(f"    Valid actions: {total_valid_actions} / {total_original_actions}")

    print(f"✓ Loaded experiment data for {len(eval_data)} users ({total_valid_actions} actions to evaluate)")

    metadata["skipped_actions_count"] = total_skipped_actions
    metadata["skipped_actions_by_type"] = skipped_actions_by_type
    metadata["skipped_scene_types"] = list(SKIP_SCENE_TYPES)
    metadata["actual_users_count"] = len(eval_data)
    metadata["actual_test_actions_count"] = total_valid_actions
    
    return eval_data, metadata


def main():
    """Main evaluation pipeline for a single model."""
    import argparse

    parser = argparse.ArgumentParser(description="User simulation evaluation — single-model rolling prediction")
    parser.add_argument("--use-fixed-data", type=str, default=None,
                       help="Path to fixed experiment data file (overrides config.py)")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Max concurrency per model (default: from config.py)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name to evaluate, e.g. --model Qwen3-8B")
    parser.add_argument("--max-history-tokens", type=int, default=None,
                       help="Max token limit for history (overrides dataset config)")
    parser.add_argument("--max-history-days", type=int, default=None,
                       help="Max history day window (overrides dataset config)")
    parser.add_argument("--skip-model-eval", action="store_true",
                       help="Skip model calls and produce mock predictions (for SFT data export)")
    parser.add_argument("--judge-model", type=str, default=None,
                       help="Override teacher model name from config.py for shop text scoring")
    parser.add_argument("--judge-max-retries", type=int, default=3,
                       help="Max retries per LLM judge record")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed per-field metric and token logs")

    args = parser.parse_args()
    
    print("=" * 80)
    print("User Simulation Evaluation System - Rolling Prediction")
    print("=" * 80)

    print("\nStep 1: Load and prepare data")
    print("-" * 80)

    fixed_data_path = args.use_fixed_data or DEFAULT_EXPERIMENT_DATA_PATH

    if not os.path.exists(fixed_data_path):
        print(f"Error: dataset not found: {fixed_data_path}")
        print(f"Please run: python src/data/prepare_experiment_data.py")
        return

    print(f"Using experiment dataset: {fixed_data_path}")
    eval_data, metadata = load_fixed_experiment_data(fixed_data_path)
    experiment_name = os.path.splitext(os.path.basename(fixed_data_path))[0]

    # strip verbose prefixes, e.g. multi_scene_user_stats_filtered_output_top50_30t -> top50_30t
    redundant_prefixes = [
        "multi_scene_user_stats_filtered_output_",
        "multi_scene_user_stats_",
        "multi_scene_",
    ]
    for prefix in redundant_prefixes:
        if experiment_name.startswith(prefix):
            experiment_name = experiment_name[len(prefix):]
            break
    
    # CLI args take priority over metadata values
    max_history_tokens = args.max_history_tokens if args.max_history_tokens is not None else metadata.get('max_history_tokens')
    max_history_days = args.max_history_days if args.max_history_days is not None else metadata.get('max_history_days')

    if args.max_history_tokens is not None and args.max_history_tokens != metadata.get('max_history_tokens'):
        print(f"⚠️  CLI override: max_history_tokens = {args.max_history_tokens} (dataset: {metadata.get('max_history_tokens')})")
    if args.max_history_days is not None and args.max_history_days != metadata.get('max_history_days'):
        print(f"⚠️  CLI override: max_history_days = {args.max_history_days} (dataset: {metadata.get('max_history_days')})")

    eval_params_suffix = ""

    if max_history_tokens and max_history_tokens > 0:
        tokens_k = max_history_tokens // 1000
        if f"_{tokens_k}k" not in experiment_name and f"{tokens_k}k" not in experiment_name:
            eval_params_suffix += f"_{tokens_k}k"
        print(f"Max history tokens: {max_history_tokens} ({tokens_k}K)")

    if max_history_days and max_history_days > 0:
        if f"_{max_history_days}d" not in experiment_name and f"{max_history_days}d" not in experiment_name:
            eval_params_suffix += f"_{max_history_days}d"
        print(f"Max history days: {max_history_days}")

    if eval_params_suffix:
        experiment_name = f"{experiment_name}{eval_params_suffix}"
        print(f"Result filename includes eval params: {experiment_name}")

    if not eval_data:
        print("No matching user data found, exiting")
        return

    print("\nStep 2: Determine model to evaluate")
    print("-" * 80)

    try:
        from config import MODELS_TO_EVALUATE

        if not args.model:
            print("❌ Error: --model argument is required")
            print("\nAvailable models:")
            for m in MODELS_TO_EVALUATE:
                print(f"  - {m['name']}")
            print("\nExamples:")
            print("  python src/evaluation/evaluator.py --model gpt-5")
            return

        model_name = args.model
        model_config = next((m for m in MODELS_TO_EVALUATE if m["name"] == model_name), None)

        if not model_config:
            print(f"❌ Error: model not found in config: {model_name}")
            print("\nAvailable models:")
            for m in MODELS_TO_EVALUATE:
                print(f"  - {m['name']}")
            return

        model_info = model_config.get('model', 'N/A')
        print(f"✓ Model: {model_config['name']} ({model_config['type']}, {model_info})")

    except ImportError:
        print("❌ Error: MODELS_TO_EVALUATE not found in config.py")
        print("Please ensure config.py defines MODELS_TO_EVALUATE")
        return

    shanghai_tz = pytz.timezone('Asia/Shanghai')
    now_shanghai = datetime.now(shanghai_tz)
    run_date = now_shanghai.strftime("%Y-%m-%d")
    run_timestamp = now_shanghai.strftime("%Y-%m-%d_%H-%M-%S")

    print(f"\nRun time (Shanghai): {run_timestamp}")
    print(f"All model results will be saved to:")
    print(f"  - output/{run_date}/{run_timestamp}/")
    print(f"  - results/{run_date}/{run_timestamp}/")

    print(f"\nStep 3: Run evaluation")
    print("=" * 80)

    model_workers = args.max_workers
    if model_workers is None:
        model_workers = _resolve_model_workers(model_config)
        print(f"Concurrency: {_describe_model_workers(model_config)}")
    else:
        print(f"Concurrency: {model_workers} (CLI override)")
    print()

    model_experiment_name = f"{experiment_name}_{model_config['name']}"

    try:
        evaluator = UserSimulationEvaluator(
            model_config=model_config,
            experiment_name=model_experiment_name,
            run_date=run_date,
            run_timestamp=run_timestamp,
            max_history_tokens=max_history_tokens,
            max_history_days=max_history_days,
            skip_model_eval=args.skip_model_eval,
            verbose=args.verbose,
        )

        results = evaluator.evaluate_all(eval_data, max_workers=model_workers)

        results = run_llm_judge_evaluation(
            results,
            judge_model_name=args.judge_model,
            max_retries=args.judge_max_retries,
        )
        evaluator.record_llm_judge_summary(results)

        print(f"\nCalculating metrics for {model_config['name']}...")
        metrics = evaluator.calculate_metrics(results)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        report_path = f"{RESULTS_DIR}/{model_experiment_name}_evaluation_report.json"
        evaluator.save_metrics_report(metrics, report_path)
        metrics["run_summary_path"] = evaluator.run_summary_path

        score_summary = {
            **round_floats(metrics.get("score_summary", {}), decimals=2),
            "run_summary_json": metrics.get("run_summary_path"),
        }
        print(f"✓ {model_config['name']} evaluation complete")
        print(json.dumps({"evaluation_complete": True, "score_summary": score_summary}, ensure_ascii=False))

    except Exception as e:
        print(f"❌ {model_config['name']} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
