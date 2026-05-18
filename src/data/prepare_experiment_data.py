import json
import os
import sys
import random
import hashlib
import glob
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from action_types import get_action_type
from data.data_processor import load_user_data
from models.prompt_builder import estimate_token_count, format_action_context, format_action_result, should_filter_action
from datetime import datetime
import pytz


DEFAULT_MAX_HISTORY_TOKENS = 0


def _resolve_project_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(__file__).parent.parent.parent / p


def _load_seed_offsets() -> dict:
    """Load pre-computed seed offsets from raw_user_data/seed_offsets.json if present."""
    seed_offsets_path = _resolve_project_path("raw_user_data") / "seed_offsets.json"
    if seed_offsets_path.exists():
        with open(seed_offsets_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_stable_user_seed_offset(user_id: str, seed_offsets: dict = None) -> int:
    """Return a stable seed offset for a user. Uses pre-computed values from seed_offsets.json if available, else SHA256 hash."""
    if seed_offsets and user_id in seed_offsets:
        return seed_offsets[user_id]
    hash_obj = hashlib.sha256(user_id.encode('utf-8'))
    hash_bytes = hash_obj.digest()[:8]
    return int.from_bytes(hash_bytes, byteorder='big') % (10**9)


def generate_config_signature(
    num_users: int,
    test_length: int,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    seed: int = 42,
    input_data_path: str = None,
) -> str:
    """Generate an 8-char hash signature from sampling parameters only. Evaluation params (max_history_tokens, etc.) are excluded so the same dataset can be reused across eval configs."""
    config_str = f"{num_users}|{test_length}|" \
                 f"{history_time_start or ''}|{history_time_end or ''}|" \
                 f"{test_time_start or ''}|{test_time_end or ''}|" \
                 f"{seed}|{input_data_path or ''}"
    hash_obj = hashlib.md5(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def generate_descriptive_dirname(
    num_users: int,
    test_length: int,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    seed: int = 42,
    signature: str = None
) -> str:
    """Generate a descriptive directory name from sampling params. Format: 200u_30t_h~0930_t1001~1130_s42_abc12345"""
    parts = [
        f"{num_users}u",
        f"{test_length}t",
    ]
    if history_time_start or history_time_end:
        h_start = history_time_start[5:7] + history_time_start[8:10] if history_time_start else ""
        h_end = history_time_end[5:7] + history_time_end[8:10] if history_time_end else ""
        parts.append(f"h{h_start}~{h_end}")
    if test_time_start or test_time_end:
        t_start = test_time_start[5:7] + test_time_start[8:10] if test_time_start else ""
        t_end = test_time_end[5:7] + test_time_end[8:10] if test_time_end else ""
        parts.append(f"t{t_start}~{t_end}")
    parts.append(f"s{seed}")
    if signature:
        parts.append(signature)
    
    return "_".join(parts)


def find_existing_dataset(
    output_dir: str,
    signature: str
) -> Optional[str]:
    """Find an existing dataset matching the given signature. Returns the JSON file path if found, else None."""
    if not os.path.exists(output_dir):
        return None
    pattern = os.path.join(output_dir, "*", f"*_{signature}", "*.json")
    
    matching_files = glob.glob(pattern)
    # exclude analysis/distribution files, keep main data files
    main_files = [
        f for f in matching_files 
        if not f.endswith("_analysis.json") 
        and not f.endswith("_test_distribution.json")
        and not os.path.basename(f).endswith("_sampled_actions.json")
    ]
    
    if main_files:
        main_files.sort(key=os.path.getmtime, reverse=True)
        return main_files[0]
    
    return None


# High-value action types per scene
HIGH_VALUE_ACTIONS = {
    "视频浏览": {"like", "collect", "share", "comment"},
    "直播间": {"send_gift", "add_to_cart", "follow"},
    "商城购物": {"purchase", "order_success", "add_to_cart"},
    "广告推荐": {"conversion", "activation", "purchase", "submit", "click"},
    "电商客服对话": {"purchase"},
}


def _get_action_summary(action: Dict, max_length: int = 100) -> str:
    """Return a short text summary of an action, truncated to max_length."""
    action_type = get_action_type(action)
    context = action.get("context", {})
    
    if action_type == "视频浏览":
        caption = context.get("caption", "")
        if caption:
            return caption[:max_length] + ("..." if len(caption) > max_length else "")
        return "untitled video"

    elif action_type == "直播间":
        live_title = context.get("live_title", "")
        live_category = context.get("live_category", "")
        if live_title:
            return f"[{live_category}] {live_title}"[:max_length]
        return f"[{live_category}] live stream"

    elif action_type == "商城购物":
        product_name = context.get("product_name", context.get("item_name", ""))
        if product_name:
            return product_name[:max_length]
        return "product browse"

    elif action_type == "广告推荐":
        ad_title = context.get("ad_title", context.get("title", ""))
        if ad_title:
            return ad_title[:max_length]
        return "ad content"

    else:
        for key in ["title", "name", "content", "description", "caption"]:
            if key in context and context[key]:
                return str(context[key])[:max_length]
        return f"{action_type} action"


def is_high_value_action(item: Dict) -> bool:
    """Return True if the item contains at least one high-value action for its domain."""
    domain = get_action_type(item)
    if domain not in HIGH_VALUE_ACTIONS:
        return False
    target_actions = HIGH_VALUE_ACTIONS[domain]
    actions = item.get("action", [])
    if not isinstance(actions, list):
        return False
    
    for act in actions:
        if act.get("type") in target_actions:
            return True
    
    return False


def has_sufficient_context_for_prediction(action: Dict) -> bool:
    """Return True if the action's context has enough text fields for prediction.

    Video actions require at least one of caption/ocr/asr.
    Live-stream actions require at least live_title or live_category.
    All other action types are always accepted.
    """
    action_type = get_action_type(action)
    context = action.get("context", {})

    if action_type == "视频浏览":
        caption = context.get("caption", "")
        ocr = context.get("ocr", "")
        asr = context.get("asr", "")
        if (not caption or not caption.strip()) and \
           (not ocr or not ocr.strip()) and \
           (not asr or not asr.strip()):
            return False

    elif action_type == "直播间":
        live_title = context.get("live_title", "")
        live_category = context.get("live_category", "")
        if (not live_title or not live_title.strip()) and (not live_category or not live_category.strip()):
            return False
    
    return True


def split_user_actions_by_tokens(
    user_data: Dict,
    test_length: int,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    seed: int = None
) -> Tuple[Dict, List[Dict], int, List[Dict]]:
    """Split a user's actions into history and test sets using separate time ranges.

    Sampling uses balanced selection (time + domain + high/low-value balance).
    Only actions with insufficient context are skipped; the rest are sampled to fill test_length.
    max_history_tokens is not applied here; it is an evaluation-time concern.

    Returns:
        (history_data, test_actions_with_index, actual_tokens, skipped_actions_info)
    """
    all_original_actions = user_data["action_history"]
    
    use_separated_time_range = (history_time_start or history_time_end or test_time_start or test_time_end)
    base_history_pool = []
    for action in all_original_actions:
        timestamp = action.get("timestamp", "")
        in_range = True
        if history_time_start and timestamp < history_time_start:
            in_range = False
        if history_time_end and timestamp > history_time_end:
            in_range = False
        if in_range:
            base_history_pool.append(action)
    test_time_all_actions = []
    for action in all_original_actions:
        timestamp = action.get("timestamp", "")
        in_range = True
        if test_time_start and timestamp < test_time_start:
            in_range = False
        if test_time_end and timestamp > test_time_end:
            in_range = False
        if in_range:
            test_time_all_actions.append(action)
    test_candidates_all = []
    skipped_actions_info = []
    
    for idx, action in enumerate(test_time_all_actions):
        action_type = get_action_type(action)
        
        if not has_sufficient_context_for_prediction(action):
            skipped_actions_info.append({
                "action": action,
                "skip_reason": "insufficient context",
                "action_type": action_type,
                "timestamp": action.get("timestamp", "")
            })
            continue
        
        if should_filter_action(action):
            skipped_actions_info.append({
                "action": action,
                "skip_reason": "filtered at eval time (e.g. play_duration=0)",
                "action_type": action_type,
                "timestamp": action.get("timestamp", "")
            })
            continue
        
        test_candidates_all.append({"action": action, "test_time_index": idx})
    
    test_actions_with_index = _balanced_sample(
        test_candidates_all,
        test_length,
        m_buckets=10,
        seed=seed
    )
    
    filtered_base_history = base_history_pool
    base_history_tokens = 0
    for action in filtered_base_history:
        timestamp = action.get("timestamp", "unknown time")
        action_type = get_action_type(action, "unknown action")
        context_str = format_action_context(action)
        result_str = format_action_result(action)

        action_text = (
            f"【行为】时间：{timestamp}\n"
            f"  场景：{action_type}\n"
            f"  详情：{context_str}\n"
            f"  反应：{result_str}\n"
        )
        base_history_tokens += estimate_token_count(action_text)
    
    result_data = {
        "user_profile": user_data["user_profile"],
        "base_history": base_history_pool,
        "test_time_all_actions": test_time_all_actions,
    }
    
    return result_data, test_actions_with_index, base_history_tokens, skipped_actions_info


def _balanced_sample(
    candidates_all: List[Dict],
    total_count: int,
    m_buckets: int = 10,
    seed: int = None
) -> List[Dict]:
    """Balanced sampling: time-balanced buckets with round-robin domain and 50/50 high/low-value selection within each bucket."""
    if seed is not None:
        random.seed(seed)
    if not candidates_all or total_count <= 0:
        return []
    
    n = len(candidates_all)
    if n <= total_count:
        return candidates_all.copy()
    actual_buckets = min(m_buckets, n // 2, total_count)
    if actual_buckets < 1:
        actual_buckets = 1
    
    base_sample_per_bucket = total_count // actual_buckets
    remainder = total_count % actual_buckets
    bucket_data_size = n // actual_buckets
    buckets = []
    start_idx = 0
    for i in range(actual_buckets):
        if i == actual_buckets - 1:
            end_idx = n
        else:
            end_idx = start_idx + bucket_data_size
        buckets.append(candidates_all[start_idx:end_idx])
        start_idx = end_idx
    sampled_results = []
    for i, bucket_items in enumerate(buckets):
        if not bucket_items:
            continue
        
        target_k = base_sample_per_bucket + (1 if i < remainder else 0)
        if target_k <= 0:
            continue
        if len(bucket_items) <= target_k:
            sampled_results.extend(bucket_items)
            continue
        domain_pools = defaultdict(lambda: {'high': [], 'low': []})
        
        for item in bucket_items:
            action = item.get("action", item)  # support both formats
            domain = get_action_type(action, "unknown")
            is_high = is_high_value_action(action)
            key = 'high' if is_high else 'low'
            domain_pools[domain][key].append(item)
        
        active_domains = list(domain_pools.keys())
        random.shuffle(active_domains)
        
        bucket_sampled = []
        domain_idx = 0
        
        while len(bucket_sampled) < target_k and active_domains:
            current_domain = active_domains[domain_idx]
            pool = domain_pools[current_domain]
            has_high = len(pool['high']) > 0
            has_low = len(pool['low']) > 0
            if not has_high and not has_low:
                active_domains.pop(domain_idx)
                if active_domains:
                    domain_idx = domain_idx % len(active_domains)
                continue
            
            if has_high and has_low:
                pick_high = random.random() < 0.5
            elif has_high:
                pick_high = True
            else:
                pick_high = False
            
            if pick_high:
                pop_idx = random.randint(0, len(pool['high']) - 1)
                selected_item = pool['high'].pop(pop_idx)
            else:
                pop_idx = random.randint(0, len(pool['low']) - 1)
                selected_item = pool['low'].pop(pop_idx)
            
            bucket_sampled.append(selected_item)
            if active_domains:
                domain_idx = (domain_idx + 1) % len(active_domains)
        
        sampled_results.extend(bucket_sampled)
    
    sampled_results.sort(key=lambda x: x.get("test_time_index", 0))
    
    return sampled_results


def analyze_sampled_actions(experiment_data: Dict, output_path: str):
    """Analyze sampled test actions and save a JSON report to output_path."""
    analysis = {
        "summary": {},
        "time_analysis": {},
        "type_distribution": {},
        "action_distribution": {},
        "value_distribution": {},
        "per_user_stats": []
    }
    
    all_sampled_actions = []
    all_timestamps = []
    type_counts = defaultdict(int)
    action_counts = defaultdict(int)
    high_value_count = 0
    low_value_count = 0
    
    for user_entry in experiment_data["users"]:
        user_id = user_entry["user_id"]
        test_actions = user_entry.get("test_actions", [])
        
        user_type_counts = defaultdict(int)
        user_action_counts = defaultdict(int)
        user_high_value = 0
        user_low_value = 0
        user_timestamps = []
        
        for test_item in test_actions:
            action = test_item["action"]
            all_sampled_actions.append(action)
            
            timestamp = action.get("timestamp", "")
            if timestamp:
                all_timestamps.append(timestamp)
                user_timestamps.append(timestamp)
            
            action_type = get_action_type(action, "unknown")
            type_counts[action_type] += 1
            user_type_counts[action_type] += 1
            
            actions_list = action.get("action", [])
            if isinstance(actions_list, list):
                for act in actions_list:
                    act_type = act.get("type", "unknown")
                    action_counts[act_type] += 1
                    user_action_counts[act_type] += 1
            
            if is_high_value_action(action):
                high_value_count += 1
                user_high_value += 1
            else:
                low_value_count += 1
                user_low_value += 1
        
        user_stats = {
            "user_id": user_id,
            "sampled_count": len(test_actions),
            "time_range": {
                "earliest": min(user_timestamps) if user_timestamps else None,
                "latest": max(user_timestamps) if user_timestamps else None
            },
            "type_distribution": dict(user_type_counts),
            "action_distribution": dict(user_action_counts),
            "high_value_count": user_high_value,
            "low_value_count": user_low_value,
            "high_value_ratio": user_high_value / len(test_actions) if test_actions else 0
        }
        analysis["per_user_stats"].append(user_stats)
    
    total_sampled = len(all_sampled_actions)
    analysis["summary"] = {
        "total_users": len(experiment_data["users"]),
        "total_sampled_actions": total_sampled,
        "avg_actions_per_user": total_sampled / len(experiment_data["users"]) if experiment_data["users"] else 0,
        "seed": experiment_data["metadata"].get("seed", "unknown"),
        "sampling_strategy": experiment_data["metadata"].get("sampling_strategy", "balanced")
    }
    
    if all_timestamps:
        sorted_timestamps = sorted(all_timestamps)
        analysis["time_analysis"] = {
            "earliest": sorted_timestamps[0],
            "latest": sorted_timestamps[-1],
            "total_count": len(sorted_timestamps),
            "by_month": {}
        }
        month_counts = defaultdict(int)
        for ts in all_timestamps:
            month = ts[:7] if len(ts) >= 7 else "unknown"
            month_counts[month] += 1
        analysis["time_analysis"]["by_month"] = dict(sorted(month_counts.items()))
        day_counts = defaultdict(int)
        for ts in all_timestamps:
            day = ts[:10] if len(ts) >= 10 else "unknown"
            day_counts[day] += 1
        sorted_days = sorted(day_counts.items())
        analysis["time_analysis"]["by_day_sample"] = {
            "first_10_days": dict(sorted_days[:10]),
            "last_10_days": dict(sorted_days[-10:]) if len(sorted_days) > 10 else {}
        }
    
    analysis["type_distribution"] = {
        "counts": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "ratios": {
            k: round(v / total_sampled * 100, 2) 
            for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
        } if total_sampled > 0 else {}
    }
    
    analysis["action_distribution"] = {
        "counts": dict(sorted(action_counts.items(), key=lambda x: -x[1])),
        "ratios": {
            k: round(v / sum(action_counts.values()) * 100, 2) 
            for k, v in sorted(action_counts.items(), key=lambda x: -x[1])
        } if action_counts else {}
    }
    
    analysis["value_distribution"] = {
        "high_value_count": high_value_count,
        "low_value_count": low_value_count,
        "high_value_ratio": round(high_value_count / total_sampled * 100, 2) if total_sampled > 0 else 0,
        "low_value_ratio": round(low_value_count / total_sampled * 100, 2) if total_sampled > 0 else 0
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "=" * 80)
    print("Sampled actions analysis report")
    print("=" * 80)
    print(f"\n[Summary]")
    print(f"  Total users: {analysis['summary']['total_users']}")
    print(f"  Total sampled actions: {analysis['summary']['total_sampled_actions']}")
    print(f"  Avg per user: {analysis['summary']['avg_actions_per_user']:.1f}")
    print(f"  Seed: {analysis['summary']['seed']}")
    print(f"\n[Time range]")
    if analysis["time_analysis"]:
        print(f"  Earliest: {analysis['time_analysis']['earliest']}")
        print(f"  Latest: {analysis['time_analysis']['latest']}")
        print(f"  By month:")
        for month, count in analysis["time_analysis"]["by_month"].items():
            print(f"    {month}: {count}")
    print(f"\n[Scene type distribution]")
    for action_type, count in analysis["type_distribution"]["counts"].items():
        ratio = analysis["type_distribution"]["ratios"].get(action_type, 0)
        print(f"  {action_type}: {count} ({ratio}%)")
    print(f"\n[Action distribution] Top 10")
    for i, (act_type, count) in enumerate(list(analysis["action_distribution"]["counts"].items())[:10]):
        ratio = analysis["action_distribution"]["ratios"].get(act_type, 0)
        print(f"  {act_type}: {count} ({ratio}%)")
    print(f"\n[High/low-value distribution]")
    print(f"  High-value: {analysis['value_distribution']['high_value_count']} ({analysis['value_distribution']['high_value_ratio']}%)")
    print(f"  Low-value: {analysis['value_distribution']['low_value_count']} ({analysis['value_distribution']['low_value_ratio']}%)")
    print(f"\nAnalysis report saved to: {output_path}")
    print("=" * 80)
    
    return analysis


def _uniform_sample_by_time(
    candidates_all: List[Dict],
    total_count: int
) -> List[Dict]:
    """Uniformly sample total_count items from a time-sorted candidate list."""
    if not candidates_all or total_count <= 0:
        return []
    
    n = len(candidates_all)
    if n <= total_count:
        return candidates_all.copy()
    result = []
    if total_count == 1:
        result.append(candidates_all[n // 2])
    else:
        for i in range(total_count):
            target_idx = int(i * (n - 1) / (total_count - 1))
            result.append(candidates_all[target_idx])
    
    return result






def select_users(
    all_users: Dict,
    num_users: int,
    test_length: int,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    min_history_actions: int = 1,
    seed: int = 42
) -> List[str]:
    """Select users that meet history and test action count thresholds."""
    random.seed(seed)
    print(f"\nFiltering users...")
    print(f"  Min history actions: {min_history_actions}")
    print(f"  Min test actions with sufficient context: {test_length}")
    if history_time_start or history_time_end:
        print(f"  History time range: {history_time_start or 'any'} ~ {history_time_end or 'any'}")
    if test_time_start or test_time_end:
        print(f"  Test time range: {test_time_start or 'any'} ~ {test_time_end or 'any'}")
    
    valid_users = []

    for user_id, user_data in all_users.items():
        actions = user_data.get("action_history", [])

        # filter history actions by time range only (no scene filter)
        history_actions = []
        for action in actions:
            timestamp = action.get("timestamp", "")
            in_range = True
            if history_time_start and timestamp < history_time_start:
                in_range = False
            if history_time_end and timestamp > history_time_end:
                in_range = False
            if in_range:
                history_actions.append(action)

        # filter test actions by time range
        test_actions = []
        for action in actions:
            timestamp = action.get("timestamp", "")
            in_range = True
            if test_time_start and timestamp < test_time_start:
                in_range = False
            if test_time_end and timestamp > test_time_end:
                in_range = False
            if in_range:
                test_actions.append(action)

        # check test set: only require sufficient context (no scene type filter)
        test_candidate_actions = [
            action for action in test_actions
            if has_sufficient_context_for_prediction(action)
        ]

        if len(test_candidate_actions) >= test_length and len(history_actions) >= min_history_actions:
            valid_users.append(user_id)

    print(f"  Found {len(valid_users)} qualifying users")

    # Sort user IDs so traversal order is consistent across runs with the same seed
    valid_users.sort()

    if len(valid_users) <= num_users:
        selected = valid_users
        print(f"  Available users <= target count; selecting all {len(selected)} users")
    else:
        selected = random.sample(valid_users, num_users)
        print(f"  Randomly selected {num_users} users")

    # Sort selected users so per-user seeds remain stable
    selected.sort()
    
    return selected


def load_user_ids_from_file(file_path: str) -> List[str]:
    """Load a list of user IDs from a JSON or plain-text file (one ID per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # try to parse as JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "users" in data:
            # format: {"users": [{"user_id": "123", ...}, ...]}
            return [u["user_id"] for u in data["users"]]
        elif isinstance(data, list):
            # format: ["123", "456", ...] or [{"user_id": "123"}, ...]
            if data and isinstance(data[0], dict):
                return [u["user_id"] for u in data]
            else:
                return [str(uid) for uid in data]
    except json.JSONDecodeError:
        pass

    # plain-text fallback: one user ID per line
    return [line.strip() for line in content.split('\n') if line.strip()]


def extract_users_from_existing_dataset(
    source_dataset_path: str,
    target_user_ids: List[str] = None,
    num_users: int = None,
    output_path: str = None,
    seed: int = 42,
) -> Dict:
    """Extract a user subset from an existing experiment dataset, preserving test_actions exactly."""
    # 1. load source dataset
    print("=" * 80)
    print("Extracting user subset from existing dataset")
    print("=" * 80)
    print(f"\n[Source dataset]")
    print(f"  Path: {source_dataset_path}")

    if not os.path.exists(source_dataset_path):
        raise FileNotFoundError(f"Source dataset not found: {source_dataset_path}")

    with open(source_dataset_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    source_users = source_data.get("users", [])
    source_metadata = source_data.get("metadata", {})

    print(f"  Users: {len(source_users)}")
    print(f"  Test actions per user: {source_metadata.get('test_length', 'N/A')}")
    print(f"  Config signature: {source_metadata.get('config_signature', 'N/A')}")

    if source_metadata.get('history_time_start') or source_metadata.get('history_time_end'):
        print(f"  History time range: {source_metadata.get('history_time_start', 'any')} ~ {source_metadata.get('history_time_end', 'any')}")
    if source_metadata.get('test_time_start') or source_metadata.get('test_time_end'):
        print(f"  Test time range: {source_metadata.get('test_time_start', 'any')} ~ {source_metadata.get('test_time_end', 'any')}")

    user_data_map = {user["user_id"]: user for user in source_users}
    available_user_ids = list(user_data_map.keys())

    # 2. determine which users to extract
    print(f"\n[User extraction]")
    if target_user_ids:
        print(f"  Mode: extract from specified user ID list")
        print(f"  Specified: {len(target_user_ids)}")

        selected_user_ids = []
        missing_users = []
        for uid in target_user_ids:
            if uid in user_data_map:
                selected_user_ids.append(uid)
            else:
                missing_users.append(uid)

        if missing_users:
            print(f"\n  Warning: {len(missing_users)} user ID(s) not found in source dataset")
            if len(missing_users) <= 5:
                print(f"    Not found: {missing_users}")
            else:
                print(f"    Not found (first 5): {missing_users[:5]}...")
            print(f"    These users may not be among the {len(source_users)} users in the source")

        print(f"\n  Matched: {len(selected_user_ids)} / {len(target_user_ids)} users")
    else:
        if num_users is None or num_users <= 0:
            raise ValueError("Must specify either target_user_ids or num_users")

        print(f"  Mode: random selection from source dataset")
        print(f"  Target count: {num_users}")
        print(f"  Seed: {seed}")

        random.seed(seed)
        available_user_ids.sort()  # sort first for reproducibility

        if num_users >= len(available_user_ids):
            selected_user_ids = available_user_ids
            print(f"\n  Requested ({num_users}) >= source users ({len(available_user_ids)})")
            print(f"  Selecting all {len(selected_user_ids)} users")
        else:
            selected_user_ids = random.sample(available_user_ids, num_users)
            print(f"\n  Randomly selected {num_users} users")

    # sort for consistency
    selected_user_ids.sort()

    # 3. extract user data
    extracted_users = [user_data_map[uid] for uid in selected_user_ids]

    print(f"\n[Extracted user sample] (first 5)")
    for i, uid in enumerate(selected_user_ids[:5], 1):
        user_entry = user_data_map[uid]
        test_count = len(user_entry.get("test_actions", []))
        history_count = user_entry.get("stats", {}).get("base_history_count", "?")
        print(f"  {i}. User {uid}: {test_count} test actions, {history_count} history actions")
    if len(selected_user_ids) > 5:
        print(f"  ... total {len(selected_user_ids)} users")

    # 4. build new metadata (copy source, update user-count fields)
    new_metadata = source_metadata.copy()
    new_metadata["num_users"] = len(extracted_users)
    new_metadata["source_dataset"] = source_dataset_path
    new_metadata["source_num_users"] = len(source_users)
    new_metadata["extraction_seed"] = seed
    new_metadata["created_at"] = __import__('datetime').datetime.now().isoformat()
    new_metadata["is_subset_extraction"] = True

    # recompute statistics
    action_type_stats = defaultdict(int)
    for user_entry in extracted_users:
        for test_item in user_entry.get("test_actions", []):
            action = test_item["action"]
            action_type = get_action_type(action, "unknown")
            action_type_stats[action_type] += 1

    new_metadata["action_type_statistics"] = dict(action_type_stats)
    new_metadata["covered_action_types"] = list(action_type_stats.keys())
    new_metadata["total_test_actions"] = sum(action_type_stats.values())

    # 5. generate new config signature
    user_ids_str = ",".join(sorted(selected_user_ids))
    user_ids_hash = hashlib.md5(user_ids_str.encode('utf-8')).hexdigest()[:8]

    # subset signature = source_signature[:4] + user_count + user_ids_hash[:4]
    source_signature = source_metadata.get("config_signature", "unknown")
    subset_signature = f"sub{source_signature[:4]}_{len(selected_user_ids)}u_{user_ids_hash[:4]}"
    new_metadata["config_signature"] = subset_signature
    new_metadata["source_config_signature"] = source_signature

    descriptive_dirname = generate_descriptive_dirname(
        num_users=len(selected_user_ids),
        test_length=source_metadata.get("test_length", 30),
        history_time_start=source_metadata.get("history_time_start"),
        history_time_end=source_metadata.get("history_time_end"),
        test_time_start=source_metadata.get("test_time_start"),
        test_time_end=source_metadata.get("test_time_end"),
        seed=seed,
        signature=subset_signature
    )
    new_metadata["descriptive_dirname"] = descriptive_dirname

    # 6. build result data
    extracted_data = {
        "metadata": new_metadata,
        "users": extracted_users
    }

    # 7. save result
    if output_path:
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        now_shanghai = datetime.now(shanghai_tz)
        run_date = now_shanghai.strftime("%Y-%m-%d")

        output_base_dir = os.path.dirname(output_path)
        filename = os.path.basename(output_path)
        timestamp_output_dir = os.path.join(output_base_dir, run_date, descriptive_dirname)
        timestamped_output_path = os.path.join(timestamp_output_dir, filename)

        os.makedirs(timestamp_output_dir, exist_ok=True)
        with open(timestamped_output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)

        print(f"\n[Saved to]")
        print(f"  {timestamped_output_path}")

        analysis_output_path = os.path.join(timestamp_output_dir, "sampled_actions_analysis.json")
        analyze_sampled_actions(extracted_data, analysis_output_path)
        
        print(f"\nACTUAL_OUTPUT_PATH={timestamped_output_path}")

    # 8. print statistics
    print("\n" + "=" * 80)
    print("Extraction statistics")
    print("=" * 80)

    print(f"\n[Overview]")
    print(f"  Source: {len(source_users)} users -> extracted: {len(extracted_users)} users")
    print(f"  Test actions per user: {source_metadata.get('test_length', 'N/A')}")
    print(f"  Total test actions: {sum(action_type_stats.values())}")

    print(f"\n[Scene type distribution]")
    for action_type, count in sorted(action_type_stats.items(), key=lambda x: -x[1]):
        percentage = count / sum(action_type_stats.values()) * 100
        print(f"  {action_type:20s}: {count:5d} ({percentage:5.1f}%)")

    print("\n" + "=" * 80)
    print("User subset extraction complete.")
    print("=" * 80)
    print(f"\nKey guarantees:")
    print(f"  - test_actions for the {len(extracted_users)} extracted users match the source experiment exactly")
    print(f"  - base_history and test_time_all_actions are fully preserved")
    print(f"  - suitable for fair comparative experiments")
    print("=" * 80)
    
    return extracted_data


def prepare_fixed_experiment_data(
    input_data_path: str,
    output_path: str,
    num_users: int = 50,
    test_length: int = 20,
    max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
    max_history_days: int = None,
    history_time_start: str = None,
    history_time_end: str = None,
    test_time_start: str = None,
    test_time_end: str = None,
    seed: int = 42,
    force_resample: bool = False,
    user_ids_file: str = None,
    specified_user_ids: List[str] = None,
):
    """Prepare a fixed experiment dataset using balanced sampling (time + domain + value balance).

    Caches results by config signature; reuses existing datasets unless force_resample=True.
    Only skips actions with insufficient context; fills up to test_length actions per user.
    """
    use_specified_users = False
    target_user_ids = None

    if specified_user_ids:
        target_user_ids = specified_user_ids
        use_specified_users = True
        print(f"Using {len(target_user_ids)} directly specified user IDs")
    elif user_ids_file:
        if not os.path.exists(user_ids_file):
            raise FileNotFoundError(f"User ID file not found: {user_ids_file}")
        target_user_ids = load_user_ids_from_file(user_ids_file)
        use_specified_users = True
        print(f"Loaded {len(target_user_ids)} user IDs from file: {user_ids_file}")

    if use_specified_users:
        num_users = len(target_user_ids)
    # generate config signature from sampling params only
    # (max_history_tokens and max_history_days are excluded)
    if use_specified_users:
        # signature is based on hash of the specified user ID list
        user_ids_str = ",".join(sorted(target_user_ids))
        user_ids_hash = hashlib.md5(user_ids_str.encode('utf-8')).hexdigest()[:8]
        config_signature = generate_config_signature(
            num_users=num_users,
            test_length=test_length,
            history_time_start=history_time_start,
            history_time_end=history_time_end,
            test_time_start=test_time_start,
            test_time_end=test_time_end,
            seed=seed,
            input_data_path=f"specified_users_{user_ids_hash}"
        )
    else:
        config_signature = generate_config_signature(
            num_users=num_users,
            test_length=test_length,
            history_time_start=history_time_start,
            history_time_end=history_time_end,
            test_time_start=test_time_start,
            test_time_end=test_time_end,
            seed=seed,
            input_data_path=input_data_path
        )
    
    descriptive_dirname = generate_descriptive_dirname(
        num_users=num_users,
        test_length=test_length,
        history_time_start=history_time_start,
        history_time_end=history_time_end,
        test_time_start=test_time_start,
        test_time_end=test_time_end,
        seed=seed,
        signature=config_signature
    )
    
    output_dir = os.path.dirname(output_path)
    
    print("=" * 80)
    print("Preparing fixed experiment dataset (balanced sampling strategy)")
    print("=" * 80)
    print(f"\n[Sampling parameters] (affect user selection and data sampling; same params reuse dataset)")
    if use_specified_users:
        print(f"  User selection: using {num_users} specified user IDs")
        if user_ids_file:
            print(f"  User ID file: {user_ids_file}")
    else:
        print(f"  Users: {num_users} (random selection)")
    print(f"  Test set length: {test_length} actions")
    print(f"  Sampling strategy: time-balanced + domain-balanced + high/low-value balanced")
    print(f"  Seed: {seed}")
    print(f"  Config signature: {config_signature}")

    if history_time_start or history_time_end:
        print(f"  History time range: {history_time_start or 'any'} ~ {history_time_end or 'any'}")

    if test_time_start or test_time_end:
        print(f"  Test time range: {test_time_start or 'any'} ~ {test_time_end or 'any'}")

    print(f"\n[Evaluation parameters] (do not affect sampling; safe to change between runs)")
    print(f"  Max history tokens: {max_history_tokens}")
    if max_history_days is not None and max_history_days > 0:
        print(f"  Max history days: {max_history_days}")

    # check for existing dataset with same config signature (cache reuse)
    if not force_resample:
        existing_dataset = find_existing_dataset(output_dir, config_signature)
        if existing_dataset:
            print(f"\n" + "=" * 80)
            print("Found existing dataset with matching config signature — reusing.")
            print("=" * 80)
            print(f"\nExisting dataset path: {existing_dataset}")
            print(f"Config signature match: {config_signature}")
            print(f"\nUse --force to force resampling.")
            print("=" * 80)

            print(f"\nACTUAL_OUTPUT_PATH={existing_dataset}")

            with open(existing_dataset, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            # update evaluation params in cached metadata (do not affect sampling)
            old_max_history_tokens = cached_data.get("metadata", {}).get("max_history_tokens")
            old_max_history_days = cached_data.get("metadata", {}).get("max_history_days")

            if "metadata" in cached_data:
                cached_data["metadata"]["max_history_tokens"] = max_history_tokens
                cached_data["metadata"]["max_history_days"] = max_history_days

            params_changed = []
            if old_max_history_tokens != max_history_tokens:
                params_changed.append(f"max_history_tokens: {old_max_history_tokens} -> {max_history_tokens}")
            if old_max_history_days != max_history_days:
                params_changed.append(f"max_history_days: {old_max_history_days} -> {max_history_days}")

            if params_changed:
                print(f"\nEvaluation parameters updated (sampling data unchanged):")
                for change in params_changed:
                    print(f"   - {change}")

            return cached_data
    else:
        print(f"  Force resample: yes")

    print(f"\nData split:")
    if history_time_start or history_time_end or test_time_start or test_time_end:
        print(f"  - History: collected from history time range; token limit is applied during evaluation")
        print(f"  - Test set: balanced sample of {test_length} from test time range (skipping actions with insufficient context)")
    else:
        print(f"  Working backwards from latest user actions:")
        print(f"  - Test set: balanced sample of {test_length} (skipping actions with insufficient context)")
        print(f"  - History context: actions before test set; token limit is applied during evaluation")
    
    # 1. load data
    if use_specified_users:
        print(f"\nLoading data for {len(target_user_ids)} specified users...")
        all_users = load_user_data(
            input_data_path,
            min_actions=0,
            target_users=0  # load all, filter below
        )

        specified_users = {}
        missing_users = []
        for uid in target_user_ids:
            if uid in all_users:
                specified_users[uid] = all_users[uid]
            else:
                missing_users.append(uid)

        if missing_users:
            print(f"  Warning: {len(missing_users)} user ID(s) not found in data")
            if len(missing_users) <= 10:
                print(f"    Not found: {missing_users}")
            else:
                print(f"    Not found (first 10): {missing_users[:10]}...")

        all_users = specified_users
        print(f"  Loaded {len(all_users)} specified users")

        selected_user_ids = sorted([uid for uid in target_user_ids if uid in all_users])
        print(f"\nUsing {len(selected_user_ids)} specified users")
    else:
        # normal mode: load data and randomly select users
        min_required = test_length * 2
        all_users = load_user_data(
            input_data_path,
            min_actions=min_required,
            target_users=num_users * 3  # load extra for filtering
        )

        # 2. select users
        selected_user_ids = select_users(
            all_users,
            num_users=num_users,
            test_length=test_length,
            history_time_start=history_time_start,
            history_time_end=history_time_end,
            test_time_start=test_time_start,
            test_time_end=test_time_end,
            min_history_actions=1,
            seed=seed
        )

    use_separated_time_range = (history_time_start or history_time_end or test_time_start or test_time_end)

    # 3. prepare experiment data
    print(f"\nPreparing experiment data...")
    experiment_data = {
        "metadata": {
            "num_users": len(selected_user_ids),
            "test_length": test_length,
            "max_history_tokens": max_history_tokens,
            "max_history_days": max_history_days,
            "history_time_start": history_time_start,
            "history_time_end": history_time_end,
            "test_time_start": test_time_start,
            "test_time_end": test_time_end,
            "sampling_strategy": "balanced",
            "use_separated_time_range": use_separated_time_range,
            "use_specified_users": use_specified_users,
            "user_ids_file": user_ids_file if use_specified_users else None,
            "seed": seed,
            "created_at": __import__('datetime').datetime.now().isoformat(),
        },
        "users": []
    }
    
    action_type_stats = defaultdict(int)

    user_history_tokens = []
    all_history_timestamps = []
    all_test_timestamps = []

    user_sampled_actions_dir = None
    seed_offsets = _load_seed_offsets()
    if seed_offsets:
        print(f"  Loaded seed offsets for {len(seed_offsets)} users")
    
    for user_idx, user_id in enumerate(tqdm(
        selected_user_ids,
        desc="  Preparing users",
        unit="user",
        dynamic_ncols=True,
    )):
        user_data = all_users[user_id]
        user_seed = seed + get_stable_user_seed_offset(user_id, seed_offsets)
        
        result_data, test_actions_with_index, base_history_tokens, skipped_actions_info = split_user_actions_by_tokens(
            user_data, 
            test_length,
            history_time_start=history_time_start,
            history_time_end=history_time_end,
            test_time_start=test_time_start,
            test_time_end=test_time_end,
            seed=user_seed
        )

        for action in result_data["base_history"]:
            timestamp = action.get("timestamp", "unknown time")
            if timestamp and timestamp != "unknown time":
                all_history_timestamps.append(timestamp)
        
        user_history_tokens.append(base_history_tokens)
        
        for test_item in test_actions_with_index:
            action = test_item["action"]
            timestamp = action.get("timestamp", "")
            if timestamp:
                all_test_timestamps.append(timestamp)
        
        test_actions_data = []
        for test_item in test_actions_with_index:
            action = test_item["action"]
            test_time_index = test_item["test_time_index"]
            test_actions_data.append({
                "action": action,
                "test_time_index": test_time_index,
            })

            action_type = get_action_type(action, "unknown")
            action_type_stats[action_type] += 1
        
        experiment_data["users"].append({
            "user_id": user_id,
            "user_profile": result_data["user_profile"],
            "base_history": result_data["base_history"],
            "test_time_all_actions": result_data["test_time_all_actions"],
            "test_actions": test_actions_data,
            "skipped_actions": skipped_actions_info,
            "stats": {
                "base_history_count": len(result_data["base_history"]),
                "test_time_all_actions_count": len(result_data["test_time_all_actions"]),
                "test_count": len(test_actions_with_index),
                "skipped_count": len(skipped_actions_info),
                "base_history_tokens": base_history_tokens
            }
        })

    # 4. add statistics to metadata
    experiment_data["metadata"]["action_type_statistics"] = dict(action_type_stats)
    experiment_data["metadata"]["covered_action_types"] = list(action_type_stats.keys())
    experiment_data["metadata"]["total_test_actions"] = sum(action_type_stats.values())

    if user_history_tokens:
        import statistics
        experiment_data["metadata"]["history_token_statistics"] = {
            "avg_tokens_per_user": sum(user_history_tokens) / len(user_history_tokens),
            "min_tokens": min(user_history_tokens),
            "max_tokens": max(user_history_tokens),
            "median_tokens": statistics.median(user_history_tokens),
            "total_tokens": sum(user_history_tokens),
        }
    
    time_distribution = {}
    if all_history_timestamps:
        sorted_history = sorted(all_history_timestamps)
        time_distribution["history"] = {
            "earliest": sorted_history[0],
            "latest": sorted_history[-1],
            "count": len(sorted_history)
        }
    if all_test_timestamps:
        sorted_test = sorted(all_test_timestamps)
        time_distribution["test"] = {
            "earliest": sorted_test[0],
            "latest": sorted_test[-1],
            "count": len(sorted_test)
        }
    if time_distribution:
        experiment_data["metadata"]["time_distribution"] = time_distribution
    
    # 5. save (organized by date and descriptive directory name)
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    now_shanghai = datetime.now(shanghai_tz)
    run_date = now_shanghai.strftime("%Y-%m-%d")

    experiment_data["metadata"]["config_signature"] = config_signature
    experiment_data["metadata"]["descriptive_dirname"] = descriptive_dirname

    # output path: dataset/YYYY-MM-DD/<descriptive-name>/
    output_base_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    timestamp_output_dir = os.path.join(output_base_dir, run_date, descriptive_dirname)
    timestamped_output_path = os.path.join(timestamp_output_dir, filename)

    os.makedirs(timestamp_output_dir, exist_ok=True)
    with open(timestamped_output_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, ensure_ascii=False, indent=2)

    print(f"\nExperiment data saved to: {timestamped_output_path}")

    # 5.1 save per-user sampled action files
    user_sampled_actions_dir = os.path.join(timestamp_output_dir, "sampled_actions_distribution")
    os.makedirs(user_sampled_actions_dir, exist_ok=True)

    print(f"\nSaving per-user sampled action files...")
    for user_entry in experiment_data["users"]:
        user_id = user_entry["user_id"]
        user_sampled_file = os.path.join(user_sampled_actions_dir, f"{user_id}_sampled_actions.json")

        user_sampled_data = {
            "user_id": user_id,
            "test_length": test_length,
            "actual_sampled_count": len(user_entry["test_actions"]),
            "skipped_count": user_entry["stats"]["skipped_count"],
            "sampled_actions": [],
            "skipped_actions": user_entry.get("skipped_actions", [])
        }

        for i, test_item in enumerate(user_entry["test_actions"], 1):
            action = test_item["action"]
            user_sampled_data["sampled_actions"].append({
                "index": i,
                "timestamp": action.get("timestamp", ""),
                "scene_type": get_action_type(action),
                "summary": _get_action_summary(action),
                "full_action": action
            })

        with open(user_sampled_file, 'w', encoding='utf-8') as f:
            json.dump(user_sampled_data, f, ensure_ascii=False, indent=2)

    print(f"  Per-user sampled action files saved to: {user_sampled_actions_dir}")
    print(f"  Total user files: {len(experiment_data['users'])}")

    # 5.2 generate sampled action analysis report
    analysis_output_path = os.path.join(timestamp_output_dir, "sampled_actions_analysis.json")
    analyze_sampled_actions(experiment_data, analysis_output_path)

    # 5.3 generate sampled action visualisation charts
    try:
        from plot.plot_sampled_actions import generate_sampled_actions_charts
        generate_sampled_actions_charts(experiment_data, user_sampled_actions_dir)
    except ImportError:
        print("  ⚠️  plot_sampled_actions.py not found, skipping sampled actions chart generation")
    except Exception as e:
        print(f"  ⚠️  Sampled actions chart generation failed: {e}")

    # 5.4 generate history distribution visualisation charts
    history_distribution_dir = os.path.join(timestamp_output_dir, "history_distribution")
    try:
        from plot.plot_history_distribution import generate_history_charts
        generate_history_charts(experiment_data, history_distribution_dir)
    except ImportError:
        print("  ⚠️  plot_history_distribution.py not found, skipping history distribution chart generation")
    except Exception as e:
        print(f"  ⚠️  History distribution chart generation failed: {e}")

    # 6. print summary report
    print("\n" + "=" * 80)
    print("Experiment data statistics")
    print("=" * 80)

    total_test_actions = sum(action_type_stats.values())

    print(f"\nBasic info:")
    print(f"  Users: {len(selected_user_ids)}")
    print(f"  Total test actions: {total_test_actions}")
    print(f"  Scene types: {len(action_type_stats)}")

    if len(selected_user_ids) > 0:
        print(f"  Avg per user: {total_test_actions / len(selected_user_ids):.1f} test actions")

    if user_history_tokens:
        import statistics
        avg_tokens = sum(user_history_tokens) / len(user_history_tokens)
        median_tokens = statistics.median(user_history_tokens)
        print(f"\nBase history token statistics:")
        print(f"  Avg tokens per user: {avg_tokens:,.0f}")
        print(f"  Median tokens: {median_tokens:,.0f}")
        print(f"  Min tokens: {min(user_history_tokens):,}")
        print(f"  Max tokens: {max(user_history_tokens):,}")
        print(f"  Total tokens: {sum(user_history_tokens):,}")

    if all_history_timestamps or all_test_timestamps:
        print(f"\nTime distribution:")
        if all_history_timestamps:
            sorted_history = sorted(all_history_timestamps)
            print(f"  Base history time range:")
            print(f"    Earliest: {sorted_history[0]}")
            print(f"    Latest: {sorted_history[-1]}")
            print(f"    Count: {len(sorted_history)}")
        if all_test_timestamps:
            sorted_test = sorted(all_test_timestamps)
            print(f"  Test set time range:")
            print(f"    Earliest: {sorted_test[0]}")
            print(f"    Latest: {sorted_test[-1]}")
            print(f"    Count: {len(sorted_test)}")

    print(f"\nScene type distribution:")
    for action_type, count in sorted(action_type_stats.items(), key=lambda x: -x[1]):
        percentage = count / sum(action_type_stats.values()) * 100
        print(f"  {action_type:20s}: {count:5d} samples ({percentage:5.1f}%)")

    print(f"\nUser sample (first 3):")
    for i, user in enumerate(experiment_data["users"][:3], 1):
        print(f"  {i}. User {user['user_id']}")
        print(f"     Base history: {user['stats']['base_history_count']} actions ({user['stats']['base_history_tokens']:,} tokens)")
        print(f"     Test-time actions: {user['stats']['test_time_all_actions_count']}")
        print(f"     Test actions: {user['stats']['test_count']}")

    print("\n" + "=" * 80)
    print("Experiment data preparation complete.")
    print("=" * 80)
    print(f"\nNext step:")
    print(f"  python src/evaluation/evaluator.py --use-fixed-data {timestamped_output_path}")
    print("=" * 80)

    print(f"\nACTUAL_OUTPUT_PATH={timestamped_output_path}")
    
    return experiment_data


def main():
    """CLI entry point for preparing experiment datasets."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare a fixed experiment dataset using balanced sampling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic: select 10 users and 20 test actions; history token limit is applied during evaluation
  python src/data/prepare_experiment_data.py --num-users 10 --test-length 20

  # Specify max history tokens (128K)
  python src/data/prepare_experiment_data.py --num-users 10 --max-history-tokens 128000

  # Separate time ranges: history from Sep, test from Oct-Nov
  python src/data/prepare_experiment_data.py \\
    --num-users 10 \\
    --test-length 30 \\
    --history-time-start 2025-09-01 \\
    --history-time-end 2025-09-30 \\
    --test-time-start 2025-10-01 \\
    --test-time-end 2025-11-30 \\
    --output dataset/experiment_data.json

  # Use a specified user ID file (skip random selection)
  python src/data/prepare_experiment_data.py \\
    --user-ids-file user_stats_filtered_output.json \\
    --test-length 30 \\
    --output dataset/specified_users.json

  # Extract a user subset from an existing dataset (test_actions stay identical)
  python src/data/prepare_experiment_data.py \\
    --source-dataset dataset/2026-02-03/199u_30t_h~0930_t1001~1130_s42_882d82f7/experiment_data.json \\
    --extract-num-users 50 \\
    --output dataset/experiment_data.json

  # Extract specific user IDs from an existing dataset
  python src/data/prepare_experiment_data.py \\
    --source-dataset dataset/2026-02-03/199u_30t_h~0930_t1001~1130_s42_882d82f7/experiment_data.json \\
    --user-ids-file my_selected_users.txt \\
    --output dataset/experiment_data.json

User ID file formats supported:
  1. JSON with users array:
     {"users": [{"user_id": "123", ...}, {"user_id": "456", ...}]}
  2. JSON array:
     ["123", "456", "789"]
  3. Plain text, one ID per line:
     123
     456
     789

Sampling strategy:
  Balanced sampling (time + domain + high/low-value):
  1. Time balance: split test time range into M buckets
  2. Domain balance: round-robin across domains (e.g. video, live-stream)
  3. Value balance: 50% probability of high-value vs low-value item per domain
  4. Only skip actions with insufficient context; fill up to test_length

Data split logic:
  Separated time range mode:
  - History: collected from history-time-start ~ history-time-end
  - Test set: balanced sample from test-time-start ~ test-time-end

  Default mode (no time range specified):
  - Work backwards from user's latest actions
  - Balanced sample of test_length actions as test set
  - History context: actions before test set; token limit is applied during evaluation

Subset extraction mode (--source-dataset):
  - Extract a user subset from an existing dataset
  - test_actions for extracted users match the source experiment exactly
  - Suitable for fast testing or comparative analysis
        """
    )
    
    # subset extraction args
    parser.add_argument("--source-dataset", type=str, default=None,
                       help="Source dataset path for subset extraction mode")
    parser.add_argument("--extract-num-users", type=int, default=None,
                       help="Number of users to randomly extract from source dataset (used with --source-dataset)")

    # regular args
    parser.add_argument("--num-users", type=int, default=50,
                       help="Number of users to select (default 50; ignored when --user-ids-file is set)")
    parser.add_argument("--user-ids-file", type=str, default=None,
                       help="Path to a user ID file (JSON or plain text); skips random user selection")
    parser.add_argument("--test-length", type=int, default=20,
                       help="Number of test actions per user (default 20)")
    parser.add_argument("--max-history-tokens", type=int, default=DEFAULT_MAX_HISTORY_TOKENS,
                       help=f"Max token budget for history actions (default {DEFAULT_MAX_HISTORY_TOKENS})")
    parser.add_argument("--max-history-days", type=int, default=None,
                       help="Keep only the most recent N days of history (default: no limit)")
    parser.add_argument("--history-time-start", type=str, default=None,
                       help="History time range start (format: YYYY-MM-DD)")
    parser.add_argument("--history-time-end", type=str, default=None,
                       help="History time range end (format: YYYY-MM-DD)")
    parser.add_argument("--test-time-start", type=str, default=None,
                       help="Test set time range start (format: YYYY-MM-DD)")
    parser.add_argument("--test-time-end", type=str, default=None,
                       help="Test set time range end (format: YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default 42)")
    parser.add_argument("--output", type=str, default="dataset/experiment_data.json",
                       help="Output file path (default dataset/experiment_data.json)")
    parser.add_argument("--force", action="store_true",
                       help="Force resampling even if a dataset with the same config exists")

    args = parser.parse_args()

    if args.source_dataset:
        # subset extraction mode
        target_user_ids = None
        if args.user_ids_file:
            if not os.path.exists(args.user_ids_file):
                raise FileNotFoundError(f"User ID file not found: {args.user_ids_file}")
            target_user_ids = load_user_ids_from_file(args.user_ids_file)
            print(f"Loaded {len(target_user_ids)} user IDs from file: {args.user_ids_file}")

        extract_users_from_existing_dataset(
            source_dataset_path=args.source_dataset,
            target_user_ids=target_user_ids,
            num_users=args.extract_num_users,
            output_path=args.output,
            seed=args.seed,
        )
    else:
        # normal mode: generate new experiment dataset
        prepare_fixed_experiment_data(
            INPUT_DATA_PATH,
            args.output,
            num_users=args.num_users,
            test_length=args.test_length,
            max_history_tokens=args.max_history_tokens,
            max_history_days=args.max_history_days,
            history_time_start=args.history_time_start,
            history_time_end=args.history_time_end,
            test_time_start=args.test_time_start,
            test_time_end=args.test_time_end,
            seed=args.seed,
            force_resample=args.force,
            user_ids_file=args.user_ids_file,
        )


if __name__ == "__main__":
    main()
