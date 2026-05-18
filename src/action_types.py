from typing import Any, Dict, Iterable, Set


ACTION_TYPE_ALIASES = {
    "Video Browsing": "视频浏览",
    "Live Streaming": "直播间",
    "E-commerce": "商城购物",
    "Advertisement": "广告推荐",
    "Customer Service": "电商客服对话",
    "Search Behavior": "搜索行为",
}

_ACTION_TYPE_ALIASES_CASEFOLD = {
    key.casefold(): value for key, value in ACTION_TYPE_ALIASES.items()
}


def normalize_action_type(action_type: Any, default: str = "") -> str:
    """Map translated scene names back to the canonical Chinese names used internally."""
    if not isinstance(action_type, str):
        return default
    stripped = action_type.strip()
    if not stripped:
        return default
    return ACTION_TYPE_ALIASES.get(
        stripped,
        _ACTION_TYPE_ALIASES_CASEFOLD.get(stripped.casefold(), stripped),
    )


def get_action_type(action: Dict, default: str = "") -> str:
    return normalize_action_type(action.get("type", default), default)


def get_record_action_type(record: Dict, default: str = "") -> str:
    return normalize_action_type(record.get("action_type", default), default)


def normalize_action_type_set(action_types: Iterable[Any]) -> Set[str]:
    return {normalize_action_type(action_type) for action_type in action_types}
