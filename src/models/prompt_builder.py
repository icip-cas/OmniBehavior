import json
import functools
import sys
from pathlib import Path
from typing import Dict, List, Optional
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from action_types import get_action_type

DEFAULT_MAX_HISTORY_TOKENS = 0

QWEN_TOKENIZER = None
TOKENIZER_LOAD_ATTEMPTED = False


def _get_qwen_tokenizer():
    """Load Qwen tokenizer lazily so importing this module stays fast."""
    global QWEN_TOKENIZER, TOKENIZER_LOAD_ATTEMPTED
    if TOKENIZER_LOAD_ATTEMPTED:
        return QWEN_TOKENIZER

    TOKENIZER_LOAD_ATTEMPTED = True
    try:
        from transformers import AutoTokenizer
        QWEN_TOKENIZER = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_MODEL)
    except Exception as e:
        QWEN_TOKENIZER = None
        print(f"Warning: Qwen tokenizer not available ({e}), using approximate token counting")

    return QWEN_TOKENIZER


def _safe_numeric(value):
    """Try to coerce value to int/float; return original if unparseable."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            if '.' in s:
                return float(s)
            return int(s)
        except (ValueError, TypeError):
            pass
        import re
        match = re.match(r'^(\d+\.?\d*)', s)
        if match:
            num_str = match.group(1)
            try:
                if '.' in num_str:
                    return float(num_str)
                return int(num_str)
            except (ValueError, TypeError):
                pass
        return value
    return value


# context fields to coerce to numeric types
_NUMERIC_CONTEXT_FIELDS = {
    "duration", "fans_user_num",
    "show_cnt", "play_cnt", "complete_play_cnt",
    "like_cnt", "comment_cnt", "share_cnt", "collect_cnt", "download_cnt", "follow_cnt",
    "report_cnt", "reduce_similar_cnt",
    "item_price", "item_num_180d_sales", "item_num_good_assess",
    "item_num_180d_buyer", "item_num_180d_rebuyer", "item_num_180d_cart",
    "live_total_user_cnt", "live_total_view_cnt", "live_total_view_duration",
    "live_like_cnt", "live_comment_cnt",
}


def format_action_context(action: Dict, is_target: bool = False) -> str:
    """Format action context fields into a readable description."""
    raw_context = action.get("context", {})
    # safely coerce numeric fields to int/float to avoid comparison errors on string values
    context = {
        k: (_safe_numeric(v) if k in _NUMERIC_CONTEXT_FIELDS else v)
        for k, v in raw_context.items()
    }
    action_type = get_action_type(action)
    
    if action_type == "视频浏览":
        parts = []

        if "caption" in context and context["caption"]:
            parts.append(f"这是一个标题为《{context['caption']}》的视频")
        else:
            parts.append("这是一个视频")
        
        if "fans_user_num" in context and context["fans_user_num"] > 0:
            fans_num = context["fans_user_num"]
            is_verified = context.get("is_verified", False)
            verification_type = context.get("verification_type", "")
            
            if fans_num >= 10000:
                fans_str = f"{fans_num / 10000:.1f}万"
            else:
                fans_str = f"{fans_num}"
            
            author_info = f"作者拥有 {fans_str} 粉丝"
            if is_verified and verification_type:
                author_info += f"，并且通过了{verification_type}认证"
            elif is_verified:
                author_info += "，并且已通过平台认证"
            parts.append(author_info)
        elif context.get("is_verified", False):
            verification_type = context.get("verification_type", "")
            if verification_type:
                parts.append(f"作者已通过{verification_type}认证")
            else:
                parts.append("作者已通过平台认证")
        
        if "duration" in context and context["duration"]:
            dur = context['duration']
            if isinstance(dur, (int, float)):
                duration_val = int(dur)
                duration_minutes = duration_val // 60
                duration_seconds = duration_val % 60
                if duration_minutes > 0:
                    parts.append(f"视频时长为 {duration_minutes} 分 {duration_seconds} 秒")
                else:
                    parts.append(f"视频时长为 {duration_seconds} 秒")
            else:
                # string with unit, e.g. "60秒" — keep as-is
                parts.append(f"视频时长为 {dur}")
        
        popularity = []
        if "show_cnt" in context and context["show_cnt"] > 0:
            popularity.append(f"{context['show_cnt']}次曝光")
        if "play_cnt" in context and context["play_cnt"] > 0:
            popularity.append(f"{context['play_cnt']}次播放")
        if "complete_play_cnt" in context and context["complete_play_cnt"] > 0:
            popularity.append(f"{context['complete_play_cnt']}次完整播放")
        if "like_cnt" in context and context["like_cnt"] > 0:
            popularity.append(f"{context['like_cnt']}个赞")
        if "comment_cnt" in context and context["comment_cnt"] > 0:
            popularity.append(f"{context['comment_cnt']}条评论")
        if "share_cnt" in context and context["share_cnt"] > 0:
            popularity.append(f"{context['share_cnt']}次分享")
        if "collect_cnt" in context and context["collect_cnt"] > 0:
            popularity.append(f"{context['collect_cnt']}次收藏")
        if "download_cnt" in context and context["download_cnt"] > 0:
            popularity.append(f"{context['download_cnt']}次下载")
        if "follow_cnt" in context and context["follow_cnt"] > 0:
            popularity.append(f"{context['follow_cnt']}次关注")
        
        if popularity:
            parts.append("该视频目前收获了 " + "、".join(popularity))
        
        if "ocr" in context and context["ocr"]:
            ocr_text = context['ocr']
            if len(ocr_text) > 10000:
                parts.append(f"视频画面中识别到的文字内容为：{ocr_text[:10000]}...")
            else:
                parts.append(f"视频画面中识别到的文字内容为：{ocr_text}")
        
        if "asr" in context and context["asr"]:
            asr_text = context['asr']
            if len(asr_text) > 10000:
                parts.append(f"视频中的语音内容为：{asr_text[:10000]}...")
            else:
                parts.append(f"视频中的语音内容为：{asr_text}")

        if "upload_timestamp" in context and context["upload_timestamp"]:
            parts.append(f"发布时间是 {context['upload_timestamp']}")
        
        return "。".join(parts) + "。" if parts else "视频信息缺失。"
    
    elif action_type == "商城购物":
        parts = []
        
        # item name
        if "item_title" in context and context["item_title"]:
            parts.append(f"这是一件商品，名称为{context['item_title']}")
        else:
            parts.append("这是一件商品")
        
        # item description
        if "item_desc" in context and context["item_desc"]:
            desc = context["item_desc"]
            if len(desc) > 10000:
                parts.append(f"商品描述：{desc[:10000]}...")
            else:
                parts.append(f"商品描述：{desc}")
        
        # product source
        if "product_source" in context and context["product_source"]:
            parts.append(f"商品来自{context['product_source']}")
        
        # price
        if "item_price" in context and context["item_price"]:
            price = context['item_price']
            if price > 0:
                parts.append(f"售价为 {price} 元")
            else:
                parts.append("售价未知")
        
        # category hierarchy (up to 3 levels)
        category_info = []
        if "category_level1_name" in context and context["category_level1_name"]:
            category_info.append(context["category_level1_name"])
        if "category_level2_name" in context and context["category_level2_name"]:
            category_info.append(context["category_level2_name"])
        if "category_level3_name" in context and context["category_level3_name"]:
            category_info.append(context["category_level3_name"])
        
        if category_info:
            parts.append(f"商品具体类别为：{'，'.join(category_info)}")
        
        industry_info = []
        if "industry_level1_name" in context and context["industry_level1_name"]:
            industry_info.append(context["industry_level1_name"])
        if "industry_level2_name" in context and context["industry_level2_name"]:
            industry_info.append(context["industry_level2_name"])
        if "industry_level3_name" in context and context["industry_level3_name"]:
            industry_info.append(context["industry_level3_name"])
        
        if industry_info:
            parts.append(f"所属行业为：{'，'.join(industry_info)}")
        
        sales_info = []
        if "item_num_180d_sales" in context and context["item_num_180d_sales"] > 0:
            sales = context["item_num_180d_sales"]
            if sales >= 10000:
                sales_info.append(f"近180天已售出 {sales/10000:.1f}万件")
            else:
                sales_info.append(f"近180天已售出 {sales} 件")
        
        if "item_num_good_assess" in context and context["item_num_good_assess"] > 0:
            assess = context["item_num_good_assess"]
            if assess >= 10000:
                sales_info.append(f"{assess/10000:.1f}万条好评")
            else:
                sales_info.append(f"{assess}条好评")
        
        if "item_num_180d_buyer" in context and context["item_num_180d_buyer"] > 0:
            buyer = context["item_num_180d_buyer"]
            if buyer >= 10000:
                sales_info.append(f"{buyer/10000:.1f}万人购买")
            else:
                sales_info.append(f"{buyer}人购买")
        
        if "item_num_180d_rebuyer" in context and context["item_num_180d_rebuyer"] > 0:
            rebuyer = context["item_num_180d_rebuyer"]
            if rebuyer >= 10000:
                sales_info.append(f"{rebuyer/10000:.1f}万人复购")
            else:
                sales_info.append(f"{rebuyer}人复购")
        
        if sales_info:
            parts.append("该商品" + "、".join(sales_info))
        
        if "item_num_180d_cart" in context and context["item_num_180d_cart"] > 0:
            cart_num = context["item_num_180d_cart"]
            if cart_num >= 10000:
                parts.append(f"近180天有 {cart_num/10000:.1f}万 人加入购物车")
            else:
                parts.append(f"近180天有 {cart_num} 人加入购物车")
        
        return "。".join(parts) + "。" if parts else "商品信息缺失。"
    
    elif action_type == "广告推荐":
        parts = []
        
        if "product" in context and context["product"]:
            product_info = f"这是一条推广{context['product']}的广告"
            if "industry_main" in context and context["industry_main"]:
                product_info += f"，属于{context['industry_main']}"
                if "industry_sub" in context and context["industry_sub"] and context["industry_sub"] != context.get("industry_main"):
                    product_info += f"行业中的{context['industry_sub']}品类"
                else:
                    product_info += "行业"
            parts.append(product_info)
        else:
            parts.append("这是一条广告")
        
        if "is_verified" in context and context["is_verified"]:
            if "verification_type" in context and context["verification_type"]:
                parts.append(f"广告主已认证，认证类型为{context['verification_type']}")
            else:
                parts.append("广告主已认证")
        
        if "fans_user_num" in context and context["fans_user_num"]:
            parts.append(f"广告主粉丝数为 {context['fans_user_num']}")
        elif "fans_user_num_op_range" in context and context["fans_user_num_op_range"]:
            parts.append(f"广告主粉丝数范围为{context['fans_user_num_op_range']}")
        
        if "caption" in context and context["caption"]:
            caption = context['caption']
            if len(caption) > 500:
                parts.append(f"广告标题为：{caption[:500]}...")
            else:
                parts.append(f"广告标题为：{caption}")
        
        if "photo_type" in context and context["photo_type"]:
            parts.append(f"广告类型为{context['photo_type']}")
        
        if "duration" in context and context["duration"]:
            dur = context["duration"]
            if isinstance(dur, (int, float)):
                duration = int(dur)
                if duration >= 60:
                    minutes = duration // 60
                    seconds = duration % 60
                    if seconds > 0:
                        parts.append(f"广告时长为 {minutes} 分 {seconds} 秒")
                    else:
                        parts.append(f"广告时长为 {minutes} 分钟")
                else:
                    parts.append(f"广告时长为 {duration} 秒")
            else:
                parts.append(f"广告时长为 {dur}")

        if "upload_timestamp" in context and context["upload_timestamp"]:
            parts.append(f"广告上传时间为 {context['upload_timestamp']}")

        if "show_cnt" in context and context["show_cnt"] and context["show_cnt"] > 0:
            parts.append(f"该广告已经向用户曝光过 {context['show_cnt']} 次")
        
        if "play_cnt" in context and context["play_cnt"] and context["play_cnt"] > 0:
            parts.append(f"广告播放次数为 {context['play_cnt']}")

        if "complete_play_cnt" in context and context["complete_play_cnt"] and context["complete_play_cnt"] > 0:
            parts.append(f"广告被完整播放 {context['complete_play_cnt']} 次")

        interaction_parts = []
        if "like_cnt" in context and context["like_cnt"] and context["like_cnt"] > 0:
            interaction_parts.append(f"获得点赞 {context['like_cnt']} 次")
        if "comment_cnt" in context and context["comment_cnt"] and context["comment_cnt"] > 0:
            interaction_parts.append(f"获得评论 {context['comment_cnt']} 次")
        if "share_cnt" in context and context["share_cnt"] and context["share_cnt"] > 0:
            interaction_parts.append(f"被分享 {context['share_cnt']} 次")
        if "collect_cnt" in context and context["collect_cnt"] and context["collect_cnt"] > 0:
            interaction_parts.append(f"被收藏 {context['collect_cnt']} 次")
        if "download_cnt" in context and context["download_cnt"] and context["download_cnt"] > 0:
            interaction_parts.append(f"被下载 {context['download_cnt']} 次")
        if "follow_cnt" in context and context["follow_cnt"] and context["follow_cnt"] > 0:
            interaction_parts.append(f"被关注 {context['follow_cnt']} 次")
        if interaction_parts:
            parts.append(f"广告互动数据：{', '.join(interaction_parts)}")
        
        negative_parts = []
        if "report_cnt" in context and context["report_cnt"] and context["report_cnt"] > 0:
            negative_parts.append(f"被举报 {context['report_cnt']} 次")
        if "reduce_similar_cnt" in context and context["reduce_similar_cnt"] and context["reduce_similar_cnt"] > 0:
            negative_parts.append(f"减少类似推荐 {context['reduce_similar_cnt']} 次")
        if negative_parts:
            parts.append(f"广告负面反馈：{', '.join(negative_parts)}")
        
        if "ocr_text" in context and context["ocr_text"]:
            ocr_text = context['ocr_text']
            if len(ocr_text) > 10000:
                parts.append(f"广告画面中的文字内容为：{ocr_text[:10000]}")
            else:
                parts.append(f"广告画面中的文字内容为：{ocr_text}")

        if "asr_text" in context and context["asr_text"]:
            asr_text = context['asr_text']
            if len(asr_text) > 10000:
                parts.append(f"广告中的语音内容为：{asr_text[:10000]}")
            else:
                parts.append(f"广告中的语音内容为：{asr_text}")
        
        return "。".join(parts) + "。" if parts else "广告信息缺失。"
    
    elif action_type == "直播间":
        parts = []
        
        if "live_title" in context and context["live_title"]:
            live_intro = f"这是一个标题为{context['live_title']}的直播间"
        else:
            live_intro = "这是一个直播间"
        
        if "live_category" in context and context["live_category"]:
            live_intro += f"，直播类型是{context['live_category']} 场景"
        
        if "live_game_name" in context and context["live_game_name"]:
            live_intro += f"，正在直播《{context['live_game_name']}》"
        
        if "is_shop_live" in context:
            if context['is_shop_live']:
                live_intro += "，主播正在进行带货直播"
            else:
                live_intro += "，这是一场娱乐性的非带货直播"
        
        parts.append(live_intro)
        
        popularity_info = []
        if "live_total_user_cnt" in context and context["live_total_user_cnt"]:
            user_cnt = context['live_total_user_cnt']
            if user_cnt >= 10000:
                popularity_info.append(f"累计 {user_cnt/10000:.1f}万 人次观看")
            else:
                popularity_info.append(f"累计 {user_cnt} 人次观看")
        
        if "live_total_view_cnt" in context and context["live_total_view_cnt"]:
            view_cnt = context['live_total_view_cnt']
            if view_cnt >= 10000:
                popularity_info.append(f"总观看量达到 {view_cnt/10000:.1f}万")
            else:
                popularity_info.append(f"总观看量 {view_cnt}")
        
        if popularity_info:
            parts.append("该直播已有 " + "、".join(popularity_info))
        
        if "live_total_view_duration" in context and context["live_total_view_duration"]:
            duration_seconds = context['live_total_view_duration']
            duration_hours = duration_seconds / 3600
            if duration_hours >= 10000:
                parts.append(f"累计观看时长达到 {duration_hours/10000:.1f}万 小时")
            elif duration_hours >= 1:
                parts.append(f"累计观看时长 {duration_hours:.1f} 小时")
            else:
                duration_minutes = duration_seconds / 60
                parts.append(f"累计观看时长 {duration_minutes:.0f} 分钟")

        if "live_cover_content" in context and context["live_cover_content"]:
            parts.append(f"直播封面的内容是：{context['live_cover_content']}")
        
        interaction_info = []
        if "live_like_cnt" in context and context["live_like_cnt"] > 0:
            like_cnt = context['live_like_cnt']
            if like_cnt >= 10000:
                interaction_info.append(f"{like_cnt/10000:.1f}万个赞")
            else:
                interaction_info.append(f"{like_cnt}个赞")
        
        if "live_comment_cnt" in context and context["live_comment_cnt"] > 0:
            comment_cnt = context['live_comment_cnt']
            if comment_cnt >= 10000:
                interaction_info.append(f"{comment_cnt/10000:.1f}万条评论")
            else:
                interaction_info.append(f"{comment_cnt}条评论")
        
        if interaction_info:
            parts.append("直播间目前有 " + "、".join(interaction_info))
        
        if "items" in context and context["items"] and len(context["items"]) > 0:
            items = context["items"]
            item_details = []
            for item in items:
                item_title = item.get("title", "")
                item_price = item.get("price", "")
                if item_title and item_price:
                    item_details.append(f"{item_title}售价 {item_price} 元")
                elif item_title:
                    item_details.append(f"{item_title}")
            if item_details:
                if len(item_details) == 1:
                    parts.append(f"主播正在推荐 1 件商品：{item_details[0]}")
                else:
                    parts.append(f"主播正在推荐 {len(item_details)} 件商品：{', '.join(item_details)}")
        
        return "。".join(parts) + "。" if parts else "直播信息缺失。"
    
    elif action_type == "搜索行为":
        parts = []
        
        # keyword may be in action list or in context
        keyword = None
        query_category = None
        action_list = action.get("action", [])
        for act in action_list:
            if act.get("type") == "search":
                keyword = act.get("keyword")
                query_category = act.get("query_category")
                break
        if not keyword and "keyword" in context:
            keyword = context["keyword"]
        if not query_category and "query_category" in context:
            query_category = context["query_category"]
        
        if keyword:
            if is_target:
                search_info = "用户打开了搜索框"
            else:
                search_info = f"用户在搜索框中输入了关键词 {keyword}"
            if query_category:
                if query_category == "查询型":
                    search_info += "，希望查找相关信息"
                elif query_category == "浏览型":
                    search_info += "，想要浏览相关内容"
                else:
                    search_info += f"，搜索意图是{query_category}"
            parts.append(search_info)
        else:
            parts.append("用户进行了搜索")
        
        return "。".join(parts) + "。" if parts else "搜索信息缺失。"
    
    elif action_type == "电商客服对话":
        parts = []
        
        if "ticket_category" in context and context["ticket_category"]:
            parts.append(f"这是一次{context['ticket_category']}类型的咨询")
        else:
            parts.append("这是一次客服对话")
        
        order_details = []
        if "product_name" in context and context["product_name"]:
            order_details.append(f"涉及商品为《{context['product_name']}》")

        if "product_category_info" in context and context["product_category_info"]:
            order_details.append(f"商品类别是{context['product_category_info']}")

        price_info = []
        if "item_price" in context and context["item_price"]:
            price_info.append(f"单价 {context['item_price']} 元")
        
        if "item_qty" in context and context["item_qty"]:
            qty = context['item_qty']
            try:
                if isinstance(qty, (int, float)):
                    if qty == int(qty):  # integer value
                        price_info.append(f"购买数量 {int(qty)} 件")
                    else:
                        price_info.append(f"购买数量 {qty} 件")
                else:
                    price_info.append(f"购买数量 {qty} 件")
            except:
                price_info.append(f"购买数量 {qty} 件")
        
        if "express_fee" in context and context["express_fee"]:
            try:
                fee = float(context["express_fee"])
                if fee > 0:
                    price_info.append(f"运费 {fee} 元")
            except:
                pass
        
        if price_info:
            order_details.append("，".join(price_info))
        
        if order_details:
            parts.append("；".join(order_details))
        
        if "pay_order_time" in context and context["pay_order_time"]:
            parts.append(f"该订单的下单时间是 {context['pay_order_time']}")
        
        return "。".join(parts) + "。" if parts else "客服对话信息缺失。"
    
    return str(context)


def format_action_result(action: Dict) -> str:
    """Format the action result for a single event, grouped by scene type."""
    action_type = get_action_type(action)
    action_list = action.get("action", [])
    results = []
    
    if action_type == "视频浏览":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "watch":
                watch_details = []

                if "play_duration" in act:
                    play_duration = act["play_duration"]
                    if isinstance(play_duration, str):
                        watch_details.append(f"观看了 {play_duration}")
                    elif isinstance(play_duration, (int, float)):
                        watch_sec = play_duration
                        if watch_sec >= 60:
                            minutes = int(watch_sec // 60)
                            seconds = int(watch_sec % 60)
                            if seconds > 0:
                                watch_details.append(f"观看了 {minutes} 分 {seconds} 秒")
                            else:
                                watch_details.append(f"观看了 {minutes} 分钟")
                        else:
                            if watch_sec == int(watch_sec):
                                watch_details.append(f"观看了 {int(watch_sec)} 秒")
                            else:
                                watch_details.append(f"观看了 {watch_sec:.1f} 秒")
                elif "watch_seconds" in act:
                    watch_sec = act["watch_seconds"]
                    if watch_sec >= 60:
                        minutes = int(watch_sec // 60)
                        seconds = int(watch_sec % 60)
                        if seconds > 0:
                            results.append(f"观看了 {minutes} 分 {seconds} 秒")
                        else:
                            results.append(f"观看了 {minutes} 分钟")
                
                watch_features = []
                if act.get("played_loop_cnt"):
                    watch_features.append(f"循环播放了 {act.get('played_loop_cnt')}")
                if act.get("is_fast_forward_play"):
                    watch_features.append("使用了快进播放操作")
                if act.get("is_backward_play"):
                    watch_features.append("进行了回退观看操作")
                if act.get("is_enlarge_play"):
                    watch_features.append("放大了视频画面操作")
                # completion flag: accepts either "completed" or "is_complete_play"
                if act.get("completed") or act.get("is_complete_play"):
                    watch_features.append("完整看完了整个视频")
                
                if watch_features:
                    watch_details.append("，".join(watch_features))
                
                if watch_details:
                    results.append("、".join(watch_details))
            
            elif act_type == "like":
                results.append("对视频进行点赞操作")

            elif act_type == "comment":
                comment_detail = act.get("comment_detail_list", "")
                if comment_detail:
                    results.append(f"发表了评论：{comment_detail}")
                else:
                    results.append("发表了评论")
                if act.get("comment_stay_duration"):
                    results.append(f"在评论区停留了 {act.get('comment_stay_duration')}")
                if act.get("is_at_friend_in_comment"):
                    results.append("在评论区提及了好友")

            elif act_type == "share":
                share_cnt = act.get("share_cnt")
                if share_cnt:
                    results.append(f"分享给朋友，成功分享了{share_cnt}次")
                else:
                    results.append("分享了视频")

            elif act_type == "collect":
                collect_cnt = act.get("collect_cnt")
                if collect_cnt:
                    results.append(f"收藏了该视频，成功收藏了{collect_cnt}次")
                else:
                    results.append("收藏了该视频")

            elif act_type == "download":
                download_cnt = act.get("download_cnt")
                if download_cnt:
                    results.append(f"下载了该视频，成功下载了{download_cnt}次")
                else:
                    results.append("下载了该视频")

            elif act_type == "follow":
                results.append("关注了视频作者")

            elif act_type == "unfollow":
                if act.get("is_unfollow_action"):
                    results.append("取消关注了视频作者")

            elif act_type == "dislike":
                if act.get("reduce_similar_cnt") or act.get("reduce_simliar_cnt"):
                    results.append("选择了不感兴趣")

            elif act_type == "report":
                if act.get("report_cnt"):
                    results.append("进行了举报")

    elif action_type == "商城购物":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "cart":
                if act.get("is_add_to_cart"):
                    results.append("把商品加入了购物车")
                else:
                    results.append("浏览了商品但未加入购物车")

            elif act_type == "purchase":
                if act.get("order_success") or act.get("paid") or act.get("is_pay"):
                    results.append("成功下单购买")
                else:
                    results.append("未购买该商品")
    
    elif action_type == "广告推荐":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "watch":
                watch_seconds = 0
                if "watch_seconds" in act:
                    watch_seconds = act.get("watch_seconds", 0)
                elif "play_duration" in act:
                    play_duration = act.get("play_duration")
                    if isinstance(play_duration, str):
                        try:
                            watch_seconds = float(play_duration.replace("秒", "").strip())
                        except:
                            watch_seconds = 0
                    elif isinstance(play_duration, (int, float)):
                        watch_seconds = float(play_duration)

                watch_features = []
                if watch_seconds > 0:
                    watch_features.append(f"观看了 {watch_seconds:.1f} 秒")
                if act.get("played_loop_cnt"):
                    watch_features.append(f"循环播放了 {act.get('played_loop_cnt')} 次")
                if act.get("is_complete_play"):
                    watch_features.append("看完了整个广告")
                if watch_features:
                    results.append("、".join(watch_features))

            elif act_type == "like":
                if act.get("like_cnt"):
                    results.append(f"为广告点赞 {act.get('like_cnt')} 次")
                else:
                    results.append("为广告点赞")

            elif act_type == "comment":
                results.append("在广告下发表了评论")

            elif act_type == "share":
                share_cnt = act.get("share_cnt")
                if share_cnt:
                    results.append(f"分享了广告 {share_cnt} 次")
                else:
                    results.append("分享了广告")

            elif act_type == "conversion":
                if act.get("conversion_cnt"):
                    results.append(f"点击了广告产生转化，共 {act.get('conversion_cnt')} 次")

            elif act_type == "activation":
                if act.get("activation_cnt"):
                    results.append(f"激活了应用或服务，共 {act.get('activation_cnt')} 次")

            elif act_type == "purchase":
                purchase_info = []
                if act.get("pay_cnt"):
                    purchase_info.append(f"支付了 {act.get('pay_cnt')} 次")
                if act.get("pay_purchase_amt"):
                    purchase_info.append(f"支付金额 {act.get('pay_purchase_amt')} 元")
                if purchase_info:
                    results.append("通过广告" + "、".join(purchase_info))

            elif act_type == "submit":
                if act.get("form_submit_total_cnt"):
                    results.append(f"提交了表单 {act.get('form_submit_total_cnt')} 次")
            
            elif act_type == "follow":
                results.append("关注了广告主")
            
            elif act_type == "unfollow":
                results.append("取消关注了广告主")
            
            elif act_type == "dislike":
                if act.get("report_cnt"):
                    results.append("举报了广告")
    
    elif action_type == "直播间":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "watch":
                play_duration = act.get("play_duration", 0)
                if isinstance(play_duration, str):
                    try:
                        watch_seconds = float(play_duration.replace("秒", "").strip())
                    except:
                        watch_seconds = 0
                elif isinstance(play_duration, (int, float)):
                    watch_seconds = float(play_duration)
                else:
                    watch_seconds = 0

                watch_info = []
                if watch_seconds > 0:
                    if watch_seconds >= 60:
                        minutes = int(watch_seconds // 60)
                        seconds = int(watch_seconds % 60)
                        if seconds > 0:
                            watch_info.append(f"在直播间停留了 {minutes} 分 {seconds} 秒")
                        else:
                            watch_info.append(f"在直播间停留了 {minutes} 分钟")
                    else:
                        watch_info.append(f"在直播间停留了 {int(watch_seconds)} 秒")
                if act.get("play_count"):
                    watch_info.append(f"观看了 {act.get('play_count')} 次")
                if watch_info:
                    results.append("，".join(watch_info))

            elif act_type == "like":
                if act.get("like_cnt"):
                    like_cnt = act.get("like_cnt")
                    if like_cnt >= 10000:
                        results.append(f"为主播点赞 {like_cnt/10000:.1f}万 次")
                    else:
                        results.append(f"为主播点赞 {like_cnt} 次")
                else:
                    results.append("为主播点赞")

            elif act_type == "comment":
                comment_info = []
                if act.get("comment_cnt"):
                    cnt = act.get("comment_cnt")
                    comment_info.append(f"发送了 {cnt} 条弹幕")
                if act.get("comment_content"):
                    comments = act.get("comment_content")
                    if isinstance(comments, list) and len(comments) > 0:
                        display_comments = comments[:3]
                        comment_text = "、".join([f"「{c}」" for c in display_comments])
                        if len(comments) > 3:
                            comment_info.append(f"评论内容包括：{comment_text} 等")
                        else:
                            comment_info.append(f"评论内容：{comment_text}")
                if comment_info:
                    results.append("在直播间" + "，".join(comment_info))
                else:
                    results.append("在直播间发送了弹幕")
            
            elif act_type == "send_gift":
                if act.get("gift_amount"):
                    gift_amount = act.get("gift_amount")
                    results.append(f"给主播送了价值 {gift_amount} 元的礼物")
            
            elif act_type == "follow":
                if act.get("is_follow_action"):
                    results.append("关注了主播")
            
            elif act_type == "unfollow":
                if act.get("is_unfollow_action"):
                    results.append("取消关注了主播")
            
            elif act_type == "share":
                if act.get("share_cnt"):
                    results.append(f"分享了直播间 {act.get('share_cnt')} 次")
            
            elif act_type == "dislike":
                dislike_info = []
                if act.get("report_cnt"):
                    dislike_info.append("举报了直播间")
                if act.get("reduce_simliar_cnt"):
                    dislike_info.append("选择了不感兴趣")
                if dislike_info:
                    results.append("、".join(dislike_info))
            
            elif act_type == "click_cart":
                if act.get("is_click_cart_action"):
                    item_title = act.get("item_title", "")
                    item_price = act.get("item_price", "")
                    if item_title and item_price:
                        results.append(f"把商品 {item_title}（售价{item_price}元）加入了购物车")
                    elif item_title:
                        results.append(f"把商品 {item_title} 加入了购物车")
                    else:
                        results.append("点击了购物车")
            
            elif act_type == "join_group":
                if act.get("join_fans_group_cnt"):
                    cnt = act.get("join_fans_group_cnt")
                    results.append(f"加入了粉丝团（{cnt}次）")
    
    elif action_type == "搜索行为":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "search":
                # keyword may be in action list or in context
                pass
            elif act_type == "show":
                if act.get("show_cnt"):
                    results.append(f"系统展示了 {act.get('show_cnt')} 条搜索结果")
            elif act_type == "click":
                if act.get("click_cnt"):
                    results.append(f"点击了搜索结果 {act.get('click_cnt')} 次")
                else:
                    results.append("点击了搜索结果")

    elif action_type == "电商客服对话":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "dialogue":
                if act.get("content"):
                    dialogue_list = act.get("content")
                    if isinstance(dialogue_list, list):
                        user_msgs = sum(1 for msg in dialogue_list if msg.get("role") == "user")
                        assistant_msgs = sum(1 for msg in dialogue_list if msg.get("role") == "assistant")
                        results.append(f"与客服进行了对话（用户发送 {user_msgs} 条消息，客服回复 {assistant_msgs} 条）")
            elif act_type == "purchase":
                if act.get("paid"):
                    results.append("最终完成了支付")
                else:
                    results.append("未完成支付")

    else:
        for act in action_list:
            act_type = act.get("type", "")
            
            if act_type == "watch":
                if "play_duration" in act or "watch_seconds" in act:
                    results.append("观看了内容")
            elif act_type == "like":
                results.append("点赞")
            elif act_type == "comment":
                results.append("评论")
            elif act_type == "share":
                results.append("分享")
            elif act_type == "collect":
                results.append("收藏")
            elif act_type == "download":
                results.append("下载")
            elif act_type == "purchase":
                results.append("购买")
            elif act_type == "click":
                results.append("点击")
    
    return "，".join(results) if results else "仅浏览未进行其他操作"


def should_filter_action(action: Dict) -> bool:
    """Return True if this action record should be excluded from evaluation."""
    action_type = get_action_type(action)
    context = action.get("context", {})
    action_list = action.get("action", [])
    
    if action_type == "广告推荐":
        show_cnt = context.get("show_cnt")
        if show_cnt is not None and show_cnt == 0:
            return True

    if action_type == "直播间":
        if (len(context) == 2 and
            'live_comment_cnt' in context and
            'live_like_cnt' in context and
            context.get('live_comment_cnt') == 0 and
            context.get('live_like_cnt') == 0):
            return True

    for act in action_list:
        if act.get("type") == "watch":
            play_duration = act.get("play_duration")
            if play_duration is not None:
                if isinstance(play_duration, str):
                    try:
                        duration_value = float(play_duration.replace("秒", "").strip())
                        if duration_value == 0:
                            return True
                    except (ValueError, AttributeError):
                        pass
                elif isinstance(play_duration, (int, float)):
                    if play_duration == 0:
                        return True
            watch_seconds = act.get("watch_seconds")
            if watch_seconds is not None and watch_seconds == 0:
                return True

    return False


@functools.lru_cache(maxsize=100000)
def estimate_token_count(text: str) -> int:
    """Estimate token count using Qwen tokenizer if available, otherwise approximate."""
    tokenizer = _get_qwen_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            print(f"Warning: Qwen tokenizer encoding failed: {e}, using approximation")
    
    # Approximate: Chinese ~1.5 tok/char, other (timestamps, symbols, etc.) ~0.65 tok/char
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    
    estimated_tokens = int(chinese_chars * 1.5 + other_chars * 0.65)
    return estimated_tokens


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens using binary search; break at sentence boundary if possible."""
    if not text or max_tokens <= 0:
        return text

    current_tokens = estimate_token_count(text)
    if current_tokens <= max_tokens:
        return text

    left, right = 0, len(text)
    best_length = 0

    while left < right:
        mid = (left + right + 1) // 2
        truncated = text[:mid]
        tokens = estimate_token_count(truncated)
        if tokens <= max_tokens:
            best_length = mid
            left = mid
        else:
            right = mid - 1

    truncated_text = text[:best_length]

    last_break = -1
    for sep in ['\n\n', '\n', '。', '！', '？', '.', '!', '?']:
        pos = truncated_text.rfind(sep)
        if pos > last_break and pos > best_length * 0.7:  # keep at least 70% of content
            last_break = pos + len(sep)

    if last_break > 0:
        truncated_text = truncated_text[:last_break]

    truncated_text = truncated_text.rstrip() + "\n\n[... 内容已截断 ...]"
    return truncated_text


def get_actual_used_history(
    action_history: List[Dict],
    max_history_tokens: int = None,
    max_history_days: int = None,
    reference_timestamp: str = None
) -> Dict:
    """Return filtered/truncated history slice actually usable in the prompt."""
    from datetime import datetime, timedelta
    
    if not action_history:
        return {
            "original_count": 0,
            "filtered_count": 0,
            "actual_used_count": 0,
            "actual_used_tokens": 0,
            "actual_used_actions": [],
            "days_filtered_count": 0,
        }
    
    original_count = len(action_history)
    
    filtered_history = [action for action in action_history if not should_filter_action(action)]
    filtered_count = len(filtered_history)
    
    if not filtered_history:
        return {
            "original_count": original_count,
            "filtered_count": 0,
            "actual_used_count": 0,
            "actual_used_tokens": 0,
            "actual_used_actions": [],
            "days_filtered_count": 0,
        }
    
    days_filtered_count = 0
    if max_history_days is not None and max_history_days > 0:
        if reference_timestamp:
            try:
                if len(reference_timestamp) == 10:
                    ref_datetime = datetime.strptime(reference_timestamp, "%Y-%m-%d")
                else:
                    ref_datetime = datetime.strptime(reference_timestamp[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"Warning: could not parse reference timestamp '{reference_timestamp}', skipping day filter")
                ref_datetime = None
        else:
            ref_datetime = None

        if ref_datetime is not None:
            cutoff_datetime = ref_datetime - timedelta(days=max_history_days)
            cutoff_str = cutoff_datetime.strftime("%Y-%m-%d %H:%M:%S")
            days_filtered_history = []
            for action in filtered_history:
                action_timestamp = action.get("timestamp", "")
                if action_timestamp and action_timestamp >= cutoff_str:
                    days_filtered_history.append(action)
            days_filtered_count = filtered_count - len(days_filtered_history)
            filtered_history = days_filtered_history
    
    if not filtered_history:
        return {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "actual_used_count": 0,
            "actual_used_tokens": 0,
            "actual_used_actions": [],
            "days_filtered_count": days_filtered_count,
        }
    
    if max_history_tokens is None:
        max_history_tokens = DEFAULT_MAX_HISTORY_TOKENS

    display_actions = []
    current_tokens = 0
    for action in reversed(filtered_history):
        timestamp = action.get("timestamp", "未知时间")
        action_type = get_action_type(action, "未知行为")
        context_str = format_action_context(action)
        result_str = format_action_result(action)
        
        action_text = (
            f"【行为】时间：{timestamp}\n"
            f"  场景：{action_type}\n"
            f"  详情：{context_str}\n"
            f"  反应：{result_str}\n"
        )
        
        action_tokens = estimate_token_count(action_text)
        
        if current_tokens + action_tokens <= max_history_tokens:
            display_actions.insert(0, action)
            current_tokens += action_tokens
        else:
            break
    
    return {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "actual_used_count": len(display_actions),
        "actual_used_tokens": current_tokens,
        "actual_used_actions": display_actions,
        "days_filtered_count": days_filtered_count,
    }


def build_history_summary(
    action_history: List[Dict],
    max_history_tokens: int = None,
    max_history_days: int = None,
    reference_timestamp: str = None,
    current_action: Dict = None
) -> str:
    """Build the history string for prompt insertion."""
    if max_history_tokens is None:
        max_history_tokens = DEFAULT_MAX_HISTORY_TOKENS

    history_info = get_actual_used_history(
        action_history, 
        max_history_tokens,
        max_history_days=max_history_days,
        reference_timestamp=reference_timestamp
    )
    display_actions = history_info["actual_used_actions"]
    current_tokens = history_info["actual_used_tokens"]
    
    if not display_actions:
        if history_info["original_count"] == 0:
            return "这个用户目前还没有任何历史行为记录。"
        else:
            return "这个用户目前还没有任何有效的历史行为记录。"
    
    lines = []
    lines.append(f"以下是该用户最近的 {len(display_actions)} 条行为记录（约 {current_tokens} tokens，按时间从早到晚排列）：\n")
    
    for i, action in enumerate(display_actions, 1):
        timestamp = action.get("timestamp", "未知时间")
        action_type = get_action_type(action, "未知行为")
        context_str = format_action_context(action)
        result_str = format_action_result(action)
        
        lines.append(
            f"【行为 {i}】时间：{timestamp}\n"
            f"  场景：{action_type}\n"
            f"  详情：{context_str}\n"
            f"  反应：{result_str}\n"
        )
    
    return "\n".join(lines)


def build_test_action_description(action: Dict) -> str:
    """Format the target action as a scene description for the prompt."""
    timestamp = action.get("timestamp", "未知时间")
    action_type = get_action_type(action, "未知行为")
    context_str = format_action_context(action, is_target=True)
    
    description = (
        f"现在时间是 {timestamp}，该用户遇到了一个【{action_type}】场景。\n"
        f"场景详细信息如下：\n{context_str}"
    )
    
    return description


def build_prediction_questions(action: Dict) -> List[Dict]:
    """Build the list of prediction questions for a given action."""
    questions = []
    action_type = get_action_type(action)
    action_list = action.get("action", [])
    
    if action_type == "视频浏览":
        context = action.get("context", {})

        duration_raw = context.get("duration", 0)
        try:
            if isinstance(duration_raw, (int, float)):
                video_duration = float(duration_raw)
            elif isinstance(duration_raw, str):
                video_duration = float(duration_raw.replace("秒", "").strip()) if duration_raw else 0
            else:
                video_duration = 0
        except:
            video_duration = 0

        if video_duration > 0:
            duration_str = f"{int(video_duration)}秒"
        else:
            duration_str = "未知时长"

        watch_seconds = 0
        play_duration_str = ""
        for act in action_list:
            if act.get("type") == "watch":
                play_duration = act.get("play_duration", 0)
                if isinstance(play_duration, str):
                    play_duration_str = play_duration
                    try:
                        watch_seconds = float(play_duration.replace("秒", "").strip()) if play_duration else 0
                    except:
                        watch_seconds = 0
                elif isinstance(play_duration, (int, float)):
                    watch_seconds = float(play_duration)
                    play_duration_str = f"{watch_seconds}秒"
                break

        # completion flag: accepts either "completed" or "is_complete_play"
        completed = False
        for act in action_list:
            if act.get("type") == "watch":
                if "is_complete_play" in act:
                    completed = act.get("is_complete_play", False)
                elif video_duration > 0:
                    completed = watch_seconds >= video_duration
                break

        questions.append({
            "type": "continuous",
            "field": "video_watch_seconds",
            "true_value": watch_seconds,
            "video_duration": video_duration,
        })

        questions.append({
            "type": "binary",
            "field": "video_completed",
            "true_value": 1 if completed else 0,
        })

        liked = False
        commented = False
        shared = False
        collected = False
        followed = False
        
        for act in action_list:
            act_type = act.get("type", "")
            if act_type == "like":
                liked = True
            elif act_type == "comment":
                commented = True
            elif act_type == "share":
                shared = True
            elif act_type == "collect":
                collected = True
            elif act_type == "follow":
                followed = True
        
        questions.append({
            "type": "binary",
            "field": "video_liked",
            "true_value": 1 if liked else 0,
        })
        questions.append({
            "type": "binary",
            "field": "video_commented",
            "true_value": 1 if commented else 0,
        })
        questions.append({
            "type": "binary",
            "field": "video_shared",
            "true_value": 1 if shared else 0,
        })
        questions.append({
            "type": "binary",
            "field": "video_collected",
            "true_value": 1 if collected else 0,
        })
        questions.append({
            "type": "binary",
            "field": "video_followed",
            "true_value": 1 if followed else 0,
        })

    elif action_type == "商城购物":
        added_to_cart = False
        for act in action_list:
            if act.get("type") == "cart" and act.get("is_add_to_cart"):
                added_to_cart = True
                break

        questions.append({
            "type": "binary",
            "field": "shop_added_to_cart",
            "true_value": 1 if added_to_cart else 0,
        })

        order_success = False
        for act in action_list:
            if act.get("type") == "purchase":
                if act.get("is_pay") or act.get("paid") or act.get("order_success"):
                    order_success = True
                    break

        questions.append({
            "type": "binary",
            "field": "shop_order_success",
            "true_value": 1 if order_success else 0,
        })

    elif action_type == "广告推荐":
        watch_seconds = 0
        for act in action_list:
            if act.get("type") == "watch":
                if "watch_seconds" in act:
                    watch_seconds = act.get("watch_seconds", 0)
                elif "play_duration" in act:
                    play_duration = act.get("play_duration")
                    if isinstance(play_duration, str):
                        try:
                            watch_seconds = float(play_duration.replace("秒", "").strip())
                        except:
                            watch_seconds = 0
                    elif isinstance(play_duration, (int, float)):
                        watch_seconds = float(play_duration)
                break

        if watch_seconds > 0 or any(act.get("type") == "watch" for act in action_list):
            questions.append({
                "type": "continuous",
                "field": "ad_watch_seconds",
                "true_value": watch_seconds,
            })

        liked = False
        for act in action_list:
            if act.get("type") == "like":
                liked = True
                break
        questions.append({
            "type": "binary",
            "field": "ad_liked",
            "true_value": 1 if liked else 0,
        })

        commented = False
        for act in action_list:
            if act.get("type") == "comment":
                commented = True
                break
        questions.append({
            "type": "binary",
            "field": "ad_commented",
            "true_value": 1 if commented else 0,
        })

        activated = False
        for act in action_list:
            if act.get("type") == "activation" and act.get("activation_cnt", 0) > 0:
                activated = True
                break
        questions.append({
            "type": "binary",
            "field": "ad_activated",
            "true_value": 1 if activated else 0,
        })

        form_submitted = False
        for act in action_list:
            if act.get("type") == "submit" and act.get("form_submit_total_cnt", 0) > 0:
                form_submitted = True
                break
        questions.append({
            "type": "binary",
            "field": "ad_form_submitted",
            "true_value": 1 if form_submitted else 0,
        })

    elif action_type == "直播间":
        watch_seconds = 0
        for act in action_list:
            if act.get("type") == "watch":
                play_duration = act.get("play_duration", 0)
                if isinstance(play_duration, str):
                    try:
                        watch_seconds = float(play_duration.replace("秒", "").strip())
                    except:
                        watch_seconds = 0
                elif isinstance(play_duration, (int, float)):
                    watch_seconds = float(play_duration)
                break

        questions.append({
            "type": "continuous",
            "field": "live_watch_seconds",
            "true_value": watch_seconds,
        })

        liked = False
        for act in action_list:
            if act.get("type") == "like":
                liked = True
                break
        questions.append({
            "type": "binary",
            "field": "live_liked",
            "true_value": 1 if liked else 0,
        })

        commented = False
        for act in action_list:
            if act.get("type") == "comment":
                commented = True
                break
        
        questions.append({
            "type": "binary",
            "field": "live_commented",
            "true_value": 1 if commented else 0,
        })

        sent_gift = False
        for act in action_list:
            if act.get("type") == "send_gift":
                sent_gift = True
                break
        questions.append({
            "type": "binary",
            "field": "live_sent_gift",
            "true_value": 1 if sent_gift else 0,
        })

        followed = False
        for act in action_list:
            if act.get("type") == "follow":
                followed = True
                break
        questions.append({
            "type": "binary",
            "field": "live_followed",
            "true_value": 1 if followed else 0,
        })

        shared = False
        for act in action_list:
            if act.get("type") == "share":
                shared = True
                break
        questions.append({
            "type": "binary",
            "field": "live_shared",
            "true_value": 1 if shared else 0,
        })

        if action.get("context", {}).get("is_shop_live"):
            clicked_cart = False
            for act in action_list:
                if act.get("type") == "click_cart" and act.get("is_click_cart_action"):
                    clicked_cart = True
                    break
            questions.append({
                "type": "binary",
                "field": "live_clicked_cart",
                "true_value": 1 if clicked_cart else 0,
            })
    
    elif action_type == "搜索行为":
        keyword = None
        query_category = None
        for act in action_list:
            if act.get("type") == "search":
                keyword = act.get("keyword")
                query_category = act.get("query_category")
                break
        if keyword:
            questions.append({
                "type": "text",
                "field": "search_keyword",
                "true_value": keyword,
                "query_category": query_category,
            })
    
    elif action_type == "电商客服对话":
        # Skip patterns: messages containing "评价" (review-related) or "会话转移" are noise
        SKIP_PATTERNS = ["评价", "会话转移"]

        def should_skip_user_message(content: str) -> bool:
            if not content or not content.strip():
                return True
            for pattern in SKIP_PATTERNS:
                if pattern in content:
                    return True
            return False

        def is_too_short(content: str) -> bool:
            if not content:
                return True
            return len(content.strip()) <= 2

        dialogue_content = []
        for act in action_list:
            if act.get("type") == "dialogue":
                dialogue_content = act.get("content", [])
                break

        if dialogue_content and len(dialogue_content) > 0:
            target_user_message = None
            target_user_message_idx = -1

            for i in range(len(dialogue_content) - 1, -1, -1):
                msg = dialogue_content[i]
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if should_skip_user_message(content):
                        continue
                    if is_too_short(content):
                        continue
                    target_user_message = content
                    target_user_message_idx = i
                    break

            if target_user_message is None:
                pass
            else:
                has_dialogue_history = False
                has_prior_user_speech = False
                context_dialogue = []

                for j in range(target_user_message_idx):
                    msg = dialogue_content[j]
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"] and content and content.strip():
                        has_dialogue_history = True
                        context_dialogue.append(msg)
                        if role == "user":
                            has_prior_user_speech = True

                action_context = action.get("context", {})
                context_info = format_action_context(action)

                if not has_dialogue_history:
                    question_text = (
                        f"这是一次电商客服对话场景。用户即将开始与客服对话。\n\n"
                        f"【订单/咨询背景信息】\n{context_info}\n\n"
                        f"请你站在这个用户的角度，结合订单信息和咨询背景，预测用户最有可能说的第一句话是什么？\n\n"
                        f"请直接输出用户会说的话（不要加引号，直接输出内容即可）："
                    )
                else:
                    MAX_DIALOGUE_TOKENS = 4000
                    truncated_dialogue = []
                    current_dialogue_tokens = 0

                    for msg in reversed(context_dialogue):
                        msg_text = f"{'用户' if msg.get('role')=='user' else '客服'}: {msg.get('content', '')}\n"
                        msg_tokens = estimate_token_count(msg_text)
                        if current_dialogue_tokens + msg_tokens <= MAX_DIALOGUE_TOKENS:
                            truncated_dialogue.insert(0, msg)
                            current_dialogue_tokens += msg_tokens
                        else:
                            break

                    dialogue_text = "\n".join([
                        f"{'用户' if m.get('role')=='user' else '客服'}: {m.get('content', '')}"
                        for m in truncated_dialogue
                    ])

                    if len(truncated_dialogue) < len(context_dialogue):
                        dialogue_text = f"（以下是最近 {len(truncated_dialogue)} 轮对话，更早的 {len(context_dialogue) - len(truncated_dialogue)} 轮已省略）\n" + dialogue_text
                    
                    question_text = (
                        f"这是一段电商客服对话记录。\n\n"
                        f"【订单/咨询背景信息】\n{context_info}\n\n"
                        f"【对话历史记录】\n{dialogue_text}\n\n"
                        f"请你站在这个用户的角度，结合TA的沟通风格、当前遇到的问题以及对话的上下文，预测TA接下来最有可能说的一句话是什么？\n\n"
                        f"请直接输出用户接下来会说的话（不要加引号，直接输出内容即可）："
                    )
                
                questions.append({
                    "question": question_text,
                    "type": "text",
                    "field": "next_user_message",
                    "true_value": target_user_message,
                    "context_dialogue": context_dialogue,
                    "has_dialogue_history": has_dialogue_history,
                    "has_prior_user_speech": has_prior_user_speech,
                })
    
    return questions


def build_single_binary_prompt(
    user_profile: str,
    action_history: List[Dict],
    test_action: Dict,
    question_info: Dict,
    max_history_tokens: int = None,
    max_history_days: int = None,
) -> Dict:
    """Build prompt for a single binary (Yes/No) prediction question."""
    if should_filter_action(test_action):
        return None

    scenario_desc = build_test_action_description(test_action)
    yes_no_question = _get_yes_no_question_by_field(question_info)

    if user_profile:
        first_period_idx = user_profile.find('。')
        if first_period_idx != -1:
            user_profile_short = user_profile[:first_period_idx]
        else:
            user_profile_short = user_profile
    else:
        user_profile_short = ""
    
    prompt_parts = [
        "你是快手平台的一位真实用户。你的核心任务是：基于给定的历史行为序列，推断该用户的兴趣偏好、消费水平和性格特征，并据此模拟TA在当前某场景下的真实决策。",
        "## 核心原则",
        "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
        "2. **行为连贯性**：新决策应与用户的历史行为模式保持内在逻辑一致，体现其稳定的偏好和习惯。",
        "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签，从数据中发现真实的个性特征。",
        "4. **情境敏感性**：决策预测需考虑当前场景的特殊性，平衡长期偏好与短期情境因素的影响。",
        "5. **真实性优先**：模拟真实用户可能做出的选择，包括不感兴趣、犹豫、跳过等消极行为，而非总是给出积极响应。",
        "## 输入一：用户画像",
        "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
        user_profile_short,
        "## 输入二：历史行为轨迹信息",
        "这是该用户在过去一段时间内的真实操作记录（包含直播、商城、视频、广告等跨场景行为）。请仔细分析这些行为背后的动机和倾向，挖掘其隐含的长期偏好和短期意图：",
        "",  # history placeholder, filled below
        "## 输入三：当前测试场景",
        "用户现在遇到了以下场景：",
        scenario_desc,
        "## 预测任务",
        "请代入该用户视角，回答以下问题：",
        yes_no_question,
        "## 输出要求",
        "**请只输出 Yes 或 No，不要输出任何其他内容、解释或分析。**",
        "你的回答：",
    ]
    
    base_prompt = "\n".join(prompt_parts)
    base_tokens = estimate_token_count(base_prompt)
    total_token_limit = max_history_tokens if max_history_tokens is not None else DEFAULT_MAX_HISTORY_TOKENS
    actual_max_history_tokens = max(0, total_token_limit - base_tokens - 100)  # 100 token buffer

    reference_timestamp = test_action.get("timestamp")
    history_summary = build_history_summary(
        action_history, 
        max_history_tokens=actual_max_history_tokens,
        max_history_days=max_history_days,
        reference_timestamp=reference_timestamp,
        current_action=test_action
    )
    
    prompt_parts[12] = history_summary
    prompt = "\n".join(prompt_parts)
    
    return {
        "prompt": prompt,
        "question_info": question_info,
        "test_action": test_action,
    }


def build_single_continuous_prompt(
    user_profile: str,
    action_history: List[Dict],
    test_action: Dict,
    question_info: Dict,
    max_history_tokens: int = None,
    max_history_days: int = None,
) -> Dict:
    """Build prompt for a single continuous (numeric) prediction question."""
    if should_filter_action(test_action):
        return None

    scenario_desc = build_test_action_description(test_action)
    field = question_info.get("field", "")

    continuous_questions = {
        "video_watch_seconds": "你预计该用户会在这个视频上观看多少秒？",
        "live_watch_seconds": "你预计该用户会在这个直播间停留多少秒？",
        "ad_watch_seconds": "你预计该用户会在这条广告上停留多少秒？",
    }

    if field in continuous_questions:
        question_text = continuous_questions[field]
    else:
        question_text = f"你预计该用户会在这里停留多少秒？（{field}）"

    if user_profile:
        first_period_idx = user_profile.find('。')
        if first_period_idx != -1:
            user_profile_short = user_profile[:first_period_idx]
        else:
            user_profile_short = user_profile
    else:
        user_profile_short = ""

    prompt_parts = [
        "你是快手平台的一位真实用户。你的核心任务是：基于给定的历史行为序列，推断该用户的兴趣偏好、消费水平和性格特征，并据此模拟TA在当前某场景下的真实决策。",
        "## 核心原则",
        "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
        "2. **行为连贯性**：新决策应与用户的历史行为模式保持内在逻辑一致，体现其稳定的偏好和习惯。",
        "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签，从数据中发现真实的个性特征。",
        "4. **情境敏感性**：决策预测需考虑当前场景的特殊性，平衡长期偏好与短期情境因素的影响。",
        "5. **真实性优先**：模拟真实用户可能做出的选择，包括不感兴趣、犹豫、跳过等消极行为，而非总是给出积极响应。",
        "## 输入一：用户画像",
        "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
        user_profile_short,
        "## 输入二：历史行为轨迹信息",
        "这是该用户在过去一段时间内的真实操作记录（包含直播、商城、视频、广告等跨场景行为）。请仔细分析这些行为背后的动机和倾向，挖掘其隐含的长期偏好和短期意图：",
        "",  # history placeholder, filled below
        "## 输入三：当前测试场景",
        "用户现在遇到了以下场景：",
        scenario_desc,
        "## 预测任务",
        "请代入该用户视角，回答以下问题：",
        question_text,
        "## 输出要求",
        "**请只输出一个整数，不要输出任何其他内容、解释或单位。**",
        "你的回答：",
    ]

    base_prompt = "\n".join(prompt_parts)
    base_tokens = estimate_token_count(base_prompt)
    total_token_limit = max_history_tokens if max_history_tokens is not None else DEFAULT_MAX_HISTORY_TOKENS
    actual_max_history_tokens = max(0, total_token_limit - base_tokens - 100)  # 100 token buffer

    reference_timestamp = test_action.get("timestamp")
    history_summary = build_history_summary(
        action_history,
        max_history_tokens=actual_max_history_tokens,
        max_history_days=max_history_days,
        reference_timestamp=reference_timestamp,
        current_action=test_action
    )

    prompt_parts[12] = history_summary
    prompt = "\n".join(prompt_parts)

    return {
        "prompt": prompt,
        "question_info": question_info,
        "test_action": test_action,
    }


def build_single_text_prompt(
    user_profile: str,
    action_history: List[Dict],
    test_action: Dict,
    question_info: Dict,
    max_history_tokens: int = None,
    max_history_days: int = None,
) -> Dict:
    """Build prompt for a single text prediction question (search keyword, dialogue, etc.)."""
    if should_filter_action(test_action):
        return None

    question_text = question_info.get("question", "")
    field = question_info.get("field", "")

    if user_profile:
        first_period_idx = user_profile.find('。')
        if first_period_idx != -1:
            user_profile_short = user_profile[:first_period_idx]
        else:
            user_profile_short = user_profile
    else:
        user_profile_short = ""

    history_idx = -1

    if field == "search_keyword":
        # Don't expose the target scene — we're predicting what the user will search for
        prompt_parts = [
            "你是快手平台的一位真实用户。你的核心任务是：基于给定的历史行为序列，推断该用户的兴趣偏好、消费习惯和当前可能的需求，并预测TA接下来想要搜索的内容。",
            "## 核心原则",
            "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
            "2. **行为连贯性**：预测的搜索内容应与用户的历史行为模式保持内在逻辑一致，体现其稳定的偏好和当前意图。",
            "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签，从数据中发现真实的个性特征。",
            "4. **情境敏感性**：结合用户近期的行为趋势，推断其当前可能的需求或好奇心。",
            "5. **真实性优先**：预测真实用户可能搜索的内容，包括日常需求、兴趣探索、购物需求等。",
            "## 输入一：用户画像",
            "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
            user_profile_short,
            "## 输入二：历史行为轨迹信息",
            "这是该用户在过去一段时间内的真实操作记录（包含直播、商城、视频、广告、搜索等跨场景行为）。请仔细分析这些行为背后的动机和倾向，挖掘其隐含的长期偏好和短期意图：",
            "",  # history placeholder
            "## 预测任务",
            "请代入该用户视角，预测TA现在打开搜索框后会输入什么关键词。",
            "## 输出要求",
            "**请只输出搜索关键词内容本身，不要输出任何其他内容、解释、引号或分析。**",
            "你的回答：",
        ]
        history_idx = 12
    elif field == "next_user_message":
        has_dialogue_history = question_info.get("has_dialogue_history", True)
        prompt_parts = [
            "你是一位真实的电商平台用户。你的核心任务是：基于给定的历史行为序列，推断该用户的沟通风格、性格特征和当前需求，并据此模拟TA在客服对话中的真实表达。",
            "## 核心原则",
            "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
            "2. **风格连贯性**：预测的表达应与用户在历史中展现的沟通风格、语气和用词习惯保持一致。",
            "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签。",
            "4. **情境敏感性**：结合当前对话的上下文、用户遇到的问题和情绪状态进行预测。",
            "5. **真实性优先**：模拟真实用户可能说出的话，体现其独特的沟通风格和当前情绪。",
            "## 输入一：用户画像",
            "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
            user_profile_short,
            "## 输入二：历史行为轨迹信息",
            "这是该用户在过去一段时间内的真实操作记录（包含直播、商城、视频、广告等跨场景行为）。请仔细分析这些行为背后的动机和倾向，挖掘其沟通风格和性格特征：",
            "",
            "## 输入三：当前客服对话场景",
            question_text,
            "## 输出要求",
            "**请只输出用户会说的话，不要输出任何其他内容、解释、引号或分析。直接输出对话内容即可。**",
            "你的回答：",
        ]
        history_idx = 12
    else:
        scenario_desc = build_test_action_description(test_action)
        prompt_parts = [
            "你是快手平台的一位真实用户。你的核心任务是：基于给定的历史行为序列，推断该用户的兴趣偏好、沟通风格和性格特征，并据此模拟TA在当前场景下的真实表达。",
            "## 核心原则",
            "1. **数据驱动**：所有推断必须基于历史行为数据中的客观证据，避免无根据的臆测和假设。",
            "2. **行为连贯性**：预测的文本应与用户的历史行为模式和表达风格保持内在逻辑一致。",
            "3. **个体差异性**：充分尊重每个用户的独特性，不套用刻板印象或群体标签。",
            "4. **情境敏感性**：预测需考虑当前场景的特殊性，平衡长期风格与短期情境因素的影响。",
            "5. **真实性优先**：模拟真实用户可能说出的话，体现其独特的沟通风格。",
            "## 输入一：用户画像",
            "这是该用户的平台基本信息，可以作为理解用户背景的参考：",
            user_profile_short,
            "## 输入二：历史行为轨迹信息",
            "这是该用户在过去一段时间内的真实操作记录：",
            "",
            "## 输入三：当前测试场景",
            "用户现在遇到了以下场景：",
            scenario_desc,
            "## 预测任务",
            "请代入该用户视角，回答以下问题：",
            question_text,
            "## 输出要求",
            "**请只输出预测的文本内容本身，不要输出任何其他内容、解释、引号或分析。**",
            "你的回答：",
        ]
        history_idx = 12

    base_prompt = "\n".join(prompt_parts)
    base_tokens = estimate_token_count(base_prompt)
    total_token_limit = max_history_tokens if max_history_tokens is not None else DEFAULT_MAX_HISTORY_TOKENS
    actual_max_history_tokens = max(0, total_token_limit - base_tokens - 100)  # 100 token buffer

    reference_timestamp = test_action.get("timestamp")
    history_summary = build_history_summary(
        action_history,
        max_history_tokens=actual_max_history_tokens,
        max_history_days=max_history_days,
        reference_timestamp=reference_timestamp,
        current_action=test_action
    )

    prompt_parts[history_idx] = history_summary
    prompt = "\n".join(prompt_parts)

    return {
        "prompt": prompt,
        "question_info": question_info,
        "test_action": test_action,
    }


def _get_yes_no_question_by_field(question_info: Dict) -> str:
    """Map a question field name to its Yes/No question string."""
    field = question_info.get("field", "")
    yes_no_questions = {
        "video_completed": "基于这个用户的观看习惯，该用户会把这个视频完整看完吗？",
        "video_liked": "结合这个用户的互动习惯和对视频内容的喜爱程度，该用户会为这个视频点赞吗？",
        "video_commented": "考虑到这个用户的表达欲望和参与度，该用户会在这个视频下发表评论吗？",
        "video_shared": "基于这个用户的分享习惯和社交行为，该用户会把这个视频分享给朋友吗？",
        "video_collected": "根据这个用户的收藏偏好，该用户会收藏这个视频吗？",
        "video_followed": "考虑到这个用户的关注习惯，该用户会关注这个视频的作者吗？",
        "shop_added_to_cart": "根据这个用户的购物习惯，该用户会把这件商品加入购物车吗？",
        "shop_order_success": "结合这个用户的购物偏好和消费能力，该用户会购买这件商品吗？",
        "ad_liked": "考虑到这个用户对广告内容的喜爱程度，该用户会为这条广告点赞吗？",
        "ad_commented": "基于这个用户的表达欲望，该用户会在这条广告下发表评论吗？",
        "ad_activated": "假设用户点击了这条广告，基于TA的行为特征，该用户会激活或注册广告中的应用/服务吗？",
        "ad_form_submitted": "基于这个用户的行为特征和对广告的兴趣程度，该用户会填写并提交广告中的表单吗？",
        "live_liked": "结合这个用户的互动习惯，该用户会为主播点赞吗？",
        "live_commented": "考虑到这个用户的表达欲望，该用户会在直播间发送弹幕或评论吗？",
        "live_sent_gift": "基于这个用户的消费能力和打赏习惯，该用户会在直播间给主播送礼物吗？",
        "live_followed": "根据这个用户的关注习惯，该用户会关注这个主播吗？",
        "live_shared": "基于这个用户的分享习惯，该用户会把这个直播间分享给朋友吗？",
        "live_clicked_cart": "根据该用户的购物习惯，该用户会把直播间的商品加入购物车吗？",
        "search_clicked": "当用户搜索这个关键词后，该用户会点击搜索结果吗？",
    }

    if field in yes_no_questions:
        return yes_no_questions[field]
    return f"该用户会执行此操作吗？（{field}）"


def get_binary_questions_for_action(action: Dict) -> List[Dict]:
    """Return only the binary prediction questions for an action."""
    questions = build_prediction_questions(action)
    return [q for q in questions if q.get("type") == "binary"]


def get_all_questions_for_action(action: Dict) -> List[Dict]:
    """Return all prediction questions (binary + continuous + text) for an action."""
    return build_prediction_questions(action)
