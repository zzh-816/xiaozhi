"""记忆管理层：负责提取记忆、协调存储和检索"""
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Optional

from ..base import logger
from .memory_storage import MemoryStorage

TAG = __name__


PROFILE_EXTRACTION_PROMPT = """提取用户画像信息（稳定的个人特征）。

【核心规则】
1. 只从用户（user）消息提取，不从assistant提取
2. 只提取稳定的个人特征：姓名、年龄、职业、位置、喜好、价值观、边界、沟通偏好
3. 不提取临时状态、事件、计划（这些属于其他类型）
4. 必须返回JSON格式：{{"memories": ["键: 值"]}} 或 {{"memories": []}}

【提取类型】
- 姓名/昵称、年龄、职业、位置、喜好、价值观、边界、沟通偏好

【禁止提取】
- 临时状态："今天很累"、"今天心情不好"
- 事件："去了健身房"、"参加了会议"
- 计划："我下周六有考试"

【示例】
输入：user: 我叫小明，今年22岁
输出：{{"memories": ["姓名/昵称: 小明", "年龄: 22"]}}

输入：user: 我是学生，在北京上学
输出：{{"memories": ["职业: 学生", "位置: 北京"]}}

输入：user: 我喜欢蓝色
输出：{{"memories": ["喜好: 蓝色"]}}

输入：user: 我今天很累
输出：{{"memories": []}}

输入：user: 我下周六有英语考试
输出：{{"memories": []}}

对话内容：
{dialogue_text}

当前日期：{current_date}

请提取用户画像信息，返回JSON。"""


FACTS_EXTRACTION_PROMPT = """提取已发生的事件（过去完成的事件，有实际意义）。

【核心规则】
1. 只从用户（user）消息提取，不从assistant提取
2. 只提取已发生的事件（参加了、完成了、去了、经历了等）
3. 必须将相对时间转换为实际日期（YYYY-MM-DD格式）
4. 【重要】如果用户使用了相对时间词（如"昨天"、"上周二"），在提取的内容中也要保留这个相对时间词
5. 必须返回JSON：{{"memories": ["日期: 事件"]}} 或 {{"memories": []}}

【日期计算规则】（当前日期：{{current_date}}（{{current_weekday}}））
  - '今天' = {{current_date}}
  - '昨天' = {{current_date}}的前一天
- '前天' = {{current_date}}的前两天
- '大前天' = {{current_date}}的前三天
- '这周X' = 本周的星期X（如果今天已经是星期X，就是今天；否则是本周内该星期X的日期）
- '上周X' = 上周的星期X（从今天往前推，找到上周的星期X）
- 'N天前' = {{current_date}}往前推N天

示例计算（今天是2026-01-04（星期六））：
- "昨天" = 2026-01-03（1月4日的前一天）
- "前天" = 2026-01-02（1月4日的前两天）
- "上周二" = 2025-12-30（从1月4日星期六往前推，找到上周的星期二）
- "上周三" = 2025-12-31（从1月4日星期六往前推，找到上周的星期三）

**关键：必须准确计算，不要误算日期！**

【禁止提取】
- 提问/测试："我叫什么名字"、"我今天干嘛了"、"用户提出了问题"
- 计划/未来："我明天要开会"、"我下周六有考试"（包含"要"、"将"、"计划"）
- 画像信息："我喜欢蓝色"、"我是学生"
- 社交行为："你好"、"再见"、"用户道别"

【示例】
输入：user: 我今天去爬山了
输出：{{"memories": ["2026-01-02: 用户今天去爬山了"]}}  # 今天是2026-01-02，保留"今天"

输入：user: 我昨天去爬山了
输出：{{"memories": ["2026-01-01: 用户昨天去爬山了"]}}  # 今天是2026-01-02，保留"昨天"

输入：user: 我上周二过了恋爱纪念日
输出：{{"memories": ["2025-12-30: 用户上周二过了恋爱纪念日"]}}  # 今天是2026-01-04（星期六），上周二=12月30日，保留"上周二"

输入：user: 我下周三有英语考试
输出：{{"memories": []}}  # 未来计划，属于commitments

输入：user: 我叫什么名字
输出：{{"memories": []}}  # 提问，不是事件

输入：user: 我喜欢蓝色
输出：{{"memories": []}}  # 画像信息，不是事件

输入：user: 我明天要开会
输出：{{"memories": []}}  # 未来计划，包含"要"

对话内容：
{dialogue_text}

当前日期：{current_date}（{current_weekday}）

请提取已发生的事件，返回JSON。日期必须使用计算后的实际日期，并且保留相对时间词。"""


COMMITMENTS_EXTRACTION_PROMPT = """提取未来的计划/承诺（包含"要"、"将"、"计划"等未来词汇）。

【核心规则】
1. 只从用户（user）消息提取，不从assistant提取
2. 只提取未来的计划（要、将、计划、打算、准备等）
3. 必须将相对时间转换为实际日期（YYYY-MM-DD格式）
4. 必须返回JSON：{{"memories": ["日期: 计划"]}} 或 {{"memories": []}}

【日期计算规则】（当前日期：{{current_date}}（{{current_weekday}}））
  - '明天' = {{current_date}}的后一天
- '这周X' = 本周的星期X（如果今天已经是星期X，就是今天；否则是本周内该星期X的日期）
- '下周X' = 从今天开始，找到下一个星期X的日期（不是下下周，就是下一个该星期几）
- '下周' = 下周一（如果只说"下周"没有具体星期几，默认下周一）
- '下下X' = 下下周的星期X
- '下个月' = 下个月1号（如果今天是12月，则下个月是明年1月1号）
  **重要计算步骤：**
  1. 确定今天是星期几
  2. 计算到下一个星期X需要多少天
  3. 如果今天是星期X，则"下周X"是7天后的星期X
  4. 如果今天不是星期X，则找到本周或下周的第一个星期X

示例计算（今天是2026-01-03（星期六））：
- "下周一" = 2026-01-05（从1月3日星期六开始，下一个星期一是1月5日，距离2天）
- "下周二" = 2026-01-06（从1月3日星期六开始，下一个星期二是1月6日，距离3天）
- "下周三" = 2026-01-07（从1月3日星期六开始，下一个星期三是1月7日，距离4天）
- "下周四" = 2026-01-08（从1月3日星期六开始，下一个星期四是1月8日，距离5天）
- "下周五" = 2026-01-09（从1月3日星期六开始，下一个星期五是1月9日，距离6天）
- "下周六" = 2026-01-10（从1月3日星期六开始，下一个星期六是1月10日，距离7天）
- "下周日" = 2026-01-04（从1月3日星期六开始，下一个星期日是1月4日，距离1天）
- "下周" = 2026-01-05（如果只说"下周"没有具体星期几，默认下周一，距离2天）
- "下个月" = 2026-02-01（下个月1号）

**关键：必须准确计算，不要误算成下下周的日期！**

【禁止提取】
- 已发生事件："我昨天去爬山了"（属于facts）
- 只是想法："我应该开始锻炼了"（没有明确承诺）

【示例】
输入：user: 我下周三有英语考试
输出：{{"memories": ["2026-01-07: 用户下周三有英语考试"]}}  # 今天是2026-01-02（星期五），下周三=1月7日

输入：user: 我下周二有英语考试
输出：{{"memories": ["2026-01-06: 用户下周二有英语考试"]}}  # 今天是2026-01-03（星期六），下周二=1月6日（距离3天）

输入：user: 我明天要开会
输出：{{"memories": ["2026-01-03: 用户明天要开会"]}}  # 今天是2026-01-02，明天=1月3日

输入：user: 我下周要去旅游
输出：{{"memories": ["2026-01-05: 用户下周要去旅游"]}}  # 今天是2026-01-03（星期六），下周=下周一=1月5日

输入：user: 我下个月要去旅游
输出：{{"memories": ["2026-02-01: 用户下个月要去旅游"]}}  # 今天是2026-01-03，下个月=2月1号

输入：user: 我昨天去爬山了
输出：{{"memories": []}}  # 已发生，属于facts

输入：user: 我应该开始锻炼了
输出：{{"memories": []}}  # 只是想法，不是明确计划

对话内容：
{dialogue_text}

当前日期：{current_date}（{current_weekday}）

请提取未来的计划/承诺，返回JSON。日期必须使用计算后的实际日期。"""


class MemoryManager:
    """记忆管理层：负责使用LLM提取记忆，并协调存储"""
    
    def __init__(self, storage: MemoryStorage, llm, user_id: str = None):
        self.storage = storage
        self.llm = llm
        self.user_id = user_id
        # 记录上次提取的位置（已处理的用户消息数量），每次新会话从0开始
        self.last_extracted_user_message_count = 0
        logger.bind(tag=TAG).info(f"初始化MemoryManager: user_id={user_id}, 计数从0开始（新会话）")
    
    def reset_session(self):
        """重置会话：将提取计数重置为0（每次新会话开始时调用）"""
        self.last_extracted_user_message_count = 0
        logger.bind(tag=TAG).info(f"会话已重置：提取计数重置为0（新会话开始）")
    
    async def extract_memories(self, messages: List) -> Dict[str, any]:
        """使用LLM提取记忆（profile、facts、commitments）- 增量提取，只处理新增对话"""
        if not self.llm:
            return {"profile": [], "facts": [], "commitments": []}
        
        # 统计用户消息，只处理新增的部分（增量提取）
        user_messages = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                if msg.role == 'user':  # 只提取用户消息
                    content = msg.content if msg.content is not None else ""
                    if content.strip():
                        user_messages.append(msg)
            elif isinstance(msg, dict) and msg.get('role') == 'user':
                # 兼容字典格式的消息
                content = msg.get('content', '')
                if content and content.strip():
                    user_messages.append(msg)
        
        logger.bind(tag=TAG).debug(f"消息统计：总消息数={len(messages)}, 用户消息数={len(user_messages)}, 已处理={self.last_extracted_user_message_count}")
        
        # 只处理新增的用户消息（从上次提取位置之后）
        new_user_messages = user_messages[self.last_extracted_user_message_count:]
        
        if not new_user_messages:
            logger.bind(tag=TAG).info(f"没有新增的用户消息需要提取（已处理 {self.last_extracted_user_message_count} 条，总共 {len(user_messages)} 条）")
            # 显示最后几条用户消息，帮助调试
            if user_messages:
                logger.bind(tag=TAG).debug(f"最后3条用户消息: {[msg.content if hasattr(msg, 'content') else msg.get('content', '') for msg in user_messages[-3:]]}")
            return {"profile": [], "facts": [], "commitments": []}
        
        logger.bind(tag=TAG).info(f"增量提取：处理新增的 {len(new_user_messages)} 条用户消息（已处理 {self.last_extracted_user_message_count} 条，总共 {len(user_messages)} 条）")
        for i, msg in enumerate(new_user_messages, 1):
            content = msg.content if hasattr(msg, 'content') else str(msg)
            logger.bind(tag=TAG).debug(f"  新增消息 {i}: {content[:100]}")
        
        # 构建对话文本（只包含新增的用户消息）
        dialogue_text = ""
        for msg in new_user_messages:
            content = msg.content if msg.content is not None else ""
            if content.strip():
                dialogue_text += f"user: {content}\n"
        
        # 获取当前日期和星期几
        from core.utils.current_time import get_current_weekday
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_weekday = get_current_weekday()
        
        results = {"profile": [], "facts": [], "commitments": []}
        
        def extract_json_from_response(response_text: str) -> Optional[Dict]:
            """从LLM响应中提取JSON（处理markdown代码块等格式）"""
            if not response_text:
                return None
            
            # 清理响应文本
            response_text = response_text.strip()
            
            # 尝试直接解析
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass
            
            # 尝试提取markdown代码块中的JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 尝试提取第一个JSON对象
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            return None
        
        try:
            # 提取profile
            profile_start = time.time()
            profile_prompt = PROFILE_EXTRACTION_PROMPT.format(
                dialogue_text=dialogue_text,
                current_date=current_date
            )
            profile_llm_start = time.time()
            profile_result = self.llm.response_no_stream("", profile_prompt, max_tokens=500)
            profile_llm_time = time.time() - profile_llm_start
            logger.bind(tag=TAG).info(f"Profile LLM调用耗时: {profile_llm_time:.3f}s")
            logger.bind(tag=TAG).debug(f"Profile LLM原始响应: {profile_result[:500]}")
            
            profile_parse_start = time.time()
            profile_data = extract_json_from_response(profile_result)
            profile_parse_time = time.time() - profile_parse_start
            profile_total_time = time.time() - profile_start
            if profile_data:
                if profile_data.get("memories"):
                    results["profile"] = profile_data["memories"]
                    logger.bind(tag=TAG).info(f"Profile提取成功: {len(results['profile'])} 条，总耗时: {profile_total_time:.3f}s（LLM: {profile_llm_time:.3f}s, 解析: {profile_parse_time:.3f}s）")
                    for item in results["profile"]:
                        logger.bind(tag=TAG).info(f"  - Profile项: {item}")
                else:
                    logger.bind(tag=TAG).debug(f"Profile响应中没有memories字段: {profile_data}，耗时: {profile_total_time:.3f}s")
            else:
                logger.bind(tag=TAG).error(f"Profile提取结果解析失败，无法提取JSON，原始结果: {profile_result[:500]}，耗时: {profile_total_time:.3f}s")
            
            # 提取facts
            facts_start = time.time()
            facts_prompt = FACTS_EXTRACTION_PROMPT.format(
                dialogue_text=dialogue_text,
                current_date=current_date,
                current_weekday=current_weekday
            )
            facts_llm_start = time.time()
            facts_result = self.llm.response_no_stream("", facts_prompt, max_tokens=500)
            facts_llm_time = time.time() - facts_llm_start
            logger.bind(tag=TAG).info(f"Facts LLM调用耗时: {facts_llm_time:.3f}s")
            logger.bind(tag=TAG).info(f"Facts LLM原始响应: {facts_result[:500]}")
            
            facts_parse_start = time.time()
            facts_data = extract_json_from_response(facts_result)
            facts_parse_time = time.time() - facts_parse_start
            facts_total_time = time.time() - facts_start
            if facts_data:
                if facts_data.get("memories"):
                    results["facts"] = facts_data["memories"]
                    logger.bind(tag=TAG).info(f"Facts提取成功: {len(results['facts'])} 条，总耗时: {facts_total_time:.3f}s（LLM: {facts_llm_time:.3f}s, 解析: {facts_parse_time:.3f}s）")
                    for item in results["facts"]:
                        logger.bind(tag=TAG).info(f"  - Fact: {item}")
                else:
                    logger.bind(tag=TAG).warning(f"Facts响应中没有memories字段: {facts_data}，耗时: {facts_total_time:.3f}s")
            else:
                logger.bind(tag=TAG).error(f"Facts提取结果解析失败，无法提取JSON，原始结果: {facts_result[:500]}，耗时: {facts_total_time:.3f}s")
            
            # 提取commitments
            commitments_start = time.time()
            commitments_prompt = COMMITMENTS_EXTRACTION_PROMPT.format(
                dialogue_text=dialogue_text,
                current_date=current_date,
                current_weekday=current_weekday
            )
            commitments_llm_start = time.time()
            commitments_result = self.llm.response_no_stream("", commitments_prompt, max_tokens=500)
            commitments_llm_time = time.time() - commitments_llm_start
            logger.bind(tag=TAG).info(f"Commitments LLM调用耗时: {commitments_llm_time:.3f}s")
            logger.bind(tag=TAG).info(f"Commitments LLM原始响应: {commitments_result[:500]}")
            
            commitments_parse_start = time.time()
            commitments_data = extract_json_from_response(commitments_result)
            commitments_parse_time = time.time() - commitments_parse_start
            commitments_total_time = time.time() - commitments_start
            if commitments_data:
                if commitments_data.get("memories"):
                    results["commitments"] = commitments_data["memories"]
                    logger.bind(tag=TAG).info(f"Commitments提取成功: {len(results['commitments'])} 条，总耗时: {commitments_total_time:.3f}s（LLM: {commitments_llm_time:.3f}s, 解析: {commitments_parse_time:.3f}s）")
                    for item in results["commitments"]:
                        logger.bind(tag=TAG).info(f"  - Commitment: {item}")
                else:
                    logger.bind(tag=TAG).warning(f"Commitments响应中没有memories字段: {commitments_data}，耗时: {commitments_total_time:.3f}s")
            else:
                logger.bind(tag=TAG).error(f"Commitments提取结果解析失败，无法提取JSON，原始结果: {commitments_result[:500]}，耗时: {commitments_total_time:.3f}s")
        except Exception as e:
            import traceback
            logger.bind(tag=TAG).error(f"LLM提取记忆失败: {e}, 错误详情: {traceback.format_exc()}")
        finally:
            # 只有在实际处理了新消息时才更新计数（会话内计数，不持久化）
            if new_user_messages:
                self.last_extracted_user_message_count = len(user_messages)
                logger.bind(tag=TAG).debug(f"已更新提取位置：{self.last_extracted_user_message_count} 条用户消息（会话内计数）")
            else:
                logger.bind(tag=TAG).debug(f"没有新消息需要处理，保持提取位置不变：{self.last_extracted_user_message_count} 条")
        
        return results
    
    async def finalize_session(self, messages: List) -> Dict[str, any]:
        """会话关闭时调用：强制提取剩余未处理的消息，然后重置计数为0"""
        logger.bind(tag=TAG).info(f"会话关闭：开始最终提取，当前已处理 {self.last_extracted_user_message_count} 条用户消息")
        
        # 统计用户消息
        user_messages = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                if msg.role == 'user':
                    content = msg.content if msg.content is not None else ""
                    if content.strip():
                        user_messages.append(msg)
            elif isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content', '')
                if content and content.strip():
                    user_messages.append(msg)
        
        # 检查是否有未处理的消息
        remaining_messages = user_messages[self.last_extracted_user_message_count:]
        
        if remaining_messages:
            logger.bind(tag=TAG).info(f"会话关闭：发现 {len(remaining_messages)} 条未处理的用户消息，开始强制提取")
            # 强制提取剩余消息
            results = await self.extract_memories(messages)
            logger.bind(tag=TAG).info(f"会话关闭：最终提取完成，提取了 profile: {len(results.get('profile', []))}条, facts: {len(results.get('facts', []))}条, commitments: {len(results.get('commitments', []))}条")
        else:
            logger.bind(tag=TAG).info(f"会话关闭：没有未处理的消息（已处理 {self.last_extracted_user_message_count} 条，总共 {len(user_messages)} 条）")
            results = {"profile": [], "facts": [], "commitments": []}
        
        # 重置计数为0，为下次会话做准备
        self.last_extracted_user_message_count = 0
        logger.bind(tag=TAG).info(f"会话关闭：提取计数已重置为0，准备下次会话")
        
        return results
    
    async def save_extracted_memories(self, user_id: str, memories: Dict):
        """保存提取的记忆到存储层"""
        total_start = time.time()
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 保存profile（转换为字典格式）
        if memories.get("profile"):
            profile_save_start = time.time()
            profile_dict = {}
            for item in memories["profile"]:
                # 解析 "键: 值" 格式
                if ":" in item:
                    key, value = item.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # 根据键分类
                    if "姓名" in key or "昵称" in key:
                        if "基本信息" not in profile_dict:
                            profile_dict["基本信息"] = {}
                        profile_dict["基本信息"]["姓名"] = value
                    elif "年龄" in key:
                        if "基本信息" not in profile_dict:
                            profile_dict["基本信息"] = {}
                        profile_dict["基本信息"]["年龄"] = value
                    elif "职业" in key:
                        if "基本信息" not in profile_dict:
                            profile_dict["基本信息"] = {}
                        profile_dict["基本信息"]["职业"] = value
                    elif "位置" in key:
                        if "基本信息" not in profile_dict:
                            profile_dict["基本信息"] = {}
                        profile_dict["基本信息"]["位置"] = value
                    elif "喜好" in key:
                        if "基本信息" not in profile_dict:
                            profile_dict["基本信息"] = {}
                        profile_dict["基本信息"]["喜好"] = value
                    elif "价值观" in key:
                        if "价值观和边界" not in profile_dict:
                            profile_dict["价值观和边界"] = []
                        profile_dict["价值观和边界"].append(value)
                    elif "边界" in key:
                        if "价值观和边界" not in profile_dict:
                            profile_dict["价值观和边界"] = []
                        profile_dict["价值观和边界"].append(value)
                    elif "沟通偏好" in key or "偏好" in key:
                        if "沟通风格偏好" not in profile_dict:
                            profile_dict["沟通风格偏好"] = []
                        profile_dict["沟通风格偏好"].append(value)
                    else:
                        # 其他信息放入基本信息
                        if "基本信息" not in profile_dict:
                            profile_dict["基本信息"] = {}
                        profile_dict["基本信息"][key] = value
            
            if profile_dict:
                self.storage.save_profile(user_id, profile_dict)
                profile_save_time = time.time() - profile_save_start
                logger.bind(tag=TAG).info(f"Profile已保存，详细信息: {json.dumps(profile_dict, ensure_ascii=False, indent=2)}，耗时: {profile_save_time:.3f}s")
        
        # 保存facts和commitments到向量数据库（带去重逻辑）
        vector_save_start = time.time()
        saved_facts = 0
        saved_commitments = 0
        
        # 获取现有的向量元数据，用于去重
        metadata_load_start = time.time()
        existing_metadata = self.storage.get_vector_metadata()
        metadata_load_time = time.time() - metadata_load_start
        if metadata_load_time > 0.001:
            logger.bind(tag=TAG).debug(f"加载向量元数据耗时: {metadata_load_time:.3f}s")
        
        existing_texts = set()
        existing_facts_texts = set()  # 用于跨类型去重
        existing_commitments_texts = set()  # 用于跨类型去重
        dedup_start = time.time()
        for meta in existing_metadata:
            text = meta.get("text", "").strip()
            existing_texts.add(text)
            memory_type_existing = meta.get("memory_type", "")
            if memory_type_existing == "facts":
                existing_facts_texts.add(text)
            elif memory_type_existing == "commitments":
                existing_commitments_texts.add(text)
        dedup_time = time.time() - dedup_start
        if dedup_time > 0.001:
            logger.bind(tag=TAG).debug(f"构建去重索引耗时: {dedup_time:.3f}s")
        
        for memory_type in ["facts", "commitments"]:
            memory_type_start = time.time()
            memory_list = memories.get(memory_type, [])
            if not memory_list:
                logger.bind(tag=TAG).info(f"没有{memory_type}需要保存（LLM返回空列表）")
                continue
            
            # 去重：对于facts和commitments，提取关键信息进行去重
            seen_keys = set()  # 用于同一次提取中的去重
            dedup_check_time = 0
            embedding_time = 0
            save_time = 0
            for memory_text in memory_list:
                if not memory_text or not memory_text.strip():
                    logger.bind(tag=TAG).debug(f"跳过空的{memory_type}条目")
                    continue
                
                # 检查是否已存在（完全匹配）
                if memory_text.strip() in existing_texts:
                    logger.bind(tag=TAG).debug(f"跳过重复的{memory_type}（完全匹配）: {memory_text[:50]}")
                    continue
                
                # 提取关键信息进行智能去重（避免同一事件的不同表述被重复保存）
                # 格式可能是："2025-12-20: 用户去爬山" 或 "2025-12-20: 用户今天去爬山了"
                core_text = memory_text
                # 移除日期前缀（如果有）
                date_match = re.match(r'^\d{4}-\d{2}-\d{2}:\s*', core_text)
                if date_match:
                    core_text = core_text[date_match.end():]
                
                # 提取核心关键词（移除"用户"、"今天"、"了"等常见词）
                core_text = core_text.replace("用户", "").replace("今天", "").replace("了", "").strip()
                # 移除多余的标点和空格
                core_text = re.sub(r'\s+', ' ', core_text).strip()
                # 提取主要动作和对象（前40个字符作为去重key，足够覆盖大部分情况）
                key = core_text[:40].strip()
                
                # 检查现有数据中是否有相似的（通过比较key）
                # 1. 先检查同类型数据
                dedup_check_start = time.time()
                is_duplicate = False
                same_type_texts = existing_facts_texts if memory_type == "facts" else existing_commitments_texts
                for existing_text in same_type_texts:
                    existing_core = existing_text
                    # 移除日期前缀
                    existing_date_match = re.match(r'^\d{4}-\d{2}-\d{2}:\s*', existing_core)
                    if existing_date_match:
                        existing_core = existing_core[existing_date_match.end():]
                    # 提取核心关键词
                    existing_core = existing_core.replace("用户", "").replace("今天", "").replace("了", "").replace("将", "").replace("要", "").strip()
                    existing_core = re.sub(r'\s+', ' ', existing_core).strip()
                    existing_key = existing_core[:40].strip()
                    
                    # 如果key相似（包含关系或高度重叠），认为是重复
                    if key and existing_key:
                        # 检查是否包含关系（一个包含另一个的核心部分）
                        if key in existing_key or existing_key in key:
                            is_duplicate = True
                            logger.bind(tag=TAG).debug(f"跳过重复的{memory_type}（语义相似）: {memory_text[:50]} (与 {existing_text[:50]} 相似)")
                            break
                        # 检查重叠度（如果前20个字符相同，认为是重复）
                        min_len = min(len(key), len(existing_key))
                        if min_len >= 10:  # 至少10个字符才比较
                            overlap = sum(1 for i in range(min_len) if key[i] == existing_key[i])
                            if overlap >= min_len * 0.7:  # 70%重叠认为是重复
                                is_duplicate = True
                                logger.bind(tag=TAG).debug(f"跳过重复的{memory_type}（高度重叠）: {memory_text[:50]} (与 {existing_text[:50]} 相似)")
                                break
                
                # 2. 检查跨类型去重（避免同一事件被提取为不同类型）
                if not is_duplicate:
                    other_type_texts = existing_commitments_texts if memory_type == "facts" else existing_facts_texts
                    for existing_text in other_type_texts:
                        existing_core = existing_text
                        # 移除日期前缀
                        existing_date_match = re.match(r'^\d{4}-\d{2}-\d{2}:\s*', existing_core)
                        if existing_date_match:
                            existing_core = existing_core[existing_date_match.end():]
                        # 提取核心关键词（移除"用户"、"今天"、"了"、"将"、"要"等）
                        existing_core = existing_core.replace("用户", "").replace("今天", "").replace("了", "").replace("将", "").replace("要", "").replace("计划", "").strip()
                        existing_core = re.sub(r'\s+', ' ', existing_core).strip()
                        existing_key = existing_core[:40].strip()
                        
                        # 如果key相似，认为是同一事件，跳过保存
                        if key and existing_key:
                            # 检查是否包含关系（一个包含另一个的核心部分）
                            if key in existing_key or existing_key in key:
                                is_duplicate = True
                                logger.bind(tag=TAG).warning(f"跳过重复的{memory_type}（跨类型重复）: {memory_text[:50]} (与 {existing_text[:50]} 相似，已在另一种类型中保存)")
                                break
                            # 检查重叠度
                            min_len = min(len(key), len(existing_key))
                            if min_len >= 10:
                                overlap = sum(1 for i in range(min_len) if key[i] == existing_key[i])
                                if overlap >= min_len * 0.7:
                                    is_duplicate = True
                                    logger.bind(tag=TAG).warning(f"跳过重复的{memory_type}（跨类型重复）: {memory_text[:50]} (与 {existing_text[:50]} 相似，已在另一种类型中保存)")
                                    break
                
                if is_duplicate:
                    continue
                
                dedup_check_time += time.time() - dedup_check_start
                
                # 检查同一次提取中是否重复
                if key in seen_keys:
                    logger.bind(tag=TAG).debug(f"跳过重复的{memory_type}（同次提取）: {key}")
                    continue
                seen_keys.add(key)
                
                # 替换LLM返回的占位符（如果LLM直接返回了{{current_date}}）
                memory_text_processed = memory_text.replace("{{current_date}}", current_date).replace("{current_date}", current_date)
                
                # 【新增】验证并纠正日期，清理相对时间词
                from core.utils.current_time import get_current_weekday
                from .memory_date_validator import validate_and_correct_date
                current_weekday = get_current_weekday()
                corrected_text, was_processed = validate_and_correct_date(
                    memory_text_processed,
                    current_date,
                    current_weekday,
                    self.llm  # 传入LLM客户端用于清理相对时间词
                )
                
                if was_processed:
                    memory_text_processed = corrected_text
                    logger.bind(tag=TAG).info(f"{memory_type}日期已验证并处理: {memory_text[:50]} → {corrected_text[:50]}")
                
                save_item_start = time.time()
                if self.storage.save_vector_memory(memory_text_processed, memory_type, user_id, current_date):
                    save_time += time.time() - save_item_start
                    if memory_type == "facts":
                        saved_facts += 1
                    else:
                        saved_commitments += 1
                    logger.bind(tag=TAG).info(f"{memory_type}已保存: {memory_text[:50]}")
                    # 添加到已存在集合，避免同一次提取中重复
                    existing_texts.add(memory_text.strip())
                else:
                    save_time += time.time() - save_item_start
            
            memory_type_time = time.time() - memory_type_start
            if memory_type_time > 0.001:
                logger.bind(tag=TAG).info(f"{memory_type}保存耗时: {memory_type_time:.3f}s（去重检查: {dedup_check_time:.3f}s, 保存: {save_time:.3f}s）")
        
        if saved_facts > 0:
            logger.bind(tag=TAG).info(f"Facts保存完成，共保存 {saved_facts} 条")
        if saved_commitments > 0:
            logger.bind(tag=TAG).info(f"Commitments保存完成，共保存 {saved_commitments} 条")
        
        # 保存向量索引
        index_save_start = time.time()
        self.storage.save_vector_index()
        index_save_time = time.time() - index_save_start
        vector_save_time = time.time() - vector_save_start
        total_time = time.time() - total_start
        logger.bind(tag=TAG).info(f"向量索引保存耗时: {index_save_time:.3f}s，向量记忆保存总耗时: {vector_save_time:.3f}s，记忆保存总耗时: {total_time:.3f}s")

