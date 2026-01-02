"""记忆管理层：负责提取记忆、协调存储和检索"""
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Optional

from ..base import logger
from .memory_storage import MemoryStorage

TAG = __name__


PROFILE_EXTRACTION_PROMPT = """你是一个专业的语言学专家，擅长从对话中精确提取稳定的用户画像信息。

任务：
**重要：只从用户（user）的消息中提取信息，不要从assistant的回复中提取**
仅当用户明确提到时，提取以下稳定的用户画像信息：
- 姓名/昵称：用户的姓名或偏好昵称
- 年龄：用户的年龄（提取数字，如"24"、"我24岁"、"24岁"）
- 职业：工作、职业或学生身份
- 位置：城市、国家或地区
- 喜好：用户的喜好、偏好（如：喜欢的颜色、食物、活动等，例如："喜欢蓝色"、"喜欢看电影"）
- 价值观：用户重视什么（例如："重视和平与安静"）
- 边界：需要避免的话题或语气（例如："不喜欢被催促"）
- 沟通偏好：沟通风格偏好（例如："偏好简短消息"）

输出格式（严格JSON格式）：
{{"memories": [
  "姓名/昵称: 小明",
  "年龄: 22",
  "职业: 学生",
  "位置: 北京",
  "喜好: 蓝色",
  "价值观: 重视学习",
  "边界: 不喜欢被打扰",
  "沟通偏好: 偏好简短回复"
]}}

规则：
- 如果没有找到相关信息，返回：{{"memories": []}}
- 最多10项；简洁精确
- 仅提取用户明确陈述的信息
- 不要推断或推测超出明确陈述的内容
- 不要提取临时状态（例如："今天很累"）
- 不要提取事件或行动（例如："去了健身房"）
- 不要提取计划、任务或承诺（这些属于commitments）
- 不要提取事实事件（这些属于facts）
- 重要：必须返回有效的JSON格式

示例：

输入：user: 我叫小明，今年22岁
输出：{{"memories": ["姓名/昵称: 小明", "年龄: 22"]}}

输入：user: 我是学生，在北京上学
输出：{{"memories": ["职业: 学生", "位置: 北京"]}}

输入：user: 我今天很累
输出：{{"memories": []}}  # 临时状态，不是profile

输入：user: 我喜欢蓝色
输出：{{"memories": ["喜好: 蓝色"]}}

输入：user: 我下周六有英语考试
输出：{{"memories": []}}  # 这是计划/任务，不是profile

对话内容：
{dialogue_text}

当前日期：{current_date}

请从以上对话中提取用户画像信息，严格按照输出格式返回JSON。"""


FACTS_EXTRACTION_PROMPT = """你是一个专业的记忆管理员。**只提取已经发生过了的事件和事实（是过去发生完了的事件）**。

任务：
**重要：只从用户（user）的消息中提取信息，不要从assistant的回复中提取**
**只提取已经发生的事件，不提取未来的计划、不提取用户画像信息**

提取以下类型的事件和事实（必须是已经发生的，是过去发生完了的事件）：
- 用户已经参加过了的活动、会议、聚会等（如：参加了党会、参加了年会、参加了婚礼等）
- 用户已经完成过了的任务或工作（如：完成了项目、参加了考试、完成了作业等）
- 用户已经去过的地方（如：去爬山了、去旅游了、去购物了等）
- 用户已经经历过了的重要生活事件（成就、挑战、里程碑）
- 用户已经经历过了的情感时刻（开心、难过、沮丧、兴奋）
- 用户已经发生过了的重要人际关系或互动

输出格式（严格JSON格式）：
{{"memories": [
  "{{current_date}}: 用户参加了音乐会，玩得很开心",
  "用户与朋友发生了争吵，朋友取消了计划"
]}}

规则：
- 当提到时间时，使用ISO日期格式（YYYY-MM-DD）
- **重要：将相对时间转换为实际日期**
  - 当前日期是：{{current_date}}
  - '今天' = {{current_date}}
  - '昨天' = {{current_date}}的前一天
  - '明天' = {{current_date}}的后一天
  - '这周四' = 本周的周四（如果今天是{{current_date}}，计算本周四的日期）
  - '下周四' = 下周的周四
  - '这周六' = 本周的周六
  - '下周六' = 下周的周六（如果今天是{{current_date}}，计算下周六的日期）
  - 例如：如果今天是2025-12-23（周一），那么"这周四"是2025-12-25，"下周六"是2026-1-3
- 每条记忆最多80个字符
- 专注于用户相关信息（只从用户消息中提取）
- 在相关时包含情感上下文
- 如果没有找到已发生的事件，返回：{{"memories": []}}
- **严格禁止：不要提取用户画像信息（姓名、年龄、职业、喜好、性格等，这些属于profile）**
  - 例如："我喜欢蓝色"、"我喜欢读科幻小说"、"我是学生"等都属于profile，不是facts
  - 例如："用户喜欢蓝色"、"用户喜欢看电影"等都属于profile，不是facts
- **严格禁止：不要提取计划、任务或承诺（这些属于commitments）**
  - facts是已经发生的事件，commitments是未来的计划
  - 如果用户说"我下周六有考试"、"我这周四有个年会要去"、"我明天要开会"，这应该提取为commitments，不是facts
  - **特别注意：包含"将"、"要"、"计划"、"打算"、"准备"等表示未来的词汇的事件，都属于commitments，不是facts**
  - 例如："用户将参加年会"、"用户要参加年会"、"用户计划参加年会"、"用户准备参加年会"都是commitments，不是facts
  - 例如："2025-12-25: 用户要去参加年会"是commitments（未来事件），不是facts
  - 只有已经完成的事件才是facts，例如："2025-12-24: 用户参加了年会"是facts（已发生）
- **严格禁止：不要提取测试性问题或对记忆系统的提问**
  - 例如："我今天干嘛了"、"我上周干嘛了"、"我之前说过什么"、"你还记得吗"等
  - 这些是用户测试记忆系统的问题，不是事实事件
  - 不要将"用户提出了疑问"、"用户测试记忆"等作为facts提取
- **严格禁止：不要提取无意义的临时性社交行为**
  - 例如："用户道别"、"用户说再见"、"用户打招呼"、"用户问候"、"用户说你好"等
  - 这些是临时的社交行为，没有实际内容价值，不值得作为记忆保存
  - 只有有实际意义的事件才应该被提取（如：参加了活动、完成了任务、去了某个地方等）
- 重要：必须返回有效的JSON格式

示例：

输入：user: 昨晚的音乐会太棒了！
输出：{{"memories": ["{{current_date}}: 用户参加了音乐会，玩得很开心"]}}

输入：user: 我喜欢读科幻小说
输出：{{"memories": []}}  # 这是喜好，属于profile，不是facts

输入：user: 我喜欢蓝色
输出：{{"memories": []}}  # 这是喜好，属于profile，不是facts

输入：user: 我今天参加了一个党会
输出：{{"memories": ["{{current_date}}: 用户参加了党会"]}}  # 已经发生的事件

输入：user: 我昨天去爬山了
输出：{{"memories": ["{{current_date}}: 用户去爬山了"]}}  # 已经发生的事件

输入：user: 我这周四有个年会要去
输出：{{"memories": []}}  # 这是未来的计划，属于commitments，不是facts

输入：user: 我下周六有考试
输出：{{"memories": []}}  # 这是未来的计划，属于commitments，不是facts

输入：user: 我明天要参加年会
输出：{{"memories": []}}  # 这是未来的计划，属于commitments，不是facts

输入：user: 2025-12-25: 用户要去参加年会并表演跳舞
输出：{{"memories": []}}  # 这是未来的计划（包含"要"），属于commitments，不是facts

输入：user: 今天怎么样？
输出：{{"memories": []}}  # 普通问候，不是事实事件

输入：user: 我今天干嘛了？
输出：{{"memories": []}}  # 这是测试性问题，不是事实事件

输入：user: 我上周干嘛了？你还记得吗？
输出：{{"memories": []}}  # 这是测试性问题，不是事实事件

输入：user: 我之前跟你说过什么？
输出：{{"memories": []}}  # 这是测试性问题，不是事实事件

输入：user: 再见
输出：{{"memories": []}}  # 这是道别，无意义的临时社交行为，不是事实事件

输入：user: 用户道别
输出：{{"memories": []}}  # 这是道别，无意义的临时社交行为，不是事实事件

输入：user: 你好
输出：{{"memories": []}}  # 这是问候，无意义的临时社交行为，不是事实事件

对话内容：
{dialogue_text}

当前日期：{current_date}

请从以上对话中提取已经发生过了的事件和事实（是过去发生完了的事件），严格按照输出格式返回JSON。"""


COMMITMENTS_EXTRACTION_PROMPT = """提取用户提到或同意的承诺、任务或计划（是未来的计划）。

任务：
**重要：只从用户（user）的消息中提取信息，不要从assistant的回复中提取**
**只提取未来的计划，不提取已经发生过了的事件**
识别用户何时：
- 做出承诺要做某事
- 同意一个计划或任务
- 设定目标或意图
- 提到待办事项
- **提到未来要参加的活动、会议、聚会等（如：参加年会、参加婚礼、参加考试等）**
- **提到未来要完成的任务或工作（如：要完成项目、要准备考试等）**

输出格式（严格JSON格式，与facts格式一致）：
{{"memories": [
  "{{current_date}}: 用户明天要给妈妈打电话",
  "用户下周六有英语考试",
  "用户计划这个周末整理房间"
]}}

规则：
- 当提到时间时，使用ISO日期格式（YYYY-MM-DD）
- **重要：将相对时间转换为实际日期**
  - 当前日期是：{{current_date}}
  - '今天' = {{current_date}}
  - '明天' = {{current_date}}的后一天
  - '这周四' = 本周的周四（如果今天是{{current_date}}，计算本周四的日期）
  - '下周四' = 下周的周四
  - '这周六' = 本周的周六
  - '下周六' = 下周的周六（如果今天是{{current_date}}，计算下周六的日期）
  - 例如：如果今天是2025-12-23（周一），那么"这周四"是2025-12-25，"下周六"是2025-12-27
- 每条记忆最多80个字符，简洁明了
- **重要：不要提取已经发生的事件（这些属于facts）**
  - 例如："我昨天去爬山了"是facts，不是commitments
  - commitments是未来的计划，facts是过去的事件
- 如果没有找到承诺，返回：{{"memories": []}}
- 重要：必须返回有效的JSON格式

示例：

输入：user: 我明天需要给妈妈打电话
输出：{{"memories": ["{{current_date}}: 用户明天要给妈妈打电话"]}}

输入：user: 我计划这个周末整理房间
输出：{{"memories": ["用户计划这个周末整理房间"]}}

输入：user: 我应该开始锻炼了
输出：{{"memories": []}}  # 只是想法，不是明确的承诺

输入：user: 我下周六有英语考试
输出：{{"memories": ["用户下周六有英语考试"]}}

输入：user: 我这周四有个年会要参加
输出：{{"memories": ["2025-12-25: 用户这周四要参加年会"]}}  # 如果今天是2025-12-23（周一），这周四是2025-12-25

输入：user: 我下周六要去医院复查
输出：{{"memories": ["2025-12-27: 用户下周六要去医院复查"]}}  # 如果今天是2025-12-23（周一），下周六是2025-12-27

输入：user: 我明天要开会
输出：{{"memories": ["2025-12-24: 用户明天要开会"]}}  # 如果今天是2025-12-23，明天是2025-12-24

输入：user: 我昨天去爬山了
输出：{{"memories": []}}  # 这是已发生的事件，属于facts，不是commitments

对话内容：
{dialogue_text}

当前日期：{current_date}

请从以上对话中提取承诺、任务或计划，严格按照输出格式返回JSON。"""


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
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
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
                current_date=current_date
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
                current_date=current_date
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

