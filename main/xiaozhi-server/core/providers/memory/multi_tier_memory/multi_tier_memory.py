"""多层级记忆系统主类：整合存储层、检索层和管理层"""
import os
import json
import threading
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque
import time

from ..base import MemoryProviderBase, logger
from config.config_loader import get_project_dir
from .memory_storage import MemoryStorage
from .memory_retriever import MemoryRetriever
from .memory_manager import MemoryManager

TAG = __name__


class MemoryProvider(MemoryProviderBase):
    """多层级记忆系统主类"""
    
    def __init__(self, config, summary_memory=None):
        super().__init__(config)
        self.role_id = None
        self.llm = None
        
        # 配置参数
        self.storage_frequency = config.get("storage_frequency", 4)  # 每N轮存储一次
        self.working_memory_limit = config.get("working_memory_limit", 5)  # 工作记忆保留轮数
        self.vector_top_k = config.get("vector_top_k", 3)  # 向量检索top-K
        
        # Embedding配置
        self.embedder_config = config  # 传递整个config，storage会从中读取embedder配置
        
        # 数据目录
        project_dir = get_project_dir().rstrip('/').rstrip('\\')
        self.data_dir = os.path.join(project_dir, "data", "multi_tier_memory")
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            logger.bind(tag=TAG).debug(f"数据目录: {self.data_dir}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"创建数据目录失败: {self.data_dir}, 错误: {e}")
            raise
        
        # 存储层、检索层、管理层（将在init_memory时初始化）
        self.storage = None
        self.retriever = None
        self.manager = None
        
        # 工作记忆（内存）
        self.working_memory = deque(maxlen=self.working_memory_limit * 2)
        
        # 对话计数器
        self.dialogue_count = 0
        
        # Profile缓存
        self.profile_cache = None
        self.profile_cache_time = None
        
        logger.bind(tag=TAG).info("多层级记忆系统初始化完成")
    
    def init_memory(self, role_id, llm, **kwargs):
        """初始化记忆系统"""
        super().init_memory(role_id, llm, **kwargs)
        
        # 清理 role_id，移除 Windows 文件名不允许的字符
        safe_role_id = role_id.replace(":", "_").replace("/", "_").replace("\\", "_")
        safe_role_id = safe_role_id.replace("<", "_").replace(">", "_").replace('"', "_")
        safe_role_id = safe_role_id.replace("|", "_").replace("?", "_").replace("*", "_")
        
        # 数据库和向量索引路径
        db_path = os.path.join(self.data_dir, f"memory_{safe_role_id}.db")
        vector_index_path = os.path.join(self.data_dir, f"vector_index_{safe_role_id}.faiss")
        vector_metadata_path = os.path.join(self.data_dir, f"vector_metadata_{safe_role_id}.json")
        
        # 从配置中获取embedding维度（如果配置了embedder）
        embedding_dim = 384  # 默认维度
        embedder = self.embedder_config.get("embedder", {})
        provider = embedder.get("provider", "hash")
        if provider == "openai":
            embedding_dim = embedder.get("config", {}).get("embedding_dims", 2560)
        elif provider == "sentence_transformers":
            embedding_dim = embedder.get("config", {}).get("embedding_dims", 512)  # BGE-small-zh默认512维
        
        # 初始化存储层（传递embedder配置）
        self.storage = MemoryStorage(
            db_path, 
            vector_index_path, 
            vector_metadata_path,
            embedding_dim=embedding_dim,
            embedder_config=self.embedder_config
        )
        
        # 初始化检索层
        self.retriever = MemoryRetriever(self.storage, self.vector_top_k)
        
        # 初始化管理层（传递user_id用于持久化提取状态）
        self.manager = MemoryManager(self.storage, llm, user_id=role_id)
        
        # 新会话开始：重置提取计数为0
        self.manager.reset_session()
        
        # 加载工作记忆
        self._load_working_memory()
        
        # 预加载profile到缓存（系统启动时立即加载）
        self._preload_profile()
        
        logger.bind(tag=TAG).info(f"记忆系统已为用户 {role_id} 初始化（新会话，计数从0开始）")
    
    def _load_working_memory(self):
        """从数据库加载工作记忆到内存"""
        if not self.role_id:
            return
        
        working_memory = self.storage.load_working_memory(self.role_id, self.working_memory_limit)
        self.working_memory.clear()
        for msg in working_memory:
            self.working_memory.append(msg)
        
        logger.bind(tag=TAG).debug(f"加载了 {len(self.working_memory)} 条工作记忆到内存")
    
    def _preload_profile(self):
        """系统启动时预加载profile到缓存"""
        if not self.role_id:
            logger.bind(tag=TAG).debug("role_id为空，跳过profile预加载")
            return
        
        if self.retriever is None:
            logger.bind(tag=TAG).debug("retriever未初始化，跳过profile预加载")
            return
        
        try:
            logger.bind(tag=TAG).info(f"系统启动时预加载profile，role_id: {self.role_id}")
            profile_start = time.time()
            profile = self.retriever.retrieve_long_term_memory(self.role_id)
            profile_time = time.time() - profile_start
            
            if profile:
                self.profile_cache = profile
                self.profile_cache_time = time.time()
                logger.bind(tag=TAG).info(f"Profile预加载成功: {json.dumps(profile, ensure_ascii=False)}, 耗时: {profile_time:.3f}s")
            else:
                logger.bind(tag=TAG).info(f"Profile预加载完成，但未找到用户画像数据，耗时: {profile_time:.3f}s")
        except Exception as e:
            logger.bind(tag=TAG).warning(f"Profile预加载失败: {e}，将在首次查询时加载")
    
    async def save_memory(self, msgs: List):
        """保存记忆（固定频率触发）"""
        if not self.role_id or not self.llm:
            return None
        
        # 检查是否已初始化
        if self.storage is None:
            logger.bind(tag=TAG).warning("记忆系统尚未初始化（storage为None），跳过保存记忆")
            return None
        
        # 增加对话计数
        self.dialogue_count += 1
        
        # 保存工作记忆（每次对话都保存）
        save_start = time.time()
        self.storage.save_working_memory(self.role_id, msgs, self.working_memory_limit)
        save_time = time.time() - save_start
        logger.bind(tag=TAG).info(f"工作记忆保存耗时: {save_time:.3f}s")
        
        # 更新内存中的工作记忆
        self.working_memory.clear()
        recent_messages = msgs[-(self.working_memory_limit * 2):] if len(msgs) > (self.working_memory_limit * 2) else msgs
        for msg in recent_messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                if msg.role in ['user', 'assistant']:
                    content = msg.content if msg.content is not None else ""
                    if content.strip():
                        self.working_memory.append({"role": msg.role, "content": content})
        
        # 固定频率存储（每N轮）
        if self.dialogue_count % self.storage_frequency == 0:
            # 异步执行记忆提取和存储
            def extract_and_save():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._extract_and_save_memories(msgs))
                    loop.close()
                except Exception as e:
                    logger.bind(tag=TAG).error(f"异步存储记忆失败: {e}")
            
            threading.Thread(target=extract_and_save, daemon=True).start()
        
        return None
    
    async def _extract_and_save_memories(self, msgs: List):
        """提取并保存记忆"""
        total_start = time.time()
        # 检查是否已初始化
        if self.manager is None:
            logger.bind(tag=TAG).warning("记忆系统尚未初始化（manager为None），跳过提取记忆")
            return
        
        logger.bind(tag=TAG).info("开始提取记忆（profile、facts、commitments）")
        
        # 使用管理层提取记忆
        extract_start = time.time()
        memories = await self.manager.extract_memories(msgs)
        extract_time = time.time() - extract_start
        logger.bind(tag=TAG).info(f"记忆提取完成 - profile: {len(memories.get('profile', []))}条, facts: {len(memories.get('facts', []))}条, commitments: {len(memories.get('commitments', []))}条，提取耗时: {extract_time:.3f}s")
        
        # 使用管理层保存记忆
        save_start = time.time()
        await self.manager.save_extracted_memories(self.role_id, memories)
        save_time = time.time() - save_start
        total_time = time.time() - total_start
        logger.bind(tag=TAG).info(f"记忆保存耗时: {save_time:.3f}s，总耗时: {total_time:.3f}s")
        
        # 清除profile缓存
        self.profile_cache = None
        self.profile_cache_time = None
    
    async def finalize_session(self, msgs: List):
        """会话关闭时调用：强制提取剩余未处理的消息，然后重置计数为0"""
        if not self.role_id or not self.llm:
            logger.bind(tag=TAG).debug("会话关闭：role_id或llm为空，跳过最终提取")
            return
        
        # 检查是否已初始化
        if self.manager is None:
            logger.bind(tag=TAG).warning("会话关闭：记忆系统尚未初始化（manager为None），跳过最终提取")
            return
        
        logger.bind(tag=TAG).info("会话关闭：开始最终提取剩余未处理的消息")
        
        try:
            # 使用管理层进行最终提取
            finalize_start = time.time()
            memories = await self.manager.finalize_session(msgs)
            finalize_time = time.time() - finalize_start
            
            # 如果有提取到记忆，保存它们
            if memories and (memories.get("profile") or memories.get("facts") or memories.get("commitments")):
                logger.bind(tag=TAG).info(f"会话关闭：最终提取到记忆，开始保存 - profile: {len(memories.get('profile', []))}条, facts: {len(memories.get('facts', []))}条, commitments: {len(memories.get('commitments', []))}条")
                save_start = time.time()
                await self.manager.save_extracted_memories(self.role_id, memories)
                save_time = time.time() - save_start
                logger.bind(tag=TAG).info(f"会话关闭：最终提取的记忆已保存，保存耗时: {save_time:.3f}s，总耗时: {finalize_time:.3f}s")
            else:
                logger.bind(tag=TAG).info(f"会话关闭：最终提取未发现新记忆，总耗时: {finalize_time:.3f}s")
        except Exception as e:
            import traceback
            logger.bind(tag=TAG).error(f"会话关闭：最终提取失败: {e}, 错误详情: {traceback.format_exc()}")
    
    async def query_memory(self, query: str, dialogue_history: List = None) -> Dict[str, str]:
        """检索记忆，返回用户信息和记忆信息（分离）"""
        if not self.role_id:
            logger.bind(tag=TAG).debug("role_id为空，跳过记忆检索")
            return {"user_info": "", "memory_info": ""}
        
        # 检查是否已初始化
        if self.retriever is None:
            logger.bind(tag=TAG).warning("记忆系统尚未初始化（retriever为None），跳过记忆检索")
            return {"user_info": "", "memory_info": ""}
        
        total_start_time = time.time()
        logger.bind(tag=TAG).info(f"开始检索记忆，查询: {query[:50]}")
        memory_parts = []
        user_info_parts = []
        
        # 1. 工作记忆（从内存获取，但排除当前对话历史中已有的消息）
        working_mem_start = time.time()
        if self.working_memory:
            current_dialogue_contents = set()
            if dialogue_history:
                for msg in dialogue_history:
                    if hasattr(msg, 'content'):
                        current_dialogue_contents.add(msg.content)
            
            working_mem_messages = []
            for msg in list(self.working_memory)[-self.working_memory_limit * 2:]:
                if msg['content'] not in current_dialogue_contents:
                    working_mem_messages.append(f"{msg['role']}: {msg['content']}")
            
            if working_mem_messages:
                working_mem = "\n".join(working_mem_messages)
                memory_parts.append(f"[工作记忆（跨会话）]\n{working_mem}")
                working_mem_time = time.time() - working_mem_start
                logger.bind(tag=TAG).info(f"检索到工作记忆: {len(working_mem_messages)} 条，耗时: {working_mem_time:.3f}s")
        else:
            working_mem_time = time.time() - working_mem_start
        
        # 2. 短期记忆（向量检索）
        short_term_start = time.time()
        short_term = self.retriever.retrieve_short_term_memory(query)
        short_term_time = time.time() - short_term_start
        logger.bind(tag=TAG).info(f"短期记忆检索耗时: {short_term_time:.3f}s")
        if short_term["facts"]:
            memory_parts.append(f"[事实记忆]\n" + "\n".join(f"- {f}" for f in short_term["facts"]))
            logger.bind(tag=TAG).info(f"检索到事实记忆: {len(short_term['facts'])} 条")
            for fact in short_term["facts"]:
                logger.bind(tag=TAG).info(f"  - 事实: {fact[:50]}")
        if short_term["commitments"]:
            memory_parts.append(f"[承诺/计划]\n" + "\n".join(f"- {c}" for c in short_term["commitments"]))
            logger.bind(tag=TAG).info(f"检索到承诺/计划: {len(short_term['commitments'])} 条")
            for commitment in short_term["commitments"]:
                logger.bind(tag=TAG).info(f"  - 承诺: {commitment[:50]}")
        
        # 3. 长期记忆（profile，使用缓存）- 单独放在user_info中
        profile_start = time.time()
        profile = self._get_profile()
        profile_time = time.time() - profile_start
        logger.bind(tag=TAG).info(f"用户画像检索耗时: {profile_time:.3f}s")
        if profile:
            # 格式化profile，使其更清晰易懂，便于LLM识别
            profile_parts = []
            
            # 提取基本信息，使用更自然的语言格式
            basic_info = profile.get("基本信息", {})
            name = basic_info.get("姓名", "").strip() if basic_info.get("姓名") else ""
            age = basic_info.get("年龄", "").strip() if basic_info.get("年龄") else ""
            job = basic_info.get("职业", "").strip() if basic_info.get("职业") else ""
            location = basic_info.get("位置", "").strip() if basic_info.get("位置") else ""
            preference = basic_info.get("喜好", "").strip() if basic_info.get("喜好") else ""
            
            # 构建基本信息字符串，使用最直接的方式，强调这是用户的信息
            if name:
                profile_parts.append(f"**用户的名字是：{name}**（当用户问'我叫什么名字'时，回答这个名字）")
            if age:
                profile_parts.append(f"用户的年龄是：{age}岁（当用户问'我的年龄'或'我多大了'时，回答这个年龄）")
            if job:
                profile_parts.append(f"用户的职业是：{job}（当用户问'我的职业'或'我是做什么的'时，回答这个职业）")
            if location:
                profile_parts.append(f"用户的位置是：{location}")
            if preference:
                profile_parts.append(f"用户的喜好是：{preference}")
            
            # 提取其他信息（价值观、边界、沟通偏好等）
            other_info_parts = []
            if "价值观和边界" in profile:
                values = profile["价值观和边界"]
                if isinstance(values, list) and values:
                    other_info_parts.append(f"价值观和边界：{', '.join(values)}")
            
            if "沟通风格偏好" in profile:
                comm_prefs = profile["沟通风格偏好"]
                if isinstance(comm_prefs, list) and comm_prefs:
                    other_info_parts.append(f"沟通偏好：{', '.join(comm_prefs)}")
            
            # 处理其他未分类的信息
            for key, value in profile.items():
                if key not in ["基本信息", "价值观和边界", "沟通风格偏好"]:
                    if isinstance(value, list) and value:
                        other_info_parts.append(f"{key}：{', '.join(str(v) for v in value)}")
                    elif isinstance(value, dict) and value:
                        for sub_key, sub_value in value.items():
                            if sub_value:
                                other_info_parts.append(f"{key}.{sub_key}：{sub_value}")
                    elif value:
                        other_info_parts.append(f"{key}：{value}")
            
            if other_info_parts:
                profile_parts.extend(other_info_parts)
            
            if profile_parts:
                profile_str = "\n".join(profile_parts)
                user_info_parts.append(profile_str)
            logger.bind(tag=TAG).info("检索到用户画像")
        
        user_info = "\n".join(user_info_parts) if user_info_parts else ""
        memory_info = "\n\n".join(memory_parts) if memory_parts else ""
        
        total_time = time.time() - total_start_time
        if user_info or memory_info:
            logger.bind(tag=TAG).info(f"记忆检索完成，用户信息长度: {len(user_info)} 字符，记忆信息长度: {len(memory_info)} 字符，总耗时: {total_time:.3f}s")
            if user_info:
                logger.bind(tag=TAG).debug(f"用户信息内容: {user_info[:300]}")
            if memory_info:
                logger.bind(tag=TAG).debug(f"记忆信息内容: {memory_info[:300]}")
        else:
            logger.bind(tag=TAG).info(f"未检索到任何记忆，总耗时: {total_time:.3f}s")
        
        return {"user_info": user_info, "memory_info": memory_info}
    
    def _get_profile(self) -> Optional[Dict]:
        """获取用户画像（带缓存）"""
        # 检查缓存（1小时有效期）
        if self.profile_cache and self.profile_cache_time:
            if (time.time() - self.profile_cache_time) < 3600:
                logger.bind(tag=TAG).debug("使用缓存的profile")
                return self.profile_cache
        
        if not self.role_id:
            logger.bind(tag=TAG).warning("role_id为空，无法检索profile")
            return None
        
        # 检查是否已初始化
        if self.retriever is None:
            logger.bind(tag=TAG).warning("记忆系统尚未初始化（retriever为None），无法检索profile")
            return None
        
        logger.bind(tag=TAG).info(f"检索profile，role_id: {self.role_id}")
        profile_start = time.time()
        profile = self.retriever.retrieve_long_term_memory(self.role_id)
        profile_time = time.time() - profile_start
        if profile:
            self.profile_cache = profile
            self.profile_cache_time = time.time()
            logger.bind(tag=TAG).info(f"Profile检索成功: {json.dumps(profile, ensure_ascii=False)}, 耗时: {profile_time:.3f}s")
        else:
            logger.bind(tag=TAG).warning(f"Profile检索失败，role_id: {self.role_id}, 耗时: {profile_time:.3f}s")
        
        return profile

