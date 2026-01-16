"""记忆去重器：基于语义相似度和时间窗口的去重机制"""
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..base import logger
from .memory_storage import MemoryStorage

TAG = __name__


class MemoryDeduplicator:
    """记忆去重器：基于语义相似度和时间窗口（同一天）"""
    
    def __init__(self, storage: MemoryStorage, similarity_threshold: float = 0.87):
        """
        初始化去重器
        
        Args:
            storage: MemoryStorage实例，用于获取embedding
            similarity_threshold: 语义相似度阈值（0-1），默认0.87
                                 对于bge-small-zh-v1.5模型，建议0.87-0.90
        """
        self.storage = storage
        self.similarity_threshold = similarity_threshold
        
        # 按日期建立索引，加速查找
        self.date_index = {}  # {日期字符串: [记忆列表]}
        
        logger.bind(tag=TAG).info(f"记忆去重器已初始化，相似度阈值: {similarity_threshold}")
    
    def build_index(self, existing_memories: List[Dict]):
        """
        建立日期索引，加速去重查找
        
        Args:
            existing_memories: 现有记忆列表
        """
        start_time = time.time()
        self.date_index = {}
        
        for memory in existing_memories:
            # 只索引active的记忆（过滤archived）
            if memory.get("status") != "active":
                continue
            
            text = memory.get("text", "").strip()
            if not text:
                continue
            
            # 提取日期
            date, _ = self._extract_date_and_content(text)
            date_key = date.strftime("%Y-%m-%d")
            
            if date_key not in self.date_index:
                self.date_index[date_key] = []
            
            self.date_index[date_key].append(memory)
        
        elapsed_time = time.time() - start_time
        total_memories = sum(len(memories) for memories in self.date_index.values())
        logger.bind(tag=TAG).debug(f"日期索引已建立: {len(self.date_index)} 个日期，{total_memories} 条记忆，耗时: {elapsed_time:.3f}s")
    
    def is_duplicate(self, new_memory_text: str, memory_type: str = None) -> Tuple[bool, Optional[Dict], float]:
        """
        判断新记忆是否与现有记忆重复
        
        规则：
        1. 提取事件发生的日期和内容（对于facts是事件日期，对于commitments是计划日期）
        2. 只与事件日期相同的现有记忆比较（不是创建记忆的日期）
        3. 在同一天事件日期的记忆中，计算语义相似度
        4. 如果相似度 >= 阈值，认为是重复
        5. 如果事件日期不同，即使内容相似也不去重
        
        示例：
        - Facts: "2026-01-01: 用户去爬山了" 与 "2026-01-01: 用户今天去爬山" → 同一天，相似 → 去重
        - Facts: "2026-01-01: 用户去爬山了" 与 "2026-01-02: 用户去爬山了" → 不同天 → 不去重
        - Commitments: "2026-01-18: 用户要去买衣服" 与 "2026-01-18: 用户想去买衣服" → 同一天，相似 → 去重
        - Commitments: "2026-01-15: 用户要去买衣服" 与 "2026-01-18: 用户要去买衣服" → 不同天 → 不去重
        
        Args:
            new_memory_text: 新记忆文本，格式如 "2026-01-05: 用户去爬山了"
            memory_type: 记忆类型（'facts' 或 'commitments'），可选，用于过滤
        
        Returns:
            (是否重复, 匹配的现有记忆, 相似度分数)
        """
        if not new_memory_text or not new_memory_text.strip():
            return False, None, 0.0
        
        start_time = time.time()
        
        # 1. 提取新记忆的日期和内容
        try:
            new_date, new_content = self._extract_date_and_content(new_memory_text)
        except Exception as e:
            logger.bind(tag=TAG).warning(f"提取日期和内容失败: {new_memory_text[:50]}, 错误: {e}")
            return False, None, 0.0
        
        date_key = new_date.strftime("%Y-%m-%d")
        
        # 2. 从索引中获取候选记忆（基于事件发生的日期，不是创建记忆的日期）
        # 关键：无论是facts还是commitments，都只与事件日期相同的记忆比较
        # - Facts：事件发生的日期（如"2026-01-01: 用户去爬山了"，日期是2026-01-01）
        # - Commitments：计划要做的日期（如"2026-01-18: 用户要去买衣服"，日期是2026-01-18）
        same_event_date_memories = self.date_index.get(date_key, [])
        
        # 3. 如果指定了memory_type，只比较同类型的记忆
        if memory_type:
            same_event_date_memories = [
                m for m in same_event_date_memories 
                if m.get("memory_type") == memory_type
            ]
        
        if not same_event_date_memories:
            # 没有同一天事件日期的记忆，肯定不重复
            return False, None, 0.0
        
        # 4. 获取新记忆的向量
        new_embedding = self._get_embedding(new_content)
        if new_embedding is None:
            logger.bind(tag=TAG).warning(f"无法生成新记忆的向量: {new_content[:50]}")
            return False, None, 0.0
        
        # 4. 与同一天事件日期的记忆计算语义相似度
        best_match = None
        best_similarity = 0.0
        
        for existing_memory in same_event_date_memories:
            existing_text = existing_memory.get("text", "").strip()
            if not existing_text:
                continue
            
            # 提取现有记忆的内容（去除日期）
            _, existing_content = self._extract_date_and_content(existing_text)
            
            # 获取现有记忆的向量
            existing_embedding = self._get_embedding(existing_content)
            if existing_embedding is None:
                continue
            
            # 计算余弦相似度
            similarity = self._cosine_similarity(new_embedding, existing_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_memory
        
        # 6. 判断是否重复
        is_dup = best_match is not None and best_similarity >= self.similarity_threshold
        
        elapsed_time = time.time() - start_time
        if is_dup:
            logger.bind(tag=TAG).info(
                f"检测到重复记忆（同一天事件日期）: {new_memory_text[:50]} "
                f"(与 {best_match.get('text', '')[:50]} 相似，相似度: {best_similarity:.4f})，"
                f"耗时: {elapsed_time:.3f}s"
            )
        elif elapsed_time > 0.01:
            logger.bind(tag=TAG).debug(
                f"去重检查完成: {new_memory_text[:50]} "
                f"(最佳相似度: {best_similarity:.4f}，阈值: {self.similarity_threshold})，"
                f"耗时: {elapsed_time:.3f}s"
            )
        
        return is_dup, best_match, best_similarity
    
    def _extract_date_and_content(self, memory_text: str) -> Tuple[datetime, str]:
        """
        提取记忆的日期和内容
        
        Args:
            memory_text: 记忆文本，格式如 "2026-01-05: 用户去爬山了"
        
        Returns:
            (日期对象, 内容文本)
        """
        # 格式："2026-01-05: 用户去爬山了" 或 "2026-01-05:用户去爬山了"
        date_match = re.match(r'^(\d{4}-\d{2}-\d{2}):\s*(.+)', memory_text)
        if date_match:
            date_str = date_match.group(1)
            content = date_match.group(2).strip()
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                return event_date, content
            except ValueError:
                logger.bind(tag=TAG).warning(f"日期格式错误: {date_str}，使用当前日期")
                return datetime.now(), memory_text
        else:
            # 如果没有日期前缀，使用当前日期
            logger.bind(tag=TAG).debug(f"记忆文本没有日期前缀: {memory_text[:50]}，使用当前日期")
            return datetime.now(), memory_text
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本的向量嵌入
        
        Args:
            text: 文本内容
        
        Returns:
            向量数组，如果失败返回None
        """
        if not text or not text.strip():
            return None
        
        try:
            # 使用存储层的embedding方法
            embedding = self.storage._get_embedding(text)
            return embedding
        except Exception as e:
            logger.bind(tag=TAG).error(f"生成embedding失败: {text[:50]}, 错误: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
        
        Returns:
            余弦相似度（0-1）
        """
        if not NUMPY_AVAILABLE:
            return 0.0
        
        try:
            # 计算点积
            dot_product = np.dot(vec1, vec2)
            
            # 计算范数
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # 避免除零
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # 余弦相似度
            similarity = dot_product / (norm1 * norm2)
            
            # 确保在[0, 1]范围内（对于归一化的向量，应该在[-1, 1]，但通常都是正数）
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.bind(tag=TAG).error(f"计算余弦相似度失败: {e}")
            return 0.0
    
    def batch_check_duplicates(self, new_memories: List[str], memory_type: str = None) -> List[Tuple[str, bool, Optional[Dict], float]]:
        """
        批量检查多个新记忆是否重复
        
        Args:
            new_memories: 新记忆文本列表
            memory_type: 记忆类型（'facts' 或 'commitments'），可选
        
        Returns:
            [(记忆文本, 是否重复, 匹配的现有记忆, 相似度), ...]
        """
        results = []
        for memory_text in new_memories:
            is_dup, matched, similarity = self.is_duplicate(memory_text, memory_type)
            results.append((memory_text, is_dup, matched, similarity))
        return results

