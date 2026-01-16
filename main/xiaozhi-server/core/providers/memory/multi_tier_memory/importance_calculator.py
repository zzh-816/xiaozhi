
"""重要性计算器：基于多因素动态计算记忆的重要性"""
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

from ..base import logger

TAG = __name__


class ImportanceCalculator:
    """重要性计算器：基于访问频率、时间衰减、最近访问等因素计算重要性"""
    
    def __init__(self):
        """初始化重要性计算器，配置各因子的权重"""
        # 权重配置（总和应该接近1.0）
        self.weights = {
            'base': 0.1,              # 基础重要性
            'access_frequency': 0.4,  # 访问频率（最重要）
            'time_decay': 0.2,        # 时间衰减
            'recent_access': 0.3,     # 最近访问加成
        }
        
        # 参数配置
        self.max_access_count = 100  # 访问次数上限（超过按100计算）
        self.decay_days = 365        # 完全衰减所需天数
        self.recent_access_days = 30  # 最近访问的有效天数
        
        logger.bind(tag=TAG).debug(f"重要性计算器已初始化，权重: {self.weights}")
    
    def calculate(self, memory_metadata: Dict) -> float:
        """
        计算记忆的重要性分数（0-1）
        
        Args:
            memory_metadata: 记忆元数据字典，包含：
                - importance: 当前重要性（可选，用于初始计算）
                - access_count: 访问次数
                - last_accessed: 最后访问时间（YYYY-MM-DD格式）
                - created_at: 创建时间（YYYY-MM-DD格式）
                - memory_type: 记忆类型（facts/commitments）
        
        Returns:
            重要性分数（0-1）
        """
        try:
            # 1. 基础重要性
            base_importance = memory_metadata.get("importance", 0.5)
            # 根据记忆类型调整基础重要性
            memory_type = memory_metadata.get("memory_type", "facts")
            if memory_type == "commitments":
                base_importance = 0.6  # commitments初始更重要
            
            base_factor = base_importance * self.weights['base']
            
            # 2. 访问频率因子
            access_count = memory_metadata.get("access_count", 0)
            access_factor = self._calculate_access_factor(access_count) * self.weights['access_frequency']
            
            # 3. 时间衰减因子
            created_at_str = memory_metadata.get("created_at", "")
            time_decay_factor = self._calculate_time_decay(created_at_str) * self.weights['time_decay']
            
            # 4. 最近访问加成
            last_accessed_str = memory_metadata.get("last_accessed", "")
            recent_access_factor = self._calculate_recent_access_factor(last_accessed_str) * self.weights['recent_access']
            
            # 5. 综合计算
            total_importance = (
                base_factor +
                access_factor +
                time_decay_factor +
                recent_access_factor
            )
            
            # 限制在[0, 1]范围内
            total_importance = max(0.0, min(1.0, total_importance))
            
            return total_importance
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"计算重要性失败: {e}，使用默认值0.5")
            return 0.5
    
    def _calculate_access_factor(self, access_count: int) -> float:
        """
        计算访问频率因子（0-1）
        
        Args:
            access_count: 访问次数
        
        Returns:
            访问频率因子（0-1）
        """
        # 归一化到[0, 1]
        normalized_count = min(access_count, self.max_access_count) / self.max_access_count
        return normalized_count
    
    def _calculate_time_decay(self, created_at_str: str) -> float:
        """
        计算时间衰减因子（0-1）
        
        Args:
            created_at_str: 创建时间字符串（YYYY-MM-DD格式）
        
        Returns:
            时间衰减因子（0-1），越新越接近1，越旧越接近0
        """
        if not created_at_str:
            return 0.5  # 默认值
        
        try:
            created_at = datetime.strptime(created_at_str, "%Y-%m-%d")
            current_date = datetime.now()
            days_since_creation = (current_date - created_at).days
            
            # 线性衰减：1年后衰减到0
            decay_factor = max(0.0, 1.0 - (days_since_creation / self.decay_days))
            return decay_factor
        except (ValueError, TypeError) as e:
            logger.bind(tag=TAG).warning(f"解析创建时间失败: {created_at_str}, 错误: {e}")
            return 0.5
    
    def _calculate_recent_access_factor(self, last_accessed_str: str) -> float:
        """
        计算最近访问加成因子（0-1）
        
        Args:
            last_accessed_str: 最后访问时间字符串（YYYY-MM-DD格式）
        
        Returns:
            最近访问因子（0-1），30天内访问过接近1，超过30天接近0
        """
        if not last_accessed_str:
            return 0.0  # 从未访问过
        
        try:
            last_accessed = datetime.strptime(last_accessed_str, "%Y-%m-%d")
            current_date = datetime.now()
            days_since_access = (current_date - last_accessed).days
            
            # 30天内访问过有加成，超过30天加成消失
            if days_since_access <= self.recent_access_days:
                factor = 1.0 - (days_since_access / self.recent_access_days) * 0.5  # 最多降低50%
            else:
                factor = 0.0
            
            return max(0.0, min(1.0, factor))
        except (ValueError, TypeError) as e:
            logger.bind(tag=TAG).warning(f"解析最后访问时间失败: {last_accessed_str}, 错误: {e}")
            return 0.0
    
    def batch_calculate(self, memories_metadata: list) -> Dict[int, float]:
        """
        批量计算多条记忆的重要性
        
        Args:
            memories_metadata: 记忆元数据列表
        
        Returns:
            {vector_index: importance} 字典
        """
        results = {}
        for memory in memories_metadata:
            vector_index = memory.get("vector_index")
            if vector_index is not None:
                importance = self.calculate(memory)
                results[vector_index] = importance
        return results
