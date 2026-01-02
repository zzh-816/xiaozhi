"""记忆检索层：负责从存储层检索记忆"""
import json
import time
from typing import List, Dict, Optional

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ..base import logger
from .memory_storage import MemoryStorage

TAG = __name__


class MemoryRetriever:
    """记忆检索层：从存储层检索工作记忆、短期记忆和长期记忆"""
    
    def __init__(self, storage: MemoryStorage, vector_top_k: int = 3):
        self.storage = storage
        self.vector_top_k = vector_top_k
    
    def retrieve_working_memory(self, user_id: str, dialogue_history: List = None, 
                               limit: int = 10) -> List[Dict]:
        """检索工作记忆（排除当前对话历史中已有的消息）"""
        working_memory = self.storage.load_working_memory(user_id, limit)
        
        # 获取当前对话历史的内容（用于去重）
        current_dialogue_contents = set()
        if dialogue_history:
            for msg in dialogue_history:
                if hasattr(msg, 'content'):
                    current_dialogue_contents.add(msg.content)
        
        # 只返回不在当前对话历史中的工作记忆（跨会话恢复）
        result = []
        for msg in working_memory:
            if msg['content'] not in current_dialogue_contents:
                result.append(msg)
        
        return result
    
    def retrieve_short_term_memory(self, query: str) -> Dict[str, List[str]]:
        """检索短期记忆（facts和commitments）"""
        start_time = time.time()
        result = {"facts": [], "commitments": []}
        
        if not FAISS_AVAILABLE:
            logger.bind(tag=TAG).debug("FAISS不可用，跳过向量检索")
            return result
        
        vector_index = self.storage.get_vector_index()
        vector_metadata = self.storage.get_vector_metadata()
        
        if vector_index is None or vector_index.ntotal == 0:
            logger.bind(tag=TAG).debug(f"向量索引为空或未初始化，索引总数: {vector_index.ntotal if vector_index else 0}")
            return result
        
        logger.bind(tag=TAG).info(f"开始向量检索，查询: {query[:50]}, 索引总数: {vector_index.ntotal}, 元数据总数: {len(vector_metadata)}")
        
        try:
            # 获取查询向量
            embedding_start = time.time()
            query_embedding = self._get_embedding(query)
            embedding_time = time.time() - embedding_start
            logger.bind(tag=TAG).info(f"生成查询向量耗时: {embedding_time:.3f}s")
            
            if query_embedding is None:
                logger.bind(tag=TAG).warning("无法生成查询向量")
                return result
            
            # 检查向量维度
            query_dim = len(query_embedding)
            index_dim = vector_index.d
            logger.bind(tag=TAG).info(f"查询向量维度: {query_dim}, 索引维度: {index_dim}")
            
            if query_dim != index_dim:
                logger.bind(tag=TAG).error(f"向量维度不匹配！查询向量维度 {query_dim} 与索引维度 {index_dim} 不一致，无法进行搜索。请清空向量索引重新生成。")
                return result
            
            # 检查向量是否全为0或异常
            if np.allclose(query_embedding, 0):
                logger.bind(tag=TAG).warning("查询向量全为0，可能模型未正确初始化")
            
            # 检索top-K
            search_start = time.time()
            k = min(self.vector_top_k, vector_index.ntotal)
            query_2d = query_embedding.reshape(1, -1)
            logger.bind(tag=TAG).debug(f"搜索参数: k={k}, 查询向量shape={query_2d.shape}, 索引向量数={vector_index.ntotal}")
            distances, indices = vector_index.search(query_2d, k)
            search_time = time.time() - search_start
            logger.bind(tag=TAG).info(f"向量搜索耗时: {search_time:.3f}s，检索到 {len(indices[0])} 个结果，距离: {distances[0]}")
            
            # 检查距离是否异常（全为0）
            if len(distances[0]) > 0 and np.allclose(distances[0], 0):
                logger.bind(tag=TAG).warning(f"所有检索结果的距离都为0，这可能是由于：1) 索引中的向量是用旧方法生成的 2) 向量维度不匹配 3) 向量生成失败")
            
            for i, idx in enumerate(indices[0]):
                if idx < len(vector_metadata):
                    meta = vector_metadata[idx]
                    distance = distances[0][i] if i < len(distances[0]) else None
                    logger.bind(tag=TAG).info(f"检索结果 {i+1}: 类型={meta['memory_type']}, 距离={distance:.4f}, 内容={meta['text'][:50]}")
                    if meta["memory_type"] == "facts":
                        result["facts"].append(meta["text"])
                    elif meta["memory_type"] == "commitments":
                        result["commitments"].append(meta["text"])
                else:
                    logger.bind(tag=TAG).warning(f"索引 {idx} 超出元数据范围 (总数: {len(vector_metadata)})")
        except Exception as e:
            import traceback
            logger.bind(tag=TAG).error(f"向量检索失败: {e}, 错误详情: {traceback.format_exc()}")
        
        total_time = time.time() - start_time
        logger.bind(tag=TAG).info(f"向量检索完成，facts: {len(result['facts'])} 条, commitments: {len(result['commitments'])} 条，总耗时: {total_time:.3f}s")
        return result
    
    def retrieve_long_term_memory(self, user_id: str) -> Optional[Dict]:
        """检索长期记忆（profile）"""
        return self.storage.load_profile(user_id)
    
    def _get_embedding(self, text: str):
        """获取文本的向量嵌入（使用存储层的embedding方法）"""
        # 直接调用storage的embedding方法，保持一致性
        return self.storage._get_embedding(text)
    

