"""记忆存储层：负责数据库和向量索引的存储操作"""
import os
import json
import re
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import time

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..base import logger

TAG = __name__


class MemoryStorage:
    """记忆存储层：处理SQLite数据库和FAISS向量索引的存储"""
    
    def __init__(self, db_path: str, vector_index_path: str, vector_metadata_path: str, 
                 embedding_dim: int = 384, embedder_config: Optional[Dict] = None):
        self.db_path = db_path
        self.vector_index_path = vector_index_path
        self.vector_metadata_path = vector_metadata_path
        self.embedding_dim = embedding_dim
        
        # Embedding配置
        self.embedder_config = embedder_config or {}
        self.embedder_client = None
        self.embedder_model = None  # 本地模型对象
        self.normalize_embeddings = True  # 是否归一化向量
        self._init_embedder()
        
        # 向量索引和元数据
        self.vector_index = None
        self.vector_metadata = []
        
        # 初始化
        self._init_database()
        self._load_vector_index()
    
    def _init_embedder(self):
        """初始化Embedding客户端"""
        embedder = self.embedder_config.get("embedder", {})
        provider = embedder.get("provider", "sentence_transformers")  # 默认使用本地模型
        
        # 优先使用本地模型（sentence_transformers）
        if provider == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            config = embedder.get("config", {})
            model_name = config.get("model_name", "BAAI/bge-small-zh-v1.5")
            device = config.get("device", "cpu")
            embedding_dims = config.get("embedding_dims", 512)
            self.normalize_embeddings = config.get("normalize_embeddings", True)
            
            try:
                # 检查是否是本地路径（相对路径或绝对路径）
                from config.config_loader import get_project_dir
                project_dir = get_project_dir().rstrip('/').rstrip('\\')
                
                # 尝试多种路径
                possible_paths = []
                if os.path.isabs(model_name):
                    possible_paths.append(model_name)
                else:
                    # 相对路径：尝试项目目录下的路径
                    possible_paths.append(os.path.join(project_dir, model_name))
                    # 尝试当前工作目录
                    possible_paths.append(os.path.join(os.getcwd(), model_name))
                    # 尝试直接路径
                    possible_paths.append(model_name)
                
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        logger.bind(tag=TAG).info(f"找到本地模型路径: {model_path}")
                        break
                
                if model_path:
                    # 验证路径是否包含模型文件
                    required_files = ['config.json', 'pytorch_model.bin']
                    if not any(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                        # 也检查 safetensors 格式
                        if not os.path.exists(os.path.join(model_path, 'model.safetensors')):
                            logger.bind(tag=TAG).warning(f"模型路径 {model_path} 存在，但未找到模型文件（config.json 或 model.safetensors），将尝试从HuggingFace加载")
                            model_path = None
                
                # 只使用本地模型，不尝试从HuggingFace加载
                if not model_path:
                    logger.bind(tag=TAG).error(f"本地模型不存在，尝试过的路径: {possible_paths}")
                    logger.bind(tag=TAG).error(f"请确保模型文件存在于以下路径之一: {possible_paths[0] if possible_paths else model_name}")
                    raise FileNotFoundError(f"本地模型不存在: {model_name}")
                
                # 尝试加载本地模型
                try:
                    logger.bind(tag=TAG).info(f"使用本地模型路径: {model_path}")
                    self.embedder_model = SentenceTransformer(model_path, device=device)
                    logger.bind(tag=TAG).info(f"成功从本地路径加载模型")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"从本地路径加载模型失败: {e}")
                    logger.bind(tag=TAG).error(f"模型路径: {model_path}")
                    raise
                
                # 验证模型是否成功加载并测试
                if self.embedder_model is None:
                    raise ValueError("模型加载失败，embedder_model为None")
                
                # 测试模型并获取实际维度
                test_embedding = self.embedder_model.encode("test", convert_to_numpy=True)
                actual_dim = len(test_embedding)
                
                # 使用实际维度（模型可能返回不同的维度）
                self.embedding_dim = actual_dim
                if actual_dim != embedding_dims:
                    logger.bind(tag=TAG).warning(f"模型实际维度 {actual_dim} 与配置维度 {embedding_dims} 不一致，使用实际维度 {actual_dim}")
                
                logger.bind(tag=TAG).info(f"Embedding客户端已初始化: {provider}, 模型: {model_name}, 维度: {self.embedding_dim}, 设备: {device}")
            except Exception as e:
                import traceback
                logger.bind(tag=TAG).error(f"初始化本地模型失败: {e}，向量生成功能将不可用，错误详情: {traceback.format_exc()}")
                self.embedder_model = None
                self.embedder_client = None
        elif provider == "openai" and OPENAI_AVAILABLE:
            config = embedder.get("config", {})
            api_key = config.get("api_key", "")
            base_url = config.get("openai_base_url", "")
            embedding_dims = config.get("embedding_dims", 2560)
            
            if api_key and base_url:
                self.embedder_client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                self.embedding_dim = embedding_dims  # 使用配置的维度
                self.embedder_model_name = config.get("model", "doubao-embedding-text-240715")
                logger.bind(tag=TAG).info(f"Embedding客户端已初始化: {provider}, 模型: {self.embedder_model_name}, 维度: {self.embedding_dim}")
            else:
                logger.bind(tag=TAG).warning("Embedding API配置不完整，向量生成功能将不可用")
                self.embedder_client = None
        else:
            if provider == "sentence_transformers" and not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.bind(tag=TAG).error(f"Embedding provider '{provider}' 需要安装 sentence-transformers，向量生成功能将不可用。请运行: pip install sentence-transformers")
            elif provider not in ["sentence_transformers", "openai"]:
                logger.bind(tag=TAG).error(f"不支持的Embedding provider '{provider}'，向量生成功能将不可用。支持的provider: sentence_transformers, openai")
            self.embedder_client = None
            self.embedder_model = None
    
    def _init_database(self):
        """初始化SQLite数据库"""
        if not self.db_path:
            logger.bind(tag=TAG).warning("数据库路径为空，跳过初始化")
            return
        
        # 确保数据库文件所在目录存在
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                logger.bind(tag=TAG).error(f"创建数据库目录失败: {db_dir}, 错误: {e}")
                raise
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 工作记忆表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT
                )
            """)
            # 创建工作记忆索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_timestamp 
                ON working_memory(user_id, timestamp)
            """)
            
            # 长期记忆表（profile）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, memory_type)
                )
            """)
            # 创建长期记忆索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_type 
                ON long_term_memory(user_id, memory_type)
            """)
            
            conn.commit()
            conn.close()
            logger.bind(tag=TAG).info(f"数据库初始化完成: {self.db_path}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"数据库初始化失败: {self.db_path}, 错误: {e}")
            raise
    
    def save_working_memory(self, user_id: str, messages: List, limit: int = 10):
        """保存工作记忆到数据库"""
        start_time = time.time()
        if not user_id:
            logger.bind(tag=TAG).warning("user_id为空，跳过保存工作记忆")
            return
        
        if not self.db_path:
            logger.bind(tag=TAG).warning("数据库路径为空，跳过保存工作记忆")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 保存最近N轮对话
            recent_messages = messages[-(limit * 2):] if len(messages) > (limit * 2) else messages
            
            saved_count = 0
            for msg in recent_messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    if msg.role in ['user', 'assistant']:
                        content = msg.content if msg.content is not None else ""
                        if not content.strip():
                            logger.bind(tag=TAG).debug(f"跳过空的{msg.role}消息")
                            continue
                        
                        cursor.execute("""
                            INSERT INTO working_memory (user_id, role, content, timestamp)
                            VALUES (?, ?, ?, ?)
                        """, (user_id, msg.role, content, datetime.now().isoformat()))
                        saved_count += 1
            
            # 清理旧的工作记忆（只保留最近N*2条）
            cursor.execute("""
                DELETE FROM working_memory 
                WHERE user_id = ? 
                AND id NOT IN (
                    SELECT id FROM working_memory 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                )
            """, (user_id, user_id, limit * 2))
            
            conn.commit()
            conn.close()
            
            elapsed_time = time.time() - start_time
            logger.bind(tag=TAG).info(f"工作记忆保存耗时: {elapsed_time:.3f}s，保存了 {saved_count} 条")
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存工作记忆失败: {e}")
    
    def load_working_memory(self, user_id: str, limit: int = 10) -> List[Dict]:
        """从数据库加载工作记忆"""
        start_time = time.time()
        if not user_id:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT role, content FROM working_memory 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (user_id, limit * 2))
            
            rows = cursor.fetchall()
            conn.close()
            
            # 从旧到新返回
            result = [{"role": role, "content": content} for role, content in reversed(rows)]
            elapsed_time = time.time() - start_time
            logger.bind(tag=TAG).debug(f"工作记忆加载耗时: {elapsed_time:.3f}s，加载了 {len(result)} 条")
            return result
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载工作记忆失败: {e}")
            return []
    
    def save_profile(self, user_id: str, profile: Dict):
        """保存用户画像到数据库（合并模式，不会覆盖已有信息）"""
        start_time = time.time()
        if not user_id or not profile:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 先加载现有的profile（如果存在）
            load_start = time.time()
            existing_profile = None
            cursor.execute("""
                SELECT content FROM long_term_memory 
                WHERE user_id = ? AND memory_type = 'profile'
                ORDER BY updated_at DESC LIMIT 1
            """, (user_id,))
            row = cursor.fetchone()
            load_time = time.time() - load_start
            if load_time > 0.001:  # 只记录耗时超过1ms的
                logger.bind(tag=TAG).debug(f"Profile加载耗时: {load_time:.3f}s")
            
            if row:
                try:
                    existing_profile = json.loads(row[0])
                    logger.bind(tag=TAG).debug(f"加载现有profile用于合并: {json.dumps(existing_profile, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    logger.bind(tag=TAG).warning("现有profile格式错误，将覆盖")
                    existing_profile = None
            
            # 合并profile（新信息覆盖旧信息，但保留旧信息中没有的字段）
            merge_start = time.time()
            if existing_profile:
                merged_profile = self._merge_profile(existing_profile, profile)
                logger.bind(tag=TAG).debug(f"合并后的profile: {json.dumps(merged_profile, ensure_ascii=False)}")
            else:
                merged_profile = profile
            merge_time = time.time() - merge_start
            if merge_time > 0.001:
                logger.bind(tag=TAG).debug(f"Profile合并耗时: {merge_time:.3f}s")
            
            save_start = time.time()
            current_date = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("""
                INSERT OR REPLACE INTO long_term_memory (user_id, memory_type, content, timestamp, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                "profile",
                json.dumps(merged_profile, ensure_ascii=False),
                current_date,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            save_time = time.time() - save_start
            total_time = time.time() - start_time
            logger.bind(tag=TAG).info(f"Profile保存耗时: {total_time:.3f}s（数据库写入: {save_time:.3f}s）")
        except Exception as e:
            import traceback
            logger.bind(tag=TAG).error(f"保存profile失败: {e}, 错误详情: {traceback.format_exc()}")
    
    def save_extraction_state(self, user_id: str, last_extracted_count: int):
        """保存提取状态（已处理的用户消息数量）"""
        if not user_id:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("""
                INSERT OR REPLACE INTO long_term_memory (user_id, memory_type, content, timestamp, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                "extraction_state",
                json.dumps({"last_extracted_user_message_count": last_extracted_count}, ensure_ascii=False),
                current_date,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.bind(tag=TAG).debug(f"提取状态已保存: user_id={user_id}, count={last_extracted_count}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存提取状态失败: {e}")
    
    def load_extraction_state(self, user_id: str) -> int:
        """加载提取状态（已处理的用户消息数量）"""
        if not user_id:
            return 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT content FROM long_term_memory 
                WHERE user_id = ? AND memory_type = 'extraction_state'
                ORDER BY updated_at DESC LIMIT 1
            """, (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                state = json.loads(row[0])
                count = state.get("last_extracted_user_message_count", 0)
                logger.bind(tag=TAG).debug(f"提取状态已加载: user_id={user_id}, count={count}")
                return count
            else:
                logger.bind(tag=TAG).debug(f"未找到提取状态，使用默认值0: user_id={user_id}")
                return 0
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载提取状态失败: {e}")
            return 0
    
    def _merge_profile(self, existing: Dict, new: Dict) -> Dict:
        """合并两个profile字典（深度合并）"""
        merged = existing.copy()
        
        for key, value in new.items():
            if key in merged:
                if isinstance(merged[key], dict) and isinstance(value, dict):
                    # 递归合并字典
                    merged[key] = self._merge_profile(merged[key], value)
                elif isinstance(merged[key], list) and isinstance(value, list):
                    # 合并列表（去重）
                    merged[key] = list(set(merged[key] + value))
                else:
                    # 新值覆盖旧值
                    merged[key] = value
            else:
                # 新键直接添加
                merged[key] = value
        
        return merged
    
    def load_profile(self, user_id: str) -> Optional[Dict]:
        """从数据库加载用户画像"""
        start_time = time.time()
        if not user_id:
            logger.bind(tag=TAG).warning(f"user_id为空，无法加载profile")
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT content FROM long_term_memory 
                WHERE user_id = ? AND memory_type = 'profile'
                ORDER BY updated_at DESC LIMIT 1
            """, (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            elapsed_time = time.time() - start_time
            if row:
                profile = json.loads(row[0])
                logger.bind(tag=TAG).info(f"Profile已从数据库加载，user_id: {user_id}, 内容: {json.dumps(profile, ensure_ascii=False)}, 耗时: {elapsed_time:.3f}s")
                return profile
            else:
                logger.bind(tag=TAG).warning(f"未找到profile，user_id: {user_id}, 数据库路径: {self.db_path}, 耗时: {elapsed_time:.3f}s")
                # 检查数据库中是否有其他user_id的记录
                check_start = time.time()
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT user_id FROM long_term_memory WHERE memory_type = 'profile'
                """)
                all_user_ids = cursor.fetchall()
                conn.close()
                check_time = time.time() - check_start
                if all_user_ids:
                    logger.bind(tag=TAG).info(f"数据库中存在的user_id: {[uid[0] for uid in all_user_ids]}（检查耗时: {check_time:.3f}s）")
        except Exception as e:
            import traceback
            logger.bind(tag=TAG).error(f"加载profile失败: {e}, 错误详情: {traceback.format_exc()}")
        
        return None
    
    def _load_vector_index(self):
        """加载向量索引"""
        if not FAISS_AVAILABLE:
            logger.bind(tag=TAG).warning("FAISS未安装，向量检索功能不可用")
            return
        
        if not self.vector_index_path:
            return
        
        try:
            if os.path.exists(self.vector_index_path):
                self.vector_index = faiss.read_index(self.vector_index_path)
                # 检查维度是否匹配
                if self.vector_index.d != self.embedding_dim:
                    logger.bind(tag=TAG).warning(f"向量索引维度不匹配: 索引维度 {self.vector_index.d}, 配置维度 {self.embedding_dim}")
                    if self.vector_index.ntotal > 0:
                        # 如果索引中有数据，使用索引的维度（避免丢失数据）
                        logger.bind(tag=TAG).warning(f"索引中有 {self.vector_index.ntotal} 条数据，使用索引维度 {self.vector_index.d}")
                        self.embedding_dim = self.vector_index.d  # 更新配置以匹配索引
                    else:
                        # 如果索引为空，重新创建以匹配配置
                        logger.bind(tag=TAG).warning("索引为空，重新创建向量索引以匹配当前维度")
                        self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
                        self.vector_metadata = []  # 清空元数据
                    logger.bind(tag=TAG).info(f"最终使用维度: {self.embedding_dim}")
                else:
                    logger.bind(tag=TAG).debug(f"向量索引已加载: {self.vector_index.ntotal} 条，维度: {self.vector_index.d}")
            else:
                # 创建新的索引
                self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
                logger.bind(tag=TAG).debug(f"创建新的向量索引，维度: {self.embedding_dim}")
            
            # 加载元数据
            if os.path.exists(self.vector_metadata_path):
                with open(self.vector_metadata_path, 'r', encoding='utf-8') as f:
                    loaded_metadata = json.load(f)
                    # 如果索引被重建，清空元数据
                    if self.vector_index.ntotal == 0:
                        self.vector_metadata = []
                        logger.bind(tag=TAG).debug("索引为空，清空元数据")
                    else:
                        self.vector_metadata = loaded_metadata
                        logger.bind(tag=TAG).debug(f"向量元数据已加载: {len(self.vector_metadata)} 条")
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载向量索引失败: {e}")
            if FAISS_AVAILABLE:
                self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
            self.vector_metadata = []
    
    def save_vector_memory(self, memory_text: str, memory_type: str, user_id: str, 
                          current_date: str) -> bool:
        """保存向量记忆（facts或commitments）"""
        start_time = time.time()
        if not FAISS_AVAILABLE or self.vector_index is None:
            return False
        
        if not memory_text or not memory_text.strip():
            return False
        
        try:
            # 获取向量
            embedding_start = time.time()
            embedding = self._get_embedding(memory_text)
            embedding_time = time.time() - embedding_start
            if embedding_time > 0.001:
                logger.bind(tag=TAG).debug(f"{memory_type}生成向量耗时: {embedding_time:.3f}s")
            
            if embedding is None:
                logger.bind(tag=TAG).warning(f"无法生成{memory_type}的向量: {memory_text[:50]}")
                return False
            
            # 检查向量维度是否匹配
            if self.vector_index.d != len(embedding):
                logger.bind(tag=TAG).error(f"向量维度不匹配: 索引期望 {self.vector_index.d} 维, 实际 {len(embedding)} 维")
                if self.vector_index.ntotal > 0:
                    logger.bind(tag=TAG).error(f"索引中有 {self.vector_index.ntotal} 条旧数据（维度: {self.vector_index.d}），但新向量维度为 {len(embedding)}。这会导致搜索失败！")
                    logger.bind(tag=TAG).error(f"建议：删除向量索引文件重新生成。文件路径: {self.vector_index_path}")
                    # 不调整向量，直接返回False，避免保存错误维度的向量
                    return False
                else:
                    logger.bind(tag=TAG).warning("索引为空，重新创建向量索引以匹配向量维度")
                    self.vector_index = faiss.IndexFlatL2(len(embedding))
                    self.embedding_dim = len(embedding)
                    self.vector_metadata = []
                    logger.bind(tag=TAG).info(f"已重新创建索引（维度: {len(embedding)}）")
            
            # 添加到索引
            add_start = time.time()
            embedding_2d = embedding.reshape(1, -1)
            self.vector_index.add(embedding_2d)
            add_time = time.time() - add_start
            
            # timestamp始终使用保存时的日期（current_date），表示这条记忆是什么时候被保存的
            # text中的日期是事件发生的日期，可能与timestamp不同（例如：用户说"我昨天去爬山了"）
            # 格式可能是："2026-01-01: 用户去爬山了" 或 "2026-01-01:用户去爬山了"
            timestamp = current_date  # 始终使用保存时的日期
            date_match = re.match(r'^(\d{4}-\d{2}-\d{2}):\s*', memory_text)
            if date_match:
                extracted_date = date_match.group(1)
                logger.bind(tag=TAG).debug(f"memory_text中包含事件日期: {extracted_date}，但timestamp使用保存时的日期: {current_date}")
            
            # 保存元数据
            self.vector_metadata.append({
                "text": memory_text,
                "memory_type": memory_type,
                "user_id": user_id,
                "timestamp": timestamp
            })
            
            total_time = time.time() - start_time
            if total_time > 0.001:
                logger.bind(tag=TAG).debug(f"{memory_type}保存到内存耗时: {total_time:.3f}s（添加到索引: {add_time:.3f}s）")
            return True
        except Exception as e:
            import traceback
            logger.bind(tag=TAG).error(f"保存{memory_type}失败: {memory_text[:50] if memory_text else '空内容'}, 错误: {traceback.format_exc()}")
            return False
    
    def save_vector_index(self):
        """保存向量索引到文件"""
        start_time = time.time()
        if not FAISS_AVAILABLE:
            logger.bind(tag=TAG).warning("FAISS未安装，无法保存向量索引")
            return
        
        if self.vector_index is None:
            logger.bind(tag=TAG).warning("向量索引为None，无法保存")
            return
        
        if not self.vector_index_path:
            logger.bind(tag=TAG).warning("向量索引路径为空，无法保存")
            return
        
        try:
            # 确保目录存在
            index_dir = os.path.dirname(self.vector_index_path)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
            
            # 保存索引文件
            index_start = time.time()
            faiss.write_index(self.vector_index, self.vector_index_path)
            index_time = time.time() - index_start
            
            # 保存元数据文件
            metadata_start = time.time()
            with open(self.vector_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.vector_metadata, f, ensure_ascii=False, indent=2)
            metadata_time = time.time() - metadata_start
            
            total_time = time.time() - start_time
            logger.bind(tag=TAG).info(f"向量索引保存耗时: {total_time:.3f}s（索引文件: {index_time:.3f}s, 元数据文件: {metadata_time:.3f}s）")
            logger.bind(tag=TAG).info(f"向量索引已保存: {self.vector_index_path} (共 {self.vector_index.ntotal} 条向量)")
            logger.bind(tag=TAG).info(f"向量元数据已保存: {self.vector_metadata_path} (共 {len(self.vector_metadata)} 条)")
        except Exception as e:
            import traceback
            logger.bind(tag=TAG).error(f"保存向量索引失败: {traceback.format_exc()}")
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本的向量嵌入（支持本地模型和API）"""
        if not FAISS_AVAILABLE:
            logger.bind(tag=TAG).error("FAISS不可用，无法生成向量")
            return None
        
        # 优先使用本地模型（sentence_transformers）
        if self.embedder_model is not None:
            try:
                embedding = self.embedder_model.encode(
                    text,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=False
                ).astype(np.float32)
                
                # 检查维度
                if len(embedding) != self.embedding_dim:
                    logger.bind(tag=TAG).warning(f"本地模型返回的向量维度不匹配: 期望 {self.embedding_dim}, 实际 {len(embedding)}")
                    # 如果索引为空，更新维度配置
                    if self.vector_index is not None and self.vector_index.ntotal == 0:
                        self.embedding_dim = len(embedding)
                        logger.bind(tag=TAG).info(f"更新embedding维度为: {self.embedding_dim}")
                
                # 检查向量是否有效
                if np.allclose(embedding, 0):
                    logger.bind(tag=TAG).warning(f"生成的向量全为0，可能模型有问题，文本: {text[:50]}")
                else:
                    logger.bind(tag=TAG).info(f"使用本地模型生成embedding: 文本长度={len(text)}, 向量维度={len(embedding)}, 向量范数={np.linalg.norm(embedding):.4f}")
                return embedding
            except Exception as e:
                import traceback
                logger.bind(tag=TAG).error(f"本地模型生成embedding失败: {e}，错误详情: {traceback.format_exc()}")
                return None
        
        # 如果配置了API客户端，使用API
        if self.embedder_client is not None:
            try:
                response = self.embedder_client.embeddings.create(
                    model=self.embedder_model_name,
                    input=text
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                
                # 检查维度
                if len(embedding) != self.embedding_dim:
                    logger.bind(tag=TAG).warning(f"API返回的向量维度不匹配: 期望 {self.embedding_dim}, 实际 {len(embedding)}")
                    if len(embedding) < self.embedding_dim:
                        embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), 'constant')
                    else:
                        embedding = embedding[:self.embedding_dim]
                
                # 归一化
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                logger.bind(tag=TAG).info(f"使用API生成embedding: 文本长度={len(text)}, 向量维度={len(embedding)}")
                return embedding
            except Exception as e:
                import traceback
                logger.bind(tag=TAG).error(f"调用Embedding API失败: {e}，错误详情: {traceback.format_exc()}")
                return None
        
        # 如果既没有本地模型也没有API，返回None
        logger.bind(tag=TAG).error(f"无法生成embedding：既没有本地模型也没有API客户端。文本: {text[:50]}")
        return None
    
    def get_vector_index(self):
        """获取向量索引（供检索层使用）"""
        return self.vector_index
    
    def get_vector_metadata(self):
        """获取向量元数据（供检索层使用）"""
        return self.vector_metadata

