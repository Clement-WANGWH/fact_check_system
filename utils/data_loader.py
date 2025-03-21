import json
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class KnowledgeBase:
    """基础知识库加载与管理类"""
    
    def __init__(self, kb_path: str):
        """
        初始化知识库
        
        参数:
            kb_path: 知识库JSONL文件路径
        """
        self.kb_path = kb_path
        self.knowledge_items = self._load_knowledge_base()
        
    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载知识条目
        
        返回:
            知识条目列表
        """
        if not os.path.exists(self.kb_path):
            print(f"警告: 知识库文件未找到，路径: {self.kb_path}")
            return []
            
        knowledge_items = []
        with open(self.kb_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    knowledge_items.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行: {line[:50]}...")
        
        print(f"已从{self.kb_path}加载{len(knowledge_items)}条知识条目")
        return knowledge_items
    
    def get_all_items(self) -> List[Dict[str, Any]]:
        """
        获取所有知识条目
        
        返回:
            所有知识条目列表
        """
        return self.knowledge_items
        
    def get_item_by_claim(self, claim: str) -> Optional[Dict[str, Any]]:
        """
        根据声明文本查找知识条目
        
        参数:
            claim: 声明文本
            
        返回:
            匹配的知识条目，未找到则返回None
        """
        for item in self.knowledge_items:
            if item.get("claim") == claim:
                return item
        return None
        
    def filter_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        根据域过滤知识条目
        
        参数:
            domain: 域名称
            
        返回:
            过滤后的知识条目列表
        """
        return [item for item in self.knowledge_items if item.get("domain") == domain]


class Document:
    """文档块数据结构，用于向量存储"""
    
    def __init__(self, id: str, text: str, metadata: Dict[str, Any], embedding: Optional[np.ndarray] = None):
        """
        初始化文档块
        
        参数:
            id: 文档ID
            text: 文档文本内容
            metadata: 文档元数据
            embedding: 文档向量表示
        """
        self.id = id
        self.text = text
        self.metadata = metadata
        self.embedding = embedding
        
    def __repr__(self) -> str:
        return f"Document(id={self.id}, text={self.text[:50]}..., metadata={self.metadata})"


class VectorStore:
    """向量存储类，管理文档向量及相似度搜索"""
    
    def __init__(self):
        """初始化向量存储"""
        self.documents = {}  # id -> Document映射
        self.embeddings = None  # 所有文档的嵌入矩阵
        self.doc_ids = []  # 对应embeddings中的文档ID
        
    def add_document(self, document: Document):
        """
        添加单个文档
        
        参数:
            document: 文档对象
        """
        if document.id in self.documents:
            print(f"警告: 文档ID {document.id} 已存在，将被覆盖")
            
        self.documents[document.id] = document
        self._update_embeddings()
        
    def add_documents(self, documents: List[Document]):
        """
        批量添加文档
        
        参数:
            documents: 文档对象列表
        """
        for doc in documents:
            self.documents[doc.id] = doc
        self._update_embeddings()
        
    def _update_embeddings(self):
        """更新内部嵌入矩阵"""
        self.doc_ids = list(self.documents.keys())
        
        if not self.doc_ids:
            self.embeddings = None
            return
            
        # 收集所有有嵌入向量的文档
        valid_embeddings = []
        valid_ids = []
        
        for doc_id in self.doc_ids:
            doc = self.documents[doc_id]
            if doc.embedding is not None:
                valid_embeddings.append(doc.embedding)
                valid_ids.append(doc_id)
                
        if valid_embeddings:
            self.embeddings = np.vstack(valid_embeddings)
            self.doc_ids = valid_ids
        else:
            self.embeddings = None
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        向量相似度搜索
        
        参数:
            query_embedding: 查询向量
            top_k: 返回的最大结果数
            
        返回:
            文档和相似度分数的元组列表
        """
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []
            
        # 计算余弦相似度
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # 获取top-k结果索引
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
        # 构建结果
        results = []
        for idx in top_indices:
            if idx < len(self.doc_ids) and similarities[idx] > 0:
                doc_id = self.doc_ids[idx]
                results.append((self.documents[doc_id], float(similarities[idx])))
                
        return results
    
    def save(self, file_path: str):
        """
        保存向量存储到磁盘
        
        参数:
            file_path: 保存路径
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'doc_ids': self.doc_ids
            }, f)
            
    @classmethod
    def load(cls, file_path: str) -> 'VectorStore':
        """
        从磁盘加载向量存储
        
        参数:
            file_path: 加载路径
            
        返回:
            加载的向量存储对象
        """
        vector_store = cls()
        
        if not os.path.exists(file_path):
            print(f"警告: 向量存储文件不存在，路径: {file_path}")
            return vector_store
            
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            vector_store.documents = data['documents']
            vector_store.embeddings = data['embeddings']
            vector_store.doc_ids = data['doc_ids']
            
        return vector_store


class RAGKnowledgeBase(KnowledgeBase):
    """RAG增强知识库，扩展基础知识库，添加向量检索能力"""
    
    def __init__(self, kb_path: str, embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化RAG知识库
        
        参数:
            kb_path: 知识库JSONL文件路径
            embedding_model_name: 嵌入模型名称
        """
        super().__init__(kb_path)
        
        # 初始化向量存储
        self.vector_store = VectorStore()
        
        # 初始化嵌入模型
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            print(f"已加载嵌入模型: {embedding_model_name}")
        except Exception as e:
            print(f"警告: 加载嵌入模型失败: {e}")
            print("将使用TF-IDF作为备用方案")
            self.embedding_model = None
            
        # 使用TF-IDF作为备用嵌入方法
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
        
        # 初始化索引
        self._build_index()
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的嵌入向量
        
        参数:
            text: 文本
            
        返回:
            嵌入向量
        """
        if self.embedding_model is not None:
            # 使用预训练模型获取嵌入
            return self.embedding_model.encode(text)
        else:
            # 退回到TF-IDF
            if not hasattr(self, 'tfidf_matrix'):
                corpus = [item["claim"] for item in self.knowledge_items]
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
                
            # 获取新文本的TF-IDF向量
            text_vector = self.tfidf_vectorizer.transform([text])
            return text_vector.toarray()[0]
            
    def _chunk_documents(self) -> List[Document]:
        """
        将知识条目拆分为文档块
        
        返回:
            文档块列表
        """
        documents = []
        
        for idx, item in enumerate(self.knowledge_items):
            claim = item.get("claim", "")
            
            if not claim:
                continue
                
            # 为每个知识条目创建一个文档
            doc_id = f"doc_{idx}"
            
            # 创建元数据
            metadata = {k: v for k, v in item.items() if k != "claim"}
            
            # 获取嵌入向量
            embedding = self._get_embedding(claim)
            
            # 创建文档对象
            document = Document(
                id=doc_id,
                text=claim,
                metadata=metadata,
                embedding=embedding
            )
            
            documents.append(document)
            
        return documents
        
    def _build_index(self):
        """构建向量索引"""
        # 拆分文档
        documents = self._chunk_documents()
        
        # 添加到向量存储
        self.vector_store.add_documents(documents)
        
        print(f"已为{len(documents)}个知识条目构建向量索引")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        基于语义相似度检索相关知识条目
        
        参数:
            query: 查询文本
            top_k: 返回的最大结果数
            
        返回:
            检索到的知识条目列表
        """
        # 获取查询嵌入
        query_embedding = self._get_embedding(query)
        
        # 执行向量搜索
        search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # 转换结果格式
        retrieved_items = []
        for doc, score in search_results:
            # 构建结果项
            item = {
                "claim": doc.text,
                "similarity_score": score,
                **doc.metadata  # 展开元数据
            }
            retrieved_items.append(item)
            
        return retrieved_items
        
    def retrieve_with_queries(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        基于多个查询变体检索并合并结果
        
        参数:
            queries: 查询文本列表
            top_k: 每个查询返回的最大结果数
            
        返回:
            合并后的检索结果
        """
        all_results = {}  # claim -> item映射
        
        for query in queries:
            results = self.retrieve(query, top_k=top_k)
            
            for item in results:
                claim = item["claim"]
                if claim not in all_results or item["similarity_score"] > all_results[claim]["similarity_score"]:
                    all_results[claim] = item
                    
        # 转换为列表并按相似度排序
        merged_results = list(all_results.values())
        merged_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # 返回top-k合并结果
        return merged_results[:top_k]
        
    def save_index(self, file_path: str):
        """
        保存向量索引到磁盘
        
        参数:
            file_path: 保存路径
        """
        self.vector_store.save(file_path)
        print(f"已保存向量索引到: {file_path}")
        
    def load_index(self, file_path: str):
        """
        从磁盘加载向量索引
        
        参数:
            file_path: 加载路径
        """
        self.vector_store = VectorStore.load(file_path)
        print(f"已从{file_path}加载向量索引")
