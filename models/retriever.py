import os
import json
import hashlib
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


class Retriever:
    """事实核查系统的检索模块，针对英文文本优化"""

    def __init__(self, knowledge_items: List[Dict[str, Any]], top_k: int,
                 cache_dir: str, force_rebuild: bool = False):
        """
        Initialize the retriever

        参数:
            knowledge_items: 知识条目列表
            top_k: 检索的知识条目
            cache_dir: 存储缓存的地址
            force_rebuild: 如果缓存存在，是否强制重建索引
        """
        self.knowledge_items = knowledge_items
        self.top_k = top_k
        self.cache_dir = cache_dir

        # 初始化必要的属性，以防_build_index失败
        self.vectorizer = None
        self.claims = []
        self.claim_vectors = None
        self.is_english = []

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

        # 构建索引
        try:
            self._build_index(force_rebuild)
        except Exception as e:
            print(f"警告: 构建索引时出错: {e}")
            print("将使用简单的文本匹配作为备用方案")
            # 确保在索引构建失败的情况下有一个基本的向量化器
            self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
            if self.knowledge_items and len(self.knowledge_items) > 0:
                self.claims = [item.get("claim", "") for item in self.knowledge_items]
                self.claim_vectors = self.vectorizer.fit_transform(self.claims)
                self.is_english = [self._is_english(claim) for claim in self.claims]

    def _calculate_kb_checksum(self) -> str:
        """
        计算知识库的哈希码，以检测变化

        返回:
            哈希码
        """
        # 创建知识库的字符串表示
        kb_str = json.dumps(self.knowledge_items, sort_keys=True)

        # 计算哈希码
        return hashlib.md5(kb_str.encode('utf-8')).hexdigest()

    def _get_cache_path(self) -> str:
        """
        获取缓存文件的路径

        返回:
            缓存文件路径
        """
        checksum = self._calculate_kb_checksum()
        return os.path.join(self.cache_dir, f"retriever_cache_{checksum}.pkl")

    def _build_index(self, force_rebuild: bool = False):
        """
        建立或加载用于检索的 TF-IDF 索引

        参数:
            force_rebuild: 是否强制重建索引
        """
        cache_path = self._get_cache_path()

        # 尝试从缓存加载
        if not force_rebuild and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.claims = cache_data['claims']
                    self.vectorizer = cache_data['vectorizer']
                    self.claim_vectors = cache_data['claim_vectors']
                    self.is_english = cache_data.get('is_english', [True] * len(self.claims))

                print(f"从:' {cache_path} ' 加载缓存成功")
                return
            except Exception as e:
                print(f"缓存加载错误，重建索引: {e}")

        # 确保知识库不为空
        if not self.knowledge_items:
            print("警告: 知识库为空，无法构建索引")
            self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
            self.claims = []
            self.claim_vectors = self.vectorizer.fit_transform([])
            self.is_english = []
            return

        self.claims = [item.get("claim", "") for item in self.knowledge_items]

        for i, claim in enumerate(self.claims):
            if not isinstance(claim, str):
                print(f"警告: 第{i}个声明不是字符串类型: {type(claim)}")
                self.claims[i] = str(claim)

        self.is_english = [self._is_english(claim) for claim in self.claims]

        # 初始化向量化器
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self._tokenize,
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1,
            stop_words='english',
            use_idf=True,
            sublinear_tf=True
        )

        # 向量化声明
        try:
            self.claim_vectors = self.vectorizer.fit_transform(self.claims)
        except Exception as e:
            print(f"向量化声明时出错: {e}")
            print("使用备用向量化策略...")
            self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
            self.claim_vectors = self.vectorizer.fit_transform(self.claims)

        self._save_to_cache(cache_path)

    def _is_english(self, text: str) -> bool:
        """
        根据字符分布检查文本是否以英文为主

        参数:
            text: 输入文本

        返回:
            是（英文）/否（非英文）
        """
        if not isinstance(text, str):
            return True  # 默认为英文

        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / max(len(text), 1) > 0.6

    def _tokenize(self, text: str) -> List[str]:
        """
        针对英文文本优化的自定义标记符号生成器

        参数:
            text: 输入文本

        返回:
            标记列表
        """
        if not isinstance(text, str):
            return []  # 返回空列表，避免错误

        # 转换为小写并分割非字母数字
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _save_to_cache(self, cache_path: str):
        """
        保存向量和嵌入缓存

        参数:
            cache_path: 缓存文件路径
        """
        cache_data = {
            'claims': self.claims,
            'vectorizer': self.vectorizer,
            'claim_vectors': self.claim_vectors,
            'is_english': self.is_english,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"保存向量和嵌入缓存至: {cache_path}")
        except Exception as e:
            print(f"存储缓存时错误: {e}")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        检索与给定查询相关的知识条目

        参数:
            query: 查询文本

        返回:
            检索到的知识条目列表
        """
        if self.vectorizer is None:
            print("错误: 向量化器未初始化，无法执行检索")
            return []

        if not self.claims or self.claim_vectors is None or self.claim_vectors.shape[0] == 0:
            print("警告: 索引为空，无法执行检索")
            return []

        try:
            query_vector = self.vectorizer.transform([query])

            similarities = cosine_similarity(query_vector, self.claim_vectors).flatten()

            if len(similarities) <= self.top_k:
                top_indices = np.argsort(similarities)[::-1]
            else:
                top_indices = np.argsort(similarities)[-self.top_k:][::-1]

            retrieved_items = []
            for idx in top_indices:
                if idx < len(self.knowledge_items) and similarities[idx] > 0:
                    item = {**self.knowledge_items[idx]}  # Copy the item
                    item["similarity_score"] = float(similarities[idx])  # Add similarity score
                    retrieved_items.append(item)

            return retrieved_items
        except Exception as e:
            print(f"检索时出错: {e}")
            return self._fallback_retrieve(query)

    def _fallback_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        备用检索方法，当向量检索失败时使用

        参数:
            query: 查询文本

        返回:
            检索到的知识条目列表
        """
        print("使用备用检索方法...")
        results = []
        query = query.lower()

        # 简单的文本匹配
        for item in self.knowledge_items:
            claim = item.get("claim", "").lower()
            # 计算简单的匹配分数
            # 1. 完全匹配
            if query == claim:
                score = 1.0
            # 2. 包含关系
            elif query in claim or claim in query:
                common_len = min(len(query), len(claim))
                max_len = max(len(query), len(claim))
                score = 0.7 * (common_len / max_len)
            # 3. 单词重叠
            else:
                query_words = set(query.split())
                claim_words = set(claim.split())
                common_words = query_words.intersection(claim_words)
                if common_words:
                    score = 0.5 * (len(common_words) / max(len(query_words), len(claim_words)))
                else:
                    score = 0

            if score > 0:
                result_item = {**item}
                result_item["similarity_score"] = score
                results.append(result_item)

        # 排序并返回top_k结果
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:self.top_k]

    def retrieve_multiple(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        从多个查询变量中检索并合并结果

        参数:
            queries: 查询变量 列表

        返回:
            检索项目的合并列表
        """
        if not queries:
            return []

        all_items = {}  # claim -> item mapping

        for query in queries:
            try:
                items = self.retrieve(query)

                for item in items:
                    claim = item.get("claim", "")

                    if claim not in all_items or item["similarity_score"] > all_items[claim]["similarity_score"]:
                        all_items[claim] = item
            except Exception as e:
                print(f"处理查询 '{query}' 时出错: {e}")
                continue

        result = list(all_items.values())
        result.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

        return result[:self.top_k]

    def clear_cache(self):
        """清除所有缓存的向量和嵌入"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.startswith("retriever_cache_"):
                    os.remove(os.path.join(self.cache_dir, filename))
            print(f"已清理在 '{self.cache_dir}' 中的缓存文件")
        except Exception as e:
            print(f"清理缓存出错: {e}")

    def rebuild_index(self):
        """强制重建知识库索引"""
        self._build_index(force_rebuild=True)