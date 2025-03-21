class Config:
    """配置类，存储系统所需的各项配置参数"""
    
    # 知识库路径
    KNOWLEDGE_BASE_PATH = "data/knowledge_base.jsonl"
    
    # 检索设置
    RETRIEVAL_TOP_K = 5  # 检索的相关文档数量

    # 缓存设置
    CACHE_DIR = "data/cache"  # 向量缓存目录
    
    # 模型设置
    LLM_MODEL_PATH = "llm/deepseek-r1"  # 本地LLM模型路径
    
    # 多模态设置（预留）
    IMAGE_MODEL_PATH = "your_future_image_model_path"  # 图像模型路径（预留）
    VIDEO_MODEL_PATH = "your_future_video_model_path"  # 视频模型路径（预留）