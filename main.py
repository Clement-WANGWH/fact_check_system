import argparse
import os
import json
from datetime import datetime
from config.config import Config
from utils.data_loader import KnowledgeBase
from models.text_processor import TextProcessor
from models.retriever import Retriever
from models.reasoner import Reasoner


def parse_arguments():
    """
    解析命令行参数

    返回:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description="事实核查系统")
    parser.add_argument('--query', type=str, help='要核查的声明')
    parser.add_argument('--kb_path', type=str, default=Config.KNOWLEDGE_BASE_PATH,
                        help='知识库JSONL文件路径')
    parser.add_argument('--image', type=str, help='要处理的图像路径（未来功能）')
    parser.add_argument('--video', type=str, help='要处理的视频路径（未来功能）')
    parser.add_argument('--force_rebuild', action='store_true',
                        help='强制重建向量索引')
    parser.add_argument('--clear_cache', action='store_true',
                        help='清除所有缓存的向量嵌入')
    parser.add_argument('--compare', action='store_true',
                        help='比较RAG增强和直接LLM推理方法')
    parser.add_argument('--output', type=str, help='比较结果输出文件路径')
    return parser.parse_args()


def run_comparison(query, text_processor, retriever, reasoner):
    """
    运行RAG增强和直接LLM推理的比较

    参数:
        query: 查询文本
        text_processor: 文本处理器实例
        retriever: 检索器实例
        reasoner: 推理器实例

    返回:
        比较结果
    """
    # 处理查询
    query_info = text_processor.process_for_retrieval(query)
    claim = query_info['claim']

    print(f"\n处理查询: {query}")
    print(f"提取声明: {claim}")

    # 方法1: 直接LLM推理
    print("\n=== 方法1: 使用直接LLM推理 ===")
    direct_result = reasoner.reason_direct(claim)
    print(f"直接推理结果 - 事实性: {direct_result['factual']}")

    # 方法2: RAG增强推理
    print("\n=== 方法2: 使用RAG检索增强 ===")
    expanded_queries = query_info['expanded_queries']
    retrieved_items = retriever.retrieve_multiple(expanded_queries)

    if retrieved_items:
        print(f"找到{len(retrieved_items)}个相关知识条目")
        for i, item in enumerate(retrieved_items[:3], 1):  # 只显示前3个
            print(f"{i}. 相似度: {item['similarity_score']:.4f}, 声明: {item['claim'][:50]}...")
    else:
        print("未找到相关知识条目")

    rag_result = reasoner.reason(claim, retrieved_items)
    print(f"RAG方法结果 - 事实性: {rag_result['factual']}, 答案: {rag_result.get('answer', 'N/A')}")

    return {
        "query": query,
        "claim": claim,
        "rag_result": rag_result,
        "direct_result": direct_result,
        "timestamp": datetime.now().isoformat()
    }


def display_comparison_result(comparison):
    """
    显示比较结果

    参数:
        comparison: 比较结果
    """
    print("\n===== 方法比较结果 =====")
    print(f"查询: {comparison['query']}")
    print(f"提取声明: {comparison['claim']}")

    print("\n方法1 (直接LLM推理):")
    print(f"事实性: {comparison['direct_result']['factual']}")

    print("\n方法2 (RAG增强):")
    print(f"事实性: {comparison['rag_result']['factual']}")
    if 'answer' in comparison['rag_result']:
        print(f"答案: {comparison['rag_result']['answer']}")

    is_factual_same = comparison['rag_result']['factual'] == comparison['direct_result']['factual']

    print("\n结果比较:")
    print(f"事实性判断是否一致: {'是' if is_factual_same else '否'}")

    print("===== 比较结束 =====")


def save_comparison_result(comparison, output_path):
    """
    保存比较结果到文件

    参数:
        comparison: 比较结果
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    simplified_result = {
        "query": comparison["query"],
        "claim": comparison["claim"],
        "rag_result": {
            "factual": comparison["rag_result"]["factual"],
            "answer": comparison["rag_result"].get("answer", "N/A")
        },
        "direct_result": {
            "factual": comparison["direct_result"]["factual"]
        },
        "timestamp": comparison["timestamp"]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_result, f, ensure_ascii=False, indent=2)

    print(f"比较结果已保存到: {output_path}")


def main():
    """事实核查系统主入口"""
    args = parse_arguments()

    # 初始化组件
    kb = KnowledgeBase(args.kb_path)
    knowledge_items = kb.get_all_items()

    retriever = Retriever(
        knowledge_items=knowledge_items,
        top_k=Config.RETRIEVAL_TOP_K,
        cache_dir=Config.CACHE_DIR,
        force_rebuild=args.force_rebuild
    )

    if args.clear_cache:
        retriever.clear_cache()
        print("缓存已清除，退出程序")
        return

    text_processor = TextProcessor()
    reasoner = Reasoner(Config.LLM_MODEL_PATH)

    # 处理用户查询
    if args.query:
        if args.compare:
            # 运行比较模式
            comparison = run_comparison(args.query, text_processor, retriever, reasoner)

            display_comparison_result(comparison)

            if args.output:
                save_comparison_result(comparison, args.output)
        else:
            query_info = text_processor.process_for_retrieval(args.query)

            print(f"\n处理查询: {args.query}")
            print(f"提取声明: {query_info['claim']}")
            print(f"关键词: {', '.join([k for k, _ in query_info['keywords']])}")

            expanded_queries = query_info['expanded_queries']
            print(f"扩展查询: {expanded_queries}")

            retrieved_items = retriever.retrieve_multiple(expanded_queries)

            if retrieved_items:
                print(f"\n找到{len(retrieved_items)}个相关知识条目:")
                for i, item in enumerate(retrieved_items, 1):
                    print(f"{i}. 相似度: {item['similarity_score']:.4f}, 声明: {item['claim']}")
            else:
                print("\n未找到相关知识条目")

            result = reasoner.reason(query_info['claim'], retrieved_items)

            print("\n===== 事实核查结果 =====")
            print(f"查询: {args.query}")
            print(f"事实性: {result['factual']}")
            if 'answer' in result:
                print(f"答案: {result['answer']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"解释: {result['explanation']}")
            print("=======================\n")
    elif args.image:
        print("图像处理功能尚未实现，未来版本将支持。")
    elif args.video:
        print("视频处理功能尚未实现，未来版本将支持。")
    else:
        print("未提供查询。使用 --query 提供要核查的声明，或使用 --image/--video 提供多模态输入（未来功能）。")
        print("使用 --compare 开启RAG增强和直接LLM推理的比较模式。")


if __name__ == "__main__":
    main()
