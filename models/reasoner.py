from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

class Reasoner:
    """推理模块，基于检索结果进行事实推理"""
    
    def __init__(self, llm_model_path: str):
        """
        初始化推理模块
        
        参数:
            llm_model_path: 本地LLM模型路径
        """
        self.llm_model_path = llm_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"初始化推理模块，使用模型：{llm_model_path}，设备：{self.device}")
        
        # 实际加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        ).eval()
        
    def generate_prompt(self, query: str, retrieved_items: List[Dict[str, Any]]) -> str:
        """
        根据查询和检索项生成LLM提示
        
        参数:
            query: 输入查询
            retrieved_items: 检索到的知识条目列表
            
        返回:
            格式化的提示字符串
        """

        prompt = f"请判断以下声明是否属实: \"{query}\"\n\n参考信息:\n"
        
        for i, item in enumerate(retrieved_items, 1):
            prompt += f"{i}. 声明: \"{item['claim']}\", 标签: {item['label']}, 答案: {item['answer']}"
            if 'similarity_score' in item:
                prompt += f", 相似度: {item['similarity_score']:.4f}"
            prompt += "\n"
            
        prompt += "\n请先仔细思考，分析参考信息与查询声明的关系，然后判断该声明是否属实，并给出详细理由，必须使用中文回答。"
        prompt += "\n请以<think>开始你的思考过程，并在</think>结束。"
        prompt += "\n然后按以下格式回答：\n事实性: [是/否/不确定]\n答案: [明确答案]\n理由: [详细解释]"
        
        return prompt

    def generate_direct_prompt(self, query: str) -> str:
        """
        生成直接LLM推理的提示（不使用RAG检索结果）

        参数:
            query: 输入查询

        返回:
            格式化的提示字符串
        """
        prompt = f"请判断以下声明是否属实: \"{query}\"\n\n"
        prompt += "请先仔细思考，分析该声明的合理性和可能性，然后判断该声明是否属实，并给出详细理由。"
        prompt += "\n请以<think>开始你的思考过程，并在</think>结束。"
        prompt += "\n然后按以下格式回答：\n事实性: [是/否/不确定]\n答案: [明确答案]\n理由: [详细解释]"

        return prompt
    
    def reason(self, query: str, retrieved_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        基于查询和检索项进行推理
        
        参数:
            query: 输入查询
            retrieved_items: 检索到的知识条目列表
            
        返回:
            推理结果
        """
        prompt = self.generate_prompt(query, retrieved_items)
        
        # 确保模型开始输出带有<think>标记
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 根据DeepSeek-R1-Distill建议进行生成
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.1,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return self._parse_response(query, response)

    def reason_direct(self, query: str) -> Dict[str, Any]:
        """
        直接基于查询进行推理（不使用RAG）

        参数:
            query: 输入查询

        返回:
            推理结果
        """
        prompt = self.generate_direct_prompt(query)

        # 确保模型开始输出带有<think>标记
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 根据DeepSeek-R1-Distill建议进行生成
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.1,
            do_sample=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return self._parse_response(query, response)
        
    def _parse_response(self, query: str, raw_response: str) -> Dict[str, Any]:
        """
        解析模型原始输出到标准格式
        
        参数:
            query: 原始查询
            raw_response: 模型原始输出
            
        返回:
            解析后的结构化输出
        """
        # 提取思考过程（如果存在）
        thinking = ""
        thinking_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
        
        # 提取事实性判断
        factual = None
        factual_match = re.search(r'事实性:\s*[（\(]?([是否不确定]+)[）\)]?', raw_response)
        if factual_match:
            fact_text = factual_match.group(1).strip()
            if '是' in fact_text:
                factual = True
            elif '否' in fact_text:
                factual = False
            else:
                factual = None
        
        # 提取答案
        answer = ""
        answer_match = re.search(r'答案:\s*(.*?)(?:\n|$)', raw_response)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # 提取理由
        explanation = ""
        reason_match = re.search(r'理由:\s*(.*)', raw_response, re.DOTALL)
        if reason_match:
            explanation = reason_match.group(1).strip()
        elif thinking:
            # 如果没有找到明确的理由但有思考过程，使用思考过程作为解释
            explanation = "基于模型分析: " + thinking
        else:
            # 如果既没有找到理由也没有思考过程，使用整个响应
            explanation = raw_response

        confidence = 0.9 if factual is not None else 0.5

        result = {
            "query": query,
            "factual": factual,
            "answer": answer,
            "confidence": confidence,
            "explanation": explanation
        }

        if thinking:
            result["thinking_process"] = thinking
            
        return result
