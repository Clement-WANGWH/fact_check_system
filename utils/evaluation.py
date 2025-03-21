from typing import List, Dict, Any

class Evaluator:
    """评估工具，用于评估系统性能"""
    
    def __init__(self):
        """初始化评估器"""
        pass
        
    def evaluate(self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估事实核查系统
        
        参数:
            predictions: 预测结果列表
            ground_truth: 真实标签列表
            
        返回:
            评估指标字典
        """
        matched_items = []
        for pred in predictions:
            query = pred["query"]

            gt_match = next((gt for gt in ground_truth if gt["claim"] == query), None)
            
            if gt_match:
                matched_items.append((pred, gt_match))
        
        # 计算指标
        total = len(matched_items)
        if total == 0:
            return {
                "准确率": 0.0,
                "精确率": 0.0,
                "召回率": 0.0,
                "F1分数": 0.0
            }
            
        correct = sum(1 for pred, gt in matched_items 
                     if (pred["factual"] and gt["label"] == "Factual") or 
                     (not pred["factual"] and gt["label"] != "Factual"))

        accuracy = correct / total if total > 0 else 0.0

        true_pos = sum(1 for pred, gt in matched_items 
                      if pred["factual"] and gt["label"] == "Factual")
        
        false_pos = sum(1 for pred, gt in matched_items 
                       if pred["factual"] and gt["label"] != "Factual")
        
        false_neg = sum(1 for pred, gt in matched_items 
                       if not pred["factual"] and gt["label"] == "Factual")
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "准确率": accuracy,
            "精确率": precision,
            "召回率": recall,
            "F1分数": f1
        }
