
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np

# *** 修正路径：使用绝对路径确保加载成功 ***
MODEL_PATH = '/content/2022280131_YifeiNLP_Final/roberta_fraud_detection' 
ROBERTA_BASE = "hfl/chinese-roberta-wwm-ext-large"
MAX_LEN = 256

class RoBERTaVictimModel:
    def __init__(self, device):
        self.device = device
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE)
        
        # 2. 加载微调后的模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, # 现在使用绝对路径
            num_labels=2, 
        ).to(self.device)
        self.model.eval()

    def get_prob(self, texts, labels):
        # 核心函数：返回文本在目标标签上的概率 (对齐 Sememe+PSO 的接口)
        inputs = self.tokenizer(
            texts, 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_LEN, 
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=1)
        
        target_probs = []
        for prob, label in zip(probs, labels):
            target_probs.append(prob[label].item())
            
        return np.array(target_probs)
