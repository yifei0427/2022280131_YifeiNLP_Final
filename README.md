# 2022280131_中文诈骗对话欺诈检测与对抗攻击项目

本项目旨在实现中文诈骗对话检测模型，并对不同架构模型（基于 PyTorch 的 RoBERTa/TextCNN 和基于 TensorFlow 的 BiLSTM/TFBert）的鲁棒性进行评估，尤其关注 **Sememe+PSO** 对抗攻击的集成与效果。

## ⚙️ 核心文件与实验环境

本项目包含两个主要实验环境的 Notebook：

| 文件名 | 实验平台 | 核心模型 | 攻击方法 | 攻击结果 |
| :--- | :--- | :--- | :--- | :--- |
| **`NLP_Final.ipynb`** | **PyTorch** | TextCNN, RoBERTa | Sememe+PSO (尝试集成) | 集成可行性证明 |
| **`NLP_Final_Tensor.ipynb`** | **TensorFlow/Keras** | BiLSTM, TFBert | Synonym+PSO (简化实现) | 攻击成功率 0% |

### 📂 核心文件清单

- **`NLP_Final.ipynb`**: **(PyTorch)** 包含 TextCNN 和 RoBERTa 模型的训练、攻击集成尝试及评估的完整 Colab Notebook。
- **`NLP_Final_Tensor.ipynb`**: **(TensorFlow)** 包含 BiLSTM 和 TFBert 模型的训练、**简化的 Synonym+PSO 攻击实现**及评估的完整 Colab Notebook。
- **`textcnn_fraud_model.pt`**: PyTorch TextCNN 模型的权重文件。
- **`roberta_victim.py`**: PyTorch RoBERTa 目标模型适配器。
- **`data/`**: 包含 `test_results.csv` 等测试和评估数据。

### 📊 实验基线结果 (原始测试集准确率)

| 模型 | 平台 | 原始测试集准确率 |
| :--- | :--- | :--- |
| RoBERTa (PyTorch) | PyTorch | [TBD - 基于 NLP_Final.ipynb 结果] |
| TextCNN (PyTorch) | PyTorch | [TBD - 基于 NLP_Final.ipynb 结果] |
| **TFBert** (TensorFlow) | TensorFlow | **0.9984** |
| **BiLSTM** (Keras) | TensorFlow | **0.9965** |

### 📉 对抗攻击结果 (Synonym+PSO)

在 **`NLP_Final_Tensor.ipynb`** 中，我们对 BiLSTM 和 TFBert 模型进行了 **Synonym+PSO 攻击**。由于环境限制，我们采用了**简化的离散 PSO 逻辑**和**有限的同义词替换列表**，最终结果如下：

| 模型 | 攻击成功率 (ASR) | 平均改动率 | D\_adv 准确率 |
| :--- | :--- | :--- | :--- |
| BiLSTM | **0.0000** (0/50) | 0.0504 | 1.0000 |
| TFBert | **0.0000** (0/50) | 0.0040 | 1.0000 |

**攻击分析：** 简化的 Synonym+PSO 攻击未能成功生成对抗样本。分析表明，模型（尤其是 TFBert）对有限的同义词替换具有极强的鲁棒性。

## ⚠️ 对抗攻击框架引用 (Sememe+PSO)

本次实验的对抗攻击（Sememe+PSO）是基于 [Zang et al., 2020] 提出的框架进行集成和调试的。

本项目在两个 Notebook 中采用了不同的策略来应对该框架的复杂性：
1.  **`NLP_Final.ipynb` (PyTorch):** 尝试通过路径添加和类名查找，集成原仓库中的 `PSOAttack` 类，证明了系统的集成可行性。
2.  **`NLP_Final_Tensor.ipynb` (TensorFlow):** 鉴于原框架的复杂依赖（如义素知识库），我们在 Notebook 内部**从头实现了简化的离散 PSO 逻辑**，并结合自定义的同义词代理替换，以在 Colab 环境下实现算法的核心思想。

本项目集成的 Sememe+PSO 框架来源于以下 GitHub 仓库：

[**Word-level Textual Adversarial Attacking as Combinatorial Optimization**](https://github.com/thunlp/SememePSO-Attack)

**（原论文链接：[Word-level Textual Adversarial Attacking as Combinatorial Optimization]）**
