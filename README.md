# 2022280131_中文诈骗对话欺诈检测与对抗攻击项目

本项目实现了基于 RoBERTa 和 TextCNN 的中文诈骗对话检测模型，并进行了 Sememe+PSO 对抗攻击的集成与评估。

## ⚙️ 核心文件与结果

- **`NLP_Final.ipynb`**: 包含所有实验代码、数据加载、模型训练、攻击流程和评估的完整 Colab Notebook。
- **`textcnn_fraud_model.pt`**: 训练完成的 TextCNN 模型权重。
- **`roberta_victim.py`**: 自定义 RoBERTa 目标模型适配器。
- **`data/`**: 包含 `test_results.csv` 等测试和评估数据。

## ⚠️ 对抗攻击框架引用 (Sememe+PSO)

由于本次实验的对抗攻击（Sememe+PSO）是基于 [Zang et al., 2020] 提出的框架进行集成和调试的，为避免仓库体积过大，我们引用原仓库而非完整克隆。

本项目在 Colab 环境中尝试集成的 Sememe+PSO 框架来源于以下 GitHub 仓库：

[**Word-level Textual Adversarial Attacking as Combinatorial Optimization**](https://github.com/thunlp/SememePSO-Attack)

**（原论文链接：[Word-level Textual Adversarial Attacking as Combinatorial Optimization]）**

在 `NLP_Final.ipynb` 中，我们通过路径添加和类名查找，尝试实例化了该框架中的 `PSOAttack` 类，证明了系统的集成可行性。
