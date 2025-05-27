# CS310 NLP 项目总览

## 1. 项目简介
本项目聚焦于中英文 LLM 生成文本检测，涵盖数据预处理、监督式分类、零样本检测、频谱分析等多种方法，支持多领域 OOD 泛化测试。

---

## 2. 目录结构说明

```
Project/
├─ face2_zh_json/           # 中文 Qwen-2 数据集（生成与人工）
│  ├─ generated/zh_qwen2/   # 生成文本（news, webnovel, wiki）
│  └─ human/zh_unicode/     # 人工文本
├─ ghostbuster-data/        # 英文 Ghostbuster 原始数据
│  ├─ essay/ reuter/ wp/ other/
├─ ghostbuster-data_reformed/ # 统一格式的英文数据
│  ├─ essay/ reuter/ wp/
├─ fft_output/              # NLL 频谱分析结果
├─ fft_results/             # Fourier-GPT 相关结果
├─ FourierGPT/              # 频谱分析与零样本检测代码
│  ├─ run_fft.py run_nll.py ...
├─ local_model_Bert_Chinese/  # 本地中文BERT模型
├─ local_model_Bert_English/  # 本地英文BERT模型
├─ notebooks/               # 数据探索与可视化
├─ results_nll/             # 中文NLL结果
├─ results_train/           # 监督式训练结果
├─ results_test/            # 监督式OOD测试结果
├─ src/                     # 源代码
│  ├─ supervised/           # 监督式训练与评估
│  ├─ zero_shot/            # 零样本检测
│  └─ utils/                # 通用工具
├─ environment.yml          # Conda环境配置
├─ requirements.txt         # Python依赖
├─ README.md                # 项目说明
├─ Project_Overview.md      # 项目总览
└─ ...
```

---

## 3. 数据集说明

### 3.1 中文 Qwen-2 数据集
- 路径：`face2_zh_json/`
- 领域：News、Webnovel、Wiki
- 格式：JSON，每条含 prompt、result、label
- 用途：中英文对比、领域泛化

### 3.2 英文 Ghostbuster 数据集
- 路径：`ghostbuster-data/`、`ghostbuster-data_reformed/`
- 子集：essay、reuter、wp、other
- 格式：TXT/JSON/JSONL，含 text、label、domain
- 用途：英文检测、监督训练、OOD

---

## 4. 主要功能模块

### 4.1 监督式分类
- 支持 BERT、RoBERTa 等 Transformer encoder
- 训练/验证/测试集划分，支持领域混合与分开
- 评估指标：Accuracy、Precision、Recall、F1-score
- 训练脚本：`src/supervised/train_supervised.py`

### 4.2 零样本检测
- 支持 Fourier-GPT、GPT-Who 等方法
- 频谱分析与特征提取：`FourierGPT/`
- NLL 计算与分析：`run_nll.py`、`src/zero_shot/`

### 4.3 频谱分析
- 对 NLL 序列做 FFT，提取频域特征
- 结果存储于 `fft_output/`、`fft_results/`

### 4.4 数据探索与可视化
- Jupyter Notebooks 位于 `notebooks/`
- 支持特征分布、模型表现等可视化

---

## 5. 环境与运行

### 5.1 环境配置
- 推荐使用 Conda 环境：`environment.yml`
- 依赖包：`requirements.txt`
- 路径设置：
  - 服务器：`export PYTHONPATH=/home/liuj_lab/cse12213012/code/CS310-NLP-Project:$PYTHONPATH`
  - 本地：`export PYTHONPATH=/Users/3mker/Desktop/Sustech/Junior/NLP/Project:$PYTHONPATH`

### 5.2 训练与评测示例

```bash
# 中文BERT监督训练
python src/supervised/train_supervised.py --if_local true --batch_size 100 --data_path face2_zh_json --model_name bert-base-chinese --epochs 30

# 英文BERT监督训练
python src/supervised/train_supervised.py --if_local true --batch_size 100 --data_path ghostbuster-data --model_name bert-base-uncased --epochs 50
```

#### 主要参数说明
- `data_path`：数据集路径（如 face2_zh_json 或 ghostbuster-data）
- `data_type`：数据子集（如 ['news','webnovel','wiki']）
- `output_dir`：结果存储目录
- `model_name`：模型名（bert-base-chinese/bert-base-uncased）
- `epochs`：训练轮数
- `batch_size`：批大小
- `if_local`：是否加载本地模型

---

## 6. 结果与分析
- 监督式与零样本方法结果分别存于 `results_train/`、`results_test/`、`results_nll/`、`fft_results/`
- 支持 OOD 泛化测试与多领域对比
- 可视化分析见 `notebooks/`

---

## 7. 参考与扩展
- 支持多语言、多领域扩展
- 可集成更多零样本检测方法
- 详细流程与数据说明见 `Project_Overview.md`

---

## 8. 主要代码文件详细说明（src/）

### supervised/
- **train_supervised.py**：
  - 用于训练监督式文本检测模型（如BERT），支持中英文数据集，自动划分训练/验证集，保存最优模型，输出训练过程的各类指标曲线和混淆矩阵。
- **test_supervised.py**：
  - 用于加载已训练模型并在不同领域（OOD）上测试，输出各领域的准确率、F1、精确率、召回率及混淆矩阵。
- **download.py**：
  - 下载并保存BERT（中/英文）模型及分词器到本地，便于离线加载。
- **check_gpu.py**：
  - 检查当前PyTorch环境下可用GPU信息，输出GPU数量、名称和显存。
- **load_saved_model.py**：
  - 演示如何加载本地已保存的BERT模型和分词器。

### zero_shot/
- **download.py**：
  - 下载并保存Mistral-7B模型及分词器到本地，便于后续零样本检测和NLL计算。
- **detect_zero_shot.py**：
  - 零样本检测主脚本，支持DetectGPT、Fast-DetectGPT、FourierGPT、GPT-who等方法（可扩展），批量处理文本并保存检测分数。
- **batch_run_fft.py**：
  - 批量对NLL结果文件进行FFT变换，生成频谱特征，支持多种归一化和取值方式，适配中英文数据结构。
- **batch_run_pwh_cls.py**：
  - 批量运行Fourier-GPT的pairwise human判别脚本，对比模型与人类文本的频谱特征，输出分类结果。

### utils/
- **data_utils.py**：
  - 数据加载与预处理工具，支持中文Qwen-2和英文Ghostbuster数据集，自动合并生成/人工文本，统一格式输出。
- **util.py**：
  - BERT输入预处理函数，将prompt/result拼接并分词，适配训练/推理流程。
- **dataset.py**：
  - 自定义BertDataset类，兼容Huggingface Trainer的数据输入格式。

---

如需了解每个脚本的详细用法和参数，请参考各文件头部注释或直接阅读源码。
