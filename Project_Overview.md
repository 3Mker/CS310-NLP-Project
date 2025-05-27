# CS310 NLP Project Overview

## 1. 项目完成流程

1. 数据收集与预处理

   - English Ghostbuster 数据：

     - 原始数据位于 `ghostbuster-data/`，包含多种来源（essay、reuter、wp、other）。
     - 使用 `ghostbuster-data_reformed/` 中的重组格式，将 essay、reuter、wp 三个子集划分成统一的训练/验证/测试集。

   - 中文 Qwen-2 数据：

     - 位于 `face2_zh_json/`，分为 `generated/zh_qwen2/`（News、Webnovel、Wiki 三个领域）和 `human/zh_unicode/`（对应的人工文本）。
     - 合并生成与人工文本，对应同一篇章，转换为统一的 JSONL 或 CSV 格式。

2. 监督式分类模型

   - 构建二分类管道，模型可选 BERT、RoBERTa 等 Transformer encoder。
   - 训练流程：

     - 划分训练/验证/测试集（可按领域分开或混合训练）。
     - 调整超参数（学习率、batch size、epoch 数）。
     - 保存最优模型。

   - 评估指标：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1-score。

3. 零样本检测方法

   - 选择GPT-Who/Fourier-GPT
   - 实现或调用官方实现，对同一批文本进行零样本检测。
   - 评估同样的指标，并与监督式模型对比。
   - 复现完成后可以把GPT2模型换为别的模型

4. 域外泛化测试（OOD）

   - 在一个领域（如 News）训练，在其他领域（Webnovel、Wiki）测试。
   - 比较各模型在 OOD 设置下的性能差异。

5. 结果对比与分析

   - 汇总监督式与零样本方法在不同数据组合下的性能表格和可视化图表。
   - 讨论模型优缺点及领域迁移性能。

6. 可选扩展

   - 多语言或多领域统一模型（如 XLM-R）。
   - 增加更多零样本检测方法。

---

## 2. 项目目录结构示例

```text
Project/
├─ face2_zh_json/         # 中文 Qwen-2 数据集
│  ├─ generated/          # 生成的文本
│  │  └─ zh_qwen2/
│  └─ human/              # 人工文本
│     └─ zh_unicode/
├─ ghostbuster-data/      # 原始 Ghostbuster 英文数据集
│  ├─ essay/
│  ├─ reuter/
│  ├─ wp/
│  └─ other/
├─ ghostbuster-data_reformed/  # NLL Ghostbuster 数据集
│  ├─ essay/
│  ├─ reuter/
│  └─ wp/
|─ fft_output/
|  |─ cn/                 # 使用mistral模型生成的中文数据nll的fft转换结果
|- fft_results/
|  |-pwh_cls/             # 使用Fourier-GPT生成的结果
├─ results_train/               # 结果文件夹
│  ├─ supervised/         # 监督式模型训练结果
│- results_test/
|- |-supervised/          # 监督式模型OOD测试结果
│- results_nll/           # 中文数据的nll
├─ src/                   # 源代码
│  ├─ supervised/         # 监督式训练与评估脚本
│  ├─ zero_shot/          # 零样本检测实现
│  └─ utils/              # 数据处理与通用函数
├─ notebooks/             # 数据探索与可视化
├─ README.md              # 项目说明文件
├─ LICENSE                # 许可证
└─ Project_Overview.md    # 项目概览
```

---

## 3. 三个数据集说明

### 3.1 Ghostbuster English 数据集

- 来源：Verma et al. (2023) 提供的 Ghostbuster 仓库。
- 结构：原始目录 `ghostbuster-data/` 下包含 `essay/`、`reuter/`、`wp/` 等子集，以及额外的 `other/` 源数据。
- 内容：每个子集内有 LLM 生成文本（如 Claude、GPT 系列）与对应的人类撰写文本，格式多为 TXT 或 JSON。
- 用途：监督式模型训练、OOD 测试。

### 3.2 Chinese Qwen-2 数据集

- 来源：Qwen-2 在新闻（News）、维基（Wiki）、网络小说（Webnovel）三大领域上生成的中文文本。
- 结构：

  - `face2_zh_json/generated/zh_qwen2/`：`news-zh.qwen2-72b-base.json`、`wiki-zh.qwen2-72b-base.json`、`webnovel.qwen2-72b-base.json`。
  - `face2_zh_json/human/zh_unicode/`：对应的 `news-zh.json`、`wiki-zh.json`、`webnovel.json`。

- 内容：每条记录包含文本及相应字段（如 id、source、text）。
- 用途：中英文对比实验、多领域 OOD 测试。

### 3.3 Ghostbuster 数据重组版本

- 存放于 `ghostbuster-data_reformed/`。
- 结构：仅保留 `essay/`、`reuter/`、`wp/` 三个核心子集，统一命名与文件格式（建议 JSONL）。
- 内容：每条记录包含 `text`、`label`（0=human, 1=LLM）、`domain` 字段。
- 用途：简化监督式训练管道，快速加载与批处理。

## 4. 环境相关说明

### 4.1 环境路径问题

- 要设置类似 `export PYTHONPATH=/home/liuj_lab/cse12213012/code/CS310-NLP-Project:$PYTHONPATH` 的才能正确识别路径
- 如果是本地则类似 `export PYTHONPATH=/Users/3mker/Desktop/Sustech/Junior/NLP/Project:$PYTHONPATH`

## 5. 运行相关参数

python src/supervised/train_supervised.py --if_local true --batch_size 100 --data_path ghostbuster-data --model_name bert-base-uncased --epochs 50

python src/supervised/train_supervised.py --if_local true --batch_size 100 --data_path face2_zh_json --model_name bert-base-chinese --epochs 30

### 5.1 data_path

- 代表使用哪个数据集，可以选择"face2_zh_json"或"ghostbuster-data"

### 5.2 data_type

- 具体选择数据集中的那些类型的数据，比如face2_zh_json就有 ['news','webnovel' 'wiki'] 

### 5.3 output_dir

- 结果存储，不建议修改

### 5.4 model_name

- 用于选择模型，目前有‘bert-base-chinese' 和‘bert-base-uncased'两种选择

### 5.5 epochs

- 训练的轮数，默认是10

### 5.6 batch_size

- 每次训练的batch大小，默认是16

### 5.7 if_local

- 是否加载本地模型，推荐是使用本地