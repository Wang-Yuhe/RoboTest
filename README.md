# RoboTest: 多模态九宫格视觉定位

RoboTest 是一个面向课程实验的多模态定位项目。系统输入一张由合成图形或真实照片组成的 3 x 3 网格图像，以及一条中文指令，输出目标格子、离散点击动作和可视化鼠标轨迹。

> 项目用于多模态理解、视觉定位和交互轨迹建模教学，不用于绕过第三方验证码或安全系统。

## 主要功能

- 生成 `easy`、`medium` 和 `hard` 三种难度的合成九宫格数据。
- 支持规则颜色基线、模板图文匹配和神经网络定位模型。
- 支持“点击单个目标”和“点击所有同类目标”两种任务。
- 将预测转换为 `move_to_cell / click / done` 动作序列，并生成连续鼠标轨迹。
- 提供 Streamlit 交互演示、静态 HTML 演示、训练、评估和单图推理脚本。
- 真实照片 click-all 实验使用 source-disjoint 数据划分，减少原始图片跨 split 泄漏。

## 环境要求

- Python 3.10+
- macOS、Linux 或 Windows
- CPU 可运行最小演示；大规模训练建议使用 CUDA GPU

可以使用 Python `venv` 或 Conda 创建独立环境。

### 使用 venv

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-minimal.txt
```

### 使用 Conda

需要先安装 Anaconda 或 Miniconda：

```bash
conda create -n robotest python=3.10 -y
conda activate robotest
python -m pip install --upgrade pip
python -m pip install -r requirements-minimal.txt
```

后续重新进入项目时，执行 `conda activate robotest` 即可激活环境。

CLIP、Transformers、FiftyOne 和 ScreenSpot-Pro 适配实验需要完整依赖：

```bash
python -m pip install -r requirements.txt
```

## 快速验收

以下流程不需要下载真实照片或预训练权重：

```bash
# 1. 生成合成数据
python scripts/generate_dataset.py --num-samples 120 --seed 7 --difficulty medium

# 2. 运行模板基线并生成 outputs/prediction.png
python scripts/predict.py --mode template

# 3. 批量评估
python scripts/evaluate.py --mode template --split val

# 4. 生成 outputs/demo.html
python scripts/make_demo_html.py --mode template
```

`outputs/demo.html` 是预先生成的静态结果页，不是 `app.py` 提供的交互页面。右键在浏览器中打开时通常使用 `file://` 本地地址，因此没有 8501 端口；如果通过 VS Code 的 Live Server 打开，端口则由该扩展决定。静态页的模式在生成时通过 `--mode` 选定，页面内不提供模型切换控件。

使用训练好的单目标模型生成静态页：

```bash
python scripts/make_demo_html.py \
  --mode model \
  --checkpoint outputs/model.pt
```

要使用完整交互页面，需要通过 Streamlit 启动 `app.py`。`app.py` 本身没有设置端口，下面的命令显式将服务固定到 8501：

```bash
streamlit run app.py --server.port 8501
```

启动后访问 `http://localhost:8501`。如果 8501 已被占用，可将命令和访问地址中的端口同时改为 8502，并以终端输出的 `Local URL` 为准。

Streamlit 页面与所需资源如下：

- `单目标定位`：可在模板图文模型、规则颜色基线和神经网络模型之间切换；没有 `outputs/model.pt` 时，神经网络选项使用未训练权重。
- `点击所有同类目标`：侧边栏提供 checkpoint、数据 split、阈值、解码策略、设备、prompt planner 和 VLM 对照选项；该页需要额外的 click-all 数据集和 action checkpoint。
- `官方服务接入`：提供合规的 Turnstile 测试接入。

如需调用可选的 prompt 改写或 VLM 对照功能，通过环境变量提供密钥，不要将密钥写入仓库：

```bash
export DEEPSEEK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

## 测试

项目测试使用 Python 标准库 `unittest`：

```bash
python -m unittest discover -s tests -v
```

## 训练与推理

训练单目标定位模型：

```bash
python scripts/train.py \
  --data-dir data/synthetic_captcha \
  --output outputs/model.pt \
  --epochs 10 \
  --batch-size 64 \
  --model-size attn

python scripts/evaluate.py \
  --mode model \
  --checkpoint outputs/model.pt \
  --split val
```

对自定义九宫格图片推理：

```bash
python scripts/predict_image.py \
  --image path/to/grid.png \
  --prompt "请点击汽车" \
  --checkpoint outputs/model.pt \
  --output outputs/custom_prediction.png
```

真实照片 click-all 数据构建、ResNet18 训练和完整 split 评估命令见 [AutoDL 训练指南](docs/action_sequence_autodl.md)。真实照片的来源、授权和下载方式见 [数据源说明](docs/data_sources.md)。

## 代表性结果

以下结果来自同一套 source-disjoint paired test split，共 2,000 个真实图片九宫格样本：

| 方法 | Cell Exact Match | Cell Precision | Cell Recall | Click Order Accuracy |
|---|---:|---:|---:|---:|
| 专训 ResNet18，clean 数据 | **0.4600** | **0.8677** | **0.8750** | **0.4600** |
| 专训 ResNet18，dirty 数据 | 0.4505 | 0.8766 | 0.8632 | 0.4505 |
| Qwen3-VL-Flash，clean 数据 | 0.1475 | 0.6998 | 0.6081 | 0.1475 |

这些数字表明专训模型在固定类别 click-all 任务上更有优势，不代表通用开放世界理解能力。指标定义、实验限制和失败案例见 [实验报告 PDF](docs/robotest_experiment_report_updated.pdf)，机器可读结果见 [action_sequence_benchmark_20260708.json](docs/results/action_sequence_benchmark_20260708.json)。

## 仓库结构

```text
.
├── app.py                         # Streamlit 交互演示
├── brief.md                      # 项目阶段性摘要
├── docs/                         # 报告、实验记录与数据说明
├── scripts/                      # 数据、训练、评估和推理入口
├── src/multimodal_captcha/       # 核心 Python 实现
├── tests/                        # 单元与脚本集成测试
├── requirements-minimal.txt      # 最小运行依赖
└── requirements.txt              # 完整实验依赖
```

`data/`、`outputs/` 和模型 checkpoint 均为本地生成或下载内容，已在 `.gitignore` 中排除，不随源码仓库交付。仓库内可直接查阅的交付物包括：

- [课程实验报告 PDF](docs/robotest_experiment_report_updated.pdf)
- [LaTeX 报告源文件](docs/robotest_experiment_report.tex)
- [机器可读实验指标](docs/results/action_sequence_benchmark_20260708.json)
- [完整环境与数据复现指南](docs/teammate_setup.md)

## 复现注意事项

- 默认合成图像不包含文字标签；颜色和类别标注保存在 `manifest.jsonl` 中。`--debug-labels` 仅用于人工检查。
- 随机过程应显式传入 `--seed`；报告中的完整真实照片实验需要额外数据和 checkpoint。
- 已发布结果使用的历史 ResNet18 训练中，冻结 backbone 时 BatchNorm running statistics 仍会更新。当前代码已修复为严格冻结，因此重跑结果可能与报告中的历史基线不同。
