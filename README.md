# Multimodal CAPTCHA Grounding

一个可运行的多模态数据处理课程项目：构建合成与真实图片 9 宫格图文任务，用专训模型或多模态大模型根据中文指令定位目标格，并生成离散点击动作和鼠标轨迹。

## 项目目标

输入：

- 一张合成或真实图片组成的 9 宫格任务图
- 一条中文任务描述，例如 `请点击红色消防栓` 或 `请点击所有汽车`

输出：

- 单目标格或所有同类目标格
- `move_to_cell / click / done` 动作序列
- 可视化点击结果与模拟鼠标运动轨迹

项目面向多模态理解、视觉定位和交互轨迹建模教学。真实图片实验使用项目自建九宫格任务，并通过 source-disjoint split 控制数据泄露；不用于绕过第三方验证码或安全系统。

## 快速开始

队友第一次 clone 后，推荐先跑最小 Demo，不需要下载真实照片数据：

```bash
git clone git@github.com:Wang-Yuhe/RoboTest.git
cd RoboTest
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-minimal.txt
python3 scripts/generate_dataset.py --num-samples 600 --difficulty medium
python3 scripts/train.py --epochs 2 --batch-size 64 --progress-every 5
python3 scripts/evaluate.py --mode template
streamlit run app.py
```

如果只想看未训练的完整流程，可以直接运行：

```bash
python3 scripts/predict.py
```

如果本机没有安装 Streamlit，可以生成静态网页 Demo：

```bash
python3 scripts/make_demo_html.py
```

然后打开 `outputs/demo.html`。

完整队友环境、真实照片数据复现、模型权重分发方式见 [docs/teammate_setup.md](docs/teammate_setup.md)。

如果队友不想重新训练，可以下载你发布的 `photo_model_100cls_attn_artifacts.tar.gz`，解压后把 checkpoint 放进 `outputs/`：

```bash
tar -xzf photo_model_100cls_attn_artifacts.tar.gz
mkdir -p outputs
cp photo_model_100cls_attn/photo_model_100cls_attn.pt outputs/
```

之后就可以直接用现有 checkpoint 做自定义九宫格推理：

```bash
python3 scripts/predict_image.py \
  --image path/to/your_grid.png \
  --prompt "请点击汽车" \
  --checkpoint outputs/photo_model_100cls_attn.pt \
  --output outputs/custom_prediction.png
```

## Web 展示

Streamlit 页面包含单目标定位、点击所有同类目标和 Cloudflare Turnstile 合规测试接入三个标签页：

```bash
python3 -m pip install -r requirements-minimal.txt
streamlit run app.py
```

单目标标签页可以自动生成合成样本。点击所有同类目标标签页需要先按下文命令准备真实图片数据和 action checkpoint，再在侧边栏选择对应路径。页面支持手工修改 prompt、结构化颜色/位置/大小 planner、DeepSeek prompt 改写，以及 Qwen-VL 单样本对比。

API key 只通过密码输入框或环境变量传入，不要写入源码：

```bash
export DEEPSEEK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

Windows PowerShell 对应使用 `$env:DEEPSEEK_API_KEY="..."` 和 `$env:DASHSCOPE_API_KEY="..."`。

## 目录结构

```text
.
├── app.py                         # Streamlit 可视化演示
├── data/                          # 生成的数据集
├── docs/
│   ├── robotest_experiment_report.tex # 实验报告 LaTeX 源文件
│   ├── robotest_experiment_report_updated.pdf # 最新编译报告
│   └── report_assets/             # 数据、失败案例和 Web 截图
├── outputs/                       # 模型权重和预测图
├── scripts/
│   ├── evaluate.py                # 批量评估并保存失败案例
│   ├── evaluate_action_sequence.py # click-all 评估、阈值选择和失败图
│   ├── evaluate_vlm_action_sequence.py # Qwen-VL 对照评测
│   ├── build_photo_action_dataset.py # source-disjoint 真实图片 action 数据
│   ├── build_photo_grid_dataset.py # 从真实照片目录拼 9 宫格
│   ├── download_openimages_subset.py # 下载 Open Images 子集
│   ├── generate_dataset.py        # 生成合成图文对
│   ├── make_demo_html.py          # 无需 Streamlit 的静态网页 Demo
│   ├── predict_image.py           # 对自定义图片和文本指令做预测
│   ├── train.py                   # 训练多模态定位模型
│   ├── train_action_sequence.py   # 训练 click-all cell selector
│   └── predict.py                 # 命令行预测示例
└── src/multimodal_captcha/
    ├── baseline.py                # 规则颜色基线
    ├── template_matcher.py        # 模板图文匹配模型
    ├── generator.py               # 图像和标注生成
    ├── dataset.py                 # PyTorch Dataset
    ├── model.py                   # 图文融合模型
    ├── action_sequence.py         # 动作 token 编解码与指标
    ├── prompt_rewriter.py         # DeepSeek/规则 prompt 改写
    ├── query_planner.py           # 颜色、位置和大小属性规划
    ├── trajectory.py              # 鼠标轨迹生成
    └── visualize.py               # 可视化工具
```

## 技术路线

1. 合成 9 宫格验证图片，每格绘制一个彩色物体。
2. 生成中文任务描述和监督标签。
3. 数据集支持 `easy / medium / hard` 三档难度。
4. 主 Demo 使用模板图文匹配模型：文本解析颜色和物体类别，图像侧做颜色匹配与形状模板匹配。
5. 对照基线包括规则颜色基线、模板图文模型和对象感知 CNN 神经模型。
6. 根据预测格子中心生成类人平滑鼠标轨迹。

## 难度设置

- `easy`：每张图 9 种颜色唯一，指令为“请点击黄色自行车”，颜色基线即可解决。
- `medium`：颜色会重复，物体类别唯一，指令仍包含颜色和类别，需要同时利用颜色和形状。
- `hard`：指令只包含物体类别，例如“请点击自行车”，必须依赖视觉形状匹配。

## 常用命令

```bash
# 生成 medium 默认实验数据
python3 scripts/generate_dataset.py --num-samples 600 --seed 41 --difficulty medium

# 生成不同难度数据
python3 scripts/generate_dataset.py --output-dir data/easy --difficulty easy
python3 scripts/generate_dataset.py --output-dir data/medium --difficulty medium
python3 scripts/generate_dataset.py --output-dir data/hard --difficulty hard

# 生成更困难的真实照片九宫格：随机裁剪、旋转、颜色扰动、噪声和模糊
python3 scripts/build_photo_grid_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_grid_hard_aug \
  --num-samples 800 \
  --hard-augment

# 一键运行 hard 测试
bash scripts/run_hard_test.sh

# 从真实照片目录构造九宫格数据集
python3 scripts/build_photo_grid_dataset.py --photo-root data/photo_objects --output-dir data/photo_grid

# 运行真实照片九宫格实验
bash scripts/run_photo_grid_pipeline.sh data/photo_objects data/photo_grid 300

# 自动下载 Open Images、裁剪物体、拼九宫格、训练并评估
bash scripts/run_openimages_auto_pipeline.sh

# 扩展下载更多真实照片类别，目标约 100 类、每类 100 张
bash scripts/run_expand_photo_classes.sh data/openimages_broad_raw data/photo_objects 20000 100 broad 100

# 如果 broad validation 仍然不够，用 train split 优先补齐最接近 100 张的类别
bash scripts/run_fill_photo_gaps.sh data/photo_objects data/openimages_gap_fill_raw 100 100 50000 2 train

# 查看真实照片类别覆盖情况
python3 scripts/report_photo_objects.py --photo-root data/photo_objects --min-images 100

# 用更多类别构造大规模真实照片九宫格
python3 scripts/build_photo_grid_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_grid_100cls \
  --num-samples 10000 \
  --min-images-per-class 100 \
  --max-classes 120 \
  --hard-augment

# 生成带格子文字标签的调试数据
python3 scripts/generate_dataset.py --num-samples 100 --debug-labels

# 训练
python3 scripts/train.py --epochs 10 --batch-size 64

# 训练真实照片模型
python3 scripts/train.py --data-dir data/photo_grid --output outputs/photo_model.pt --epochs 10 --aux-weight 0.7

# 泛化增强训练：更强残差 CNN + 数据增强 + 最佳 checkpoint
python3 scripts/train.py \
  --data-dir data/photo_grid \
  --output outputs/photo_model_v2.pt \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --aux-weight 0.7 \
  --patience 6 \
  --model-size small

# 如果机器性能允许，可以使用更大的 base 模型
python3 scripts/train.py \
  --data-dir data/photo_grid \
  --output outputs/photo_model_base.pt \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001 \
  --aux-weight 0.7 \
  --patience 8 \
  --model-size base

# 更强的注意力融合模型：残差 CNN + 9 格 Transformer + 图文交互特征
python3 scripts/train.py \
  --data-dir data/photo_grid_hard_aug \
  --output outputs/photo_model_attn.pt \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001 \
  --aux-weight 0.7 \
  --patience 8 \
  --model-size attn

# 100 类照片数据上的更强模型训练
python3 scripts/train.py \
  --data-dir data/photo_grid_100cls \
  --output outputs/photo_model_100cls_attn.pt \
  --epochs 40 \
  --batch-size 64 \
  --lr 0.001 \
  --aux-weight 0.7 \
  --patience 10 \
  --model-size attn

# 快速生成 HTML 看效果
python3 scripts/make_demo_html.py \
  --data-dir data/photo_grid_100cls \
  --mode model \
  --checkpoint outputs/photo_model_100cls_attn.pt \
  --output outputs/photo_demo_100cls.html \
  --num-examples 8

# 模板图文模型预测并保存预测图
python3 scripts/predict.py --mode template

# 对自定义图片和指令做预测
python3 scripts/predict_image.py --image data/synthetic_captcha/images/sample_00000.png --prompt "请点击黄色自行车"

# 规则颜色基线预测
python3 scripts/predict.py --mode baseline

# 使用神经模型预测
python3 scripts/predict.py --mode model --checkpoint outputs/model.pt

# 批量评估并保存指标
python3 scripts/evaluate.py --mode baseline
python3 scripts/evaluate.py --mode template
python3 scripts/evaluate.py --mode model --checkpoint outputs/model.pt
```

## 动作序列扩展

当前 `dev` 分支新增了“点击所有同类目标”的动作序列训练任务。模型输入九宫格图片和中文指令，例如 `请点击所有汽车`，先预测 9 个格子中哪些需要点击，再转换成离散动作 token：`move_to_cell / click / done`。连续鼠标轨迹仍可由现有 `trajectory.py` 根据动作序列渲染。

合成数据只适合 sanity check。正式实验应优先使用真实图片数据集，并先按源图片文件划分 train/val/test，避免同一张照片泄露到不同 split：

```bash
python3 scripts/build_photo_action_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_action_click_all \
  --num-train 10000 \
  --num-val 2000 \
  --num-test 2000 \
  --min-images-per-class 50 \
  --max-classes 80 \
  --image-size 192 \
  --min-targets 2 \
  --max-targets 4 \
  --balanced-targets \
  --hard-augment
```

真实图片训练：

```bash
python3 scripts/train_action_sequence.py \
  --data-dir data/photo_action_click_all \
  --output outputs/photo_action_model.pt \
  --epochs 30 \
  --batch-size 128 \
  --model-size base \
  --image-encoder custom \
  --aux-weight 0.7 \
  --device cuda \
  --patience 8
```

ResNet18 视觉编码器训练：
```bash
python3 scripts/train_action_sequence.py \
  --data-dir data/photo_action_click_all \
  --output outputs/photo_action_model_resnet18.pt \
  --epochs 30 \
  --batch-size 64 \
  --model-size base \
  --image-encoder resnet18 \
  --pretrained \
  --aux-weight 0.7 \
  --lr 0.0003 \
  --device cuda \
  --patience 8
```

完整 split 评估：
```bash
python3 scripts/evaluate_action_sequence.py \
  --data-dir data/photo_action_click_all \
  --checkpoint outputs/photo_action_model.pt \
  --split test \
  --threshold auto \
  --output-dir outputs/photo_action_eval \
  --max-failures 24 \
  --device cuda
```

AutoDL 训练步骤见 [docs/action_sequence_autodl.md](docs/action_sequence_autodl.md)。

`clip_vit_b32`、CLIP zero-shot 和 ScreenSpot-Pro adapter 需要完整依赖：`python3 -m pip install -r requirements.txt`。默认 ResNet18/Web 流程只需 `requirements-minimal.txt`。

训练完成后可以生成动作可视化：

```bash
python3 scripts/predict_action_sequence.py \
  --image data/photo_action_click_all/images/val_photo_action_00000.jpg \
  --prompt "请点击所有汽车" \
  --checkpoint outputs/photo_action_model.pt \
  --output outputs/action_prediction.png
```

## 代表性实验结果

以下结果来自同一套 source-disjoint paired test split（2,000 个真实图片九宫格样本），指标定义和失败案例见实验报告：

| 方法 | Cell Exact Match | Cell Precision | Cell Recall | Click Order Accuracy |
|---|---:|---:|---:|---:|
| 专训 ResNet18，clean 数据 | **0.4600** | **0.8677** | **0.8750** | **0.4600** |
| 专训 ResNet18，dirty 数据 | 0.4505 | 0.8766 | 0.8632 | 0.4505 |
| Qwen3-VL-Flash，clean 数据 | 0.1475 | 0.6998 | 0.6081 | 0.1475 |

历史 ResNet18 训练冻结了 backbone 参数，但 BatchNorm running statistics 仍随训练数据更新；当前代码已修复为严格冻结，后续需要重跑才能形成新的 strict-frozen baseline。这些数字说明当前专训模型在固定类别的九宫格 click-all 任务上更有优势，但不代表它具备通用开放世界理解能力。数据清洗带来的提升较小，主要瓶颈仍包括细粒度类别识别、多目标计数和 prompt 分布外泛化。完整分析见 [实验报告 PDF](docs/robotest_experiment_report_updated.pdf)、[LaTeX 源文件](docs/robotest_experiment_report.tex) 与 [机器可读指标](docs/results/action_sequence_benchmark_20260708.json)。

说明：默认生成的九宫格图片不包含文字标签，颜色和类别标注只保存在 `manifest.jsonl` 中。`--debug-labels` 只用于人工检查数据。当前默认实验数据使用 `medium` 难度，颜色会重复，因此单纯颜色基线不再足够；模板图文匹配模型用于展示更完整的“文本条件 + 图像颜色/形状特征 + 目标定位 + 轨迹生成”链路。神经模型是课程扩展基线，适合继续改进为 CLIP、ViT、Cross-Attention 或目标检测式方案。

轨迹生成已使用随机起点和随机格内落点，并带有曲线、轻微抖动和末端停顿，不再固定从左上角移动到格子中心。

课程实验报告见 [docs/robotest_experiment_report_updated.pdf](docs/robotest_experiment_report_updated.pdf)。
