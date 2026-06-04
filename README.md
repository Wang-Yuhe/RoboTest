# Multimodal CAPTCHA Grounding

一个可运行的多模态数据处理课程项目：自动生成 9 宫格图文验证任务，用轻量多模态模型根据中文指令定位目标格子，并生成鼠标点击轨迹。

## 项目目标

输入：

- 一张合成 9 宫格验证图片
- 一条中文任务描述，例如 `请点击红色消防栓`

输出：

- 目标格子编号
- 目标点击坐标
- 模拟鼠标运动轨迹

项目只使用合成数据，面向多模态理解、视觉定位和交互轨迹建模教学，不针对真实网站验证码或安全系统。

## 快速开始

```bash
python3 scripts/generate_dataset.py --num-samples 600
python3 scripts/train.py --epochs 5 --batch-size 64
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

## 目录结构

```text
.
├── app.py                         # Streamlit 可视化演示
├── data/                          # 生成的数据集
├── docs/
│   └── report.md                  # 课程报告骨架
├── outputs/                       # 模型权重和预测图
├── scripts/
│   ├── evaluate.py                # 批量评估并保存失败案例
│   ├── build_photo_grid_dataset.py # 从真实照片目录拼 9 宫格
│   ├── download_openimages_subset.py # 下载 Open Images 子集
│   ├── generate_dataset.py        # 生成合成图文对
│   ├── make_demo_html.py          # 无需 Streamlit 的静态网页 Demo
│   ├── predict_image.py           # 对自定义图片和文本指令做预测
│   ├── train.py                   # 训练多模态定位模型
│   └── predict.py                 # 命令行预测示例
└── src/multimodal_captcha/
    ├── baseline.py                # 规则颜色基线
    ├── template_matcher.py        # 模板图文匹配模型
    ├── generator.py               # 图像和标注生成
    ├── dataset.py                 # PyTorch Dataset
    ├── model.py                   # 图文融合模型
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

# 一键运行 hard 测试
bash scripts/run_hard_test.sh

# 从真实照片目录构造九宫格数据集
python3 scripts/build_photo_grid_dataset.py --photo-root data/photo_objects --output-dir data/photo_grid

# 运行真实照片九宫格实验
bash scripts/run_photo_grid_pipeline.sh data/photo_objects data/photo_grid 300

# 自动下载 Open Images、裁剪物体、拼九宫格、训练并评估
bash scripts/run_openimages_auto_pipeline.sh

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

说明：默认生成的九宫格图片不包含文字标签，颜色和类别标注只保存在 `manifest.jsonl` 中。`--debug-labels` 只用于人工检查数据。当前默认实验数据使用 `medium` 难度，颜色会重复，因此单纯颜色基线不再足够；模板图文匹配模型用于展示更完整的“文本条件 + 图像颜色/形状特征 + 目标定位 + 轨迹生成”链路。神经模型是课程扩展基线，适合继续改进为 CLIP、ViT、Cross-Attention 或目标检测式方案。

课程报告骨架见 [docs/report.md](docs/report.md)。
