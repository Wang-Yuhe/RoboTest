# Teammate Setup Guide

这份文档用于把项目推到远程仓库后，让队友 clone 下来能够跑通最小 Demo，并能选择复现实照片训练流程。

## 1. 获取代码

```bash
git clone <REMOTE_URL>
cd RoboTest
```

## 2. 创建环境

推荐使用 conda：

```bash
conda create -n robot-test python=3.12 -y
conda activate robot-test
pip install -r requirements.txt
```

如果 `torch` 安装较慢或失败，可以按 PyTorch 官网给出的本机平台命令单独安装 PyTorch，然后再运行：

```bash
pip install numpy pillow streamlit requests fiftyone
```

## 3. 最小可运行检查

不需要真实照片数据，直接生成合成数据并训练一个小模型：

```bash
python scripts/generate_dataset.py --num-samples 600 --difficulty medium
python scripts/train.py --epochs 2 --batch-size 64 --progress-every 5
python scripts/make_demo_html.py --mode model --checkpoint outputs/model.pt --output outputs/demo_model.html
```

然后打开：

```text
outputs/demo_model.html
```

## 4. 复现实照片 9 宫格数据

真实照片数据、训练生成的九宫格、模型权重和 HTML 输出默认不进 Git，原因是体积较大：

```text
data/
outputs/
*.pt
```

如果队友要完整复现实照片实验，可以重新下载 Open Images 子集：

```bash
bash scripts/run_expand_photo_classes.sh data/openimages_broad_raw data/photo_objects 20000 100 broad 100
bash scripts/run_fill_photo_gaps.sh data/photo_objects data/openimages_gap_fill_raw 100 100 50000 2 train
```

查看类别覆盖：

```bash
python scripts/report_photo_objects.py --photo-root data/photo_objects --min-images 100
```

构建 100 类九宫格数据：

```bash
python scripts/build_photo_grid_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_grid_100cls \
  --num-samples 10000 \
  --min-images-per-class 100 \
  --max-classes 120 \
  --hard-augment \
  --progress-every 100
```

训练当前主模型：

```bash
python scripts/train.py \
  --data-dir data/photo_grid_100cls \
  --output outputs/photo_model_100cls_attn.pt \
  --epochs 40 \
  --batch-size 64 \
  --lr 0.001 \
  --aux-weight 0.7 \
  --patience 10 \
  --model-size attn \
  --progress-every 20
```

评估：

```bash
python scripts/evaluate.py \
  --data-dir data/photo_grid_100cls \
  --mode model \
  --checkpoint outputs/photo_model_100cls_attn.pt \
  --output-dir outputs/eval_100cls_attn \
  --progress-every 100
```

生成 HTML：

```bash
python scripts/make_demo_html.py \
  --data-dir data/photo_grid_100cls \
  --mode model \
  --checkpoint outputs/photo_model_100cls_attn.pt \
  --output outputs/photo_demo_100cls_model.html \
  --num-examples 12
```

## 5. 如果不想让队友重新训练

不要直接把整个 `data/` 和 `outputs/` 提交进 Git。推荐两种方式：

1. 把 `outputs/photo_model_100cls_attn.pt`、`outputs/photo_model_100cls_attn.log.jsonl`、`outputs/eval_100cls_attn/model/metrics.json` 放到 GitHub Release 或网盘。
2. 如果必须用 Git 管理模型权重，使用 Git LFS，只追踪必要的 `.pt` 文件。

可以先用脚本打包当前训练产物：

```bash
bash scripts/export_trained_artifacts.sh
```

默认会生成：

```text
release_artifacts/photo_model_100cls_attn_artifacts.tar.gz
```

把这个压缩包发给队友或上传到 GitHub Release。队友解压后，把 `.pt` checkpoint 复制到项目的 `outputs/` 目录即可直接推理，不需要重新训练。

Git LFS 示例：

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes outputs/photo_model_100cls_attn.pt
git commit -m "Add trained checkpoint with Git LFS"
```

当前数据集 `data/` 约 1GB 以上，不建议直接进入 Git 仓库。

## 6. 当前结果参考

当前 40 轮训练结果见根目录：

```text
brief.md
```

关键结果：

```text
val samples: 1500
top-1 accuracy: 82.4%
top-3 accuracy: 96.87%
checkpoint: outputs/photo_model_100cls_attn.pt
```
