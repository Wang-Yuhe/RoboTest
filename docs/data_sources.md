# 真实照片数据源选择

## 推荐数据集：Open Images

Open Images 是当前最适合本项目的真实照片数据源：

- 规模大，约 900 万张图片；
- 包含图像级标签、目标检测框和分割等标注；
- 目标检测部分覆盖数百个可检测物体类别；
- 图片来自 Flickr，使用 Creative Commons Attribution 授权；
- 可以按类别下载子集，适合只取少量真实物体图片构造 9 宫格。

项目中建议优先使用 Open Images 的 detection 子集，按类别下载图片，并用 bounding box 裁剪目标物体。

## 备选数据集：COCO

COCO 也适合做真实图像实验：

- 包含真实场景照片；
- 有目标检测、分割和 captions；
- 类别体系更小，常用版本有 80 个类别；
- 对自行车、汽车、交通灯、消防栓等常见类别覆盖较好。

COCO 的问题是类别不够细，想要“星星、旗帜、路灯、房子”等类别时不如 Open Images 灵活。

## 当前项目落地策略

本项目新增两层脚本：

1. `scripts/download_openimages_subset.py`
   使用 FiftyOne 下载 Open Images 指定类别的小规模真实照片子集。

2. `scripts/build_photo_grid_dataset.py`
   从本地真实照片目录中抽取 9 类物体图片，裁剪为统一尺寸，拼成 9 宫格，并生成 `manifest.jsonl`。

这样做的好处是：

- 下载和拼图解耦；
- 如果网络或 FiftyOne 不可用，也可以手动放入照片目录后继续构造数据集；
- 后续可以替换成 COCO、自己的照片或其他真实图片来源。

## 如果 Open Images 下载失败

如果看到类似下面的错误：

```text
SSLError: HTTPSConnectionPool(host='storage.googleapis.com', port=443)
```

说明当前环境访问 Google Storage 不稳定。这通常是网络、代理或 SSL 握手问题，不是项目代码问题。

可以先创建手动照片目录模板：

```bash
python3 scripts/download_openimages_subset.py \
  --output-dir data/openimages_raw \
  --manual-template-only
```

然后把真实照片手动放入：

```text
data/openimages_raw/manual_photo_objects/
├── fire_hydrant/
├── bicycle/
├── car/
├── traffic_light/
├── tree/
├── house/
└── flag/
```

如果想构造完整 9 类九宫格，还需要补充：

```text
street_light/
star/
```

之后运行：

```bash
python3 scripts/build_photo_grid_dataset.py \
  --photo-root data/openimages_raw/manual_photo_objects \
  --output-dir data/photo_grid \
  --num-samples 300
```

## 自动下载完整流程

如果当前网络可以访问 Google Storage，可以直接运行：

```bash
bash scripts/run_openimages_auto_pipeline.sh
```

这个脚本会自动执行：

1. 下载 Open Images 子集；
2. 从检测框裁剪真实物体照片；
3. 拼成 9 宫格照片数据集；
4. 训练一个神经模型；
5. 评估并保存预测图。

默认参数等价于：

```bash
bash scripts/run_openimages_auto_pipeline.sh \
  data/openimages_raw \
  data/photo_objects \
  data/photo_grid \
  600 \
  80 \
  300 \
  5
```

参数含义依次是：

- Open Images 运行摘要目录。FiftyOne 实际图片缓存通常位于 `~/fiftyone/open-images-v7/`；
- 裁剪后的物体照片目录；
- 九宫格数据集目录；
- Open Images 最大下载样本数；
- 每类最多裁剪图片数；
- 九宫格样本数；
- 神经模型训练轮数。

如果你使用代理，可以先在终端设置：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
```

端口号需要改成你本机代理软件实际使用的端口。

## 扩展更多类别

为了提升泛化能力，项目提供了扩展类别预设，覆盖车辆、动物、家具、日用品和食物等 40 多个常见物体类别。运行：

```bash
bash scripts/run_expand_photo_classes.sh \
  data/openimages_extended_raw \
  data/photo_objects \
  5000 \
  120
```

含义：

- 从 Open Images 下载最多 5000 张候选图片；
- 每类最多裁剪 120 张目标物体图；
- 输出到 `data/photo_objects`；
- 自动生成类别覆盖报告。

查看当前类别数量：

```bash
python3 scripts/report_photo_objects.py \
  --photo-root data/photo_objects \
  --min-images 20
```

构建更大的训练集：

```bash
python3 scripts/build_photo_grid_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_grid_large \
  --num-samples 3000 \
  --min-images-per-class 20 \
  --hard-augment
```

当可用类别数达到 15-30 个以上时，九宫格组合会更丰富，模型更不容易只记住少量类别和背景模式。
