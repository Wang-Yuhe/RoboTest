# Action Sequence Training on AutoDL

This guide trains the click-all action model:

```text
image + Chinese instruction -> selected cells -> move_to_cell / click / done tokens
```

The first supported task is `请点击所有<类别>`, such as `请点击所有汽车`.
The model keeps one feature per grid cell, predicts a 9-cell multi-label target, and uses an auxiliary per-cell object classification loss to make the visual encoder learn object identity.

Use synthetic data only as a sanity check. Reported experiments should use the real-photo pipeline below.

## 1. Prepare Environment

Prefer an AutoDL image that already includes Python, CUDA, and PyTorch. Then run:

```bash
cd /root/autodl-tmp
git clone <your-repo-url> RoboTest
cd RoboTest
python -m pip install --upgrade pip
python -m pip install -r requirements-minimal.txt
```

If the selected AutoDL image does not include PyTorch, install the PyTorch build that matches the CUDA version shown by:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Build Real-Photo Click-All Data

Build source-disjoint train/val/test grids from `data/photo_objects`:

```bash
python scripts/build_photo_action_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_action_click_all_20k \
  --num-train 20000 \
  --num-val 4000 \
  --num-test 4000 \
  --min-images-per-class 80 \
  --max-classes 100 \
  --image-size 192 \
  --min-targets 2 \
  --max-targets 4 \
  --hard-augment \
  --balanced-targets \
  --progress-every 500
```

The builder splits source image files before composing grids, so the same source photo cannot appear across train/val/test.

For comparable experiments, also freeze the source split plan. Otherwise two generated datasets can each be internally source-disjoint while still leaking across experiment versions, for example an old training split can contain photos used by a new test split. A reused split plan fixes the source photos; it does not by itself measure the effect of dropping bad sources, because filtered-out sources are already absent from the plan.

```bash
python scripts/build_photo_action_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_action_click_all_fixed80 \
  --num-train 10000 \
  --num-val 2000 \
  --num-test 2000 \
  --min-images-per-class 40 \
  --class-list outputs/failure_analysis_20260707_1853/baseline_80_classes.txt \
  --image-size 192 \
  --min-targets 2 \
  --max-targets 4 \
  --hard-augment \
  --balanced-targets \
  --min-source-area 5000 \
  --max-source-aspect-ratio 4.0 \
  --blacklist-sources outputs/failure_analysis_20260707_1853/clean80_blacklist_sources.txt \
  --write-source-split-plan outputs/fixed80_source_split_plan.json \
  --progress-every 500
```

Later ablations should pass `--source-split-plan outputs/fixed80_source_split_plan.json` so the source photos behind train/val/test stay fixed while other choices change.

For the cleanest dirty-vs-clean data ablation, freeze the source pool before applying manual bad-source labels. This keeps the experiment from confusing random source selection with data cleaning.

First create the dirty benchmark and export both the source pool and split plan:

```bash
python scripts/build_photo_action_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_action_click_all_dirty80_paired \
  --num-train 10000 \
  --num-val 2000 \
  --num-test 2000 \
  --min-images-per-class 40 \
  --class-list outputs/failure_analysis_20260707_1853/baseline_80_classes.txt \
  --image-size 192 \
  --min-targets 2 \
  --max-targets 4 \
  --hard-augment \
  --balanced-targets \
  --write-source-pool-manifest outputs/paired80_source_pool.json \
  --write-source-split-plan outputs/paired80_source_split_plan.json \
  --progress-every 500
```

Then create the clean paired benchmark from the same source pool and split plan:

```bash
python scripts/build_photo_action_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_action_click_all_clean80_paired \
  --source-pool-manifest outputs/paired80_source_pool.json \
  --source-split-plan outputs/paired80_source_split_plan.json \
  --missing-plan-source-policy skip \
  --bad-source-labels outputs/failure_analysis_20260707_1853/bad_source_labels.json \
  --num-train 10000 \
  --num-val 2000 \
  --num-test 2000 \
  --image-size 192 \
  --min-targets 2 \
  --max-targets 4 \
  --hard-augment \
  --balanced-targets \
  --progress-every 500
```

`--bad-source-labels` expects UTF-8 JSON like:

```json
{
  "sources": [
    {
      "source": "data/photo_objects/dress/example.jpg",
      "class_key": "dress",
      "reason": "ambiguous_crop",
      "exclude": true
    }
  ]
}
```

Use `--missing-plan-source-policy skip` only for this paired cleaning setup. The default `error` mode is safer for ordinary split-plan reuse because it catches accidental path drift.

For a cleaner weak-class pass, add source-quality filters. This is useful for classes where objects are often tiny, extremely wide/tall, or semantically broad:

```bash
python scripts/build_photo_action_dataset.py \
  --photo-root data/photo_objects \
  --output-dir data/photo_action_click_all_clean_20k \
  --num-train 20000 \
  --num-val 4000 \
  --num-test 4000 \
  --min-images-per-class 80 \
  --max-classes 100 \
  --image-size 192 \
  --min-targets 2 \
  --max-targets 4 \
  --hard-augment \
  --balanced-targets \
  --min-source-area 10000 \
  --max-source-aspect-ratio 4.0 \
  --blacklist-sources outputs/failure_analysis_20260707_1853/recommended_blacklist_sources.txt \
  --exclude-classes person,toy \
  --progress-every 500
```

Use `--exclude-classes` only when the benchmark should remove overly broad categories. If you want to keep all categories, omit that flag and rely on source filtering plus the blacklist.

## 3. Quick Smoke Test

Run a tiny synthetic sanity check only if you need to confirm the script starts:

```bash
python scripts/train_action_sequence.py \
  --data-dir data/action_captcha_smoke \
  --output outputs/action_model_smoke.pt \
  --num-samples 64 \
  --epochs 1 \
  --batch-size 16 \
  --aux-weight 0.5 \
  --device auto \
  --progress-every 1
```

Expected result: `outputs/action_model_smoke.pt` and `outputs/action_model_smoke.log.jsonl`.

## 4. Recommended Training Command

For a normal AutoDL run:

```bash
python scripts/train_action_sequence.py \
  --data-dir data/photo_action_click_all_20k \
  --output outputs/photo_action_model_click_all.pt \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.001 \
  --aux-weight 0.7 \
  --model-size base \
  --image-encoder custom \
  --device cuda \
  --patience 8 \
  --progress-every 20
```

Use `data/photo_action_click_all_20k` for the real-photo dataset created above. The script can still auto-generate synthetic data if no manifest exists, but that path is only for sanity checks.

For the stronger visual baseline, use a pretrained ResNet18 cell encoder:

```bash
python scripts/train_action_sequence.py \
  --data-dir data/photo_action_click_all_20k \
  --output outputs/photo_action_model_resnet18.pt \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.0003 \
  --aux-weight 0.7 \
  --model-size base \
  --image-encoder resnet18 \
  --pretrained \
  --encoder-train-mode frozen \
  --device cuda \
  --patience 8 \
  --progress-every 20
```

For small real-photo source pools, prefer `--encoder-train-mode frozen` first. It keeps the ImageNet-pretrained ResNet18 backbone fixed and trains only the projection, text fusion, selector, and auxiliary object head. Use `--encoder-train-mode last_block` only as the next ablation; full fine-tuning can overfit quickly when each class has only dozens of unique source photos.

## 5. Run in Background

```bash
nohup python scripts/train_action_sequence.py \
  --data-dir data/photo_action_click_all_20k \
  --output outputs/photo_action_model_click_all.pt \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.001 \
  --aux-weight 0.7 \
  --model-size base \
  --image-encoder custom \
  --device cuda \
  --patience 8 \
  --progress-every 20 \
  > outputs/action_train_stdout.log 2>&1 &
```

Watch logs:

```bash
tail -f outputs/action_train_stdout.log
tail -f outputs/photo_action_model_click_all.log.jsonl
```

## 6. Outputs

Training writes:

- `outputs/photo_action_model_click_all.pt`: PyTorch checkpoint
- `outputs/photo_action_model_click_all.log.jsonl`: training metrics
- `outputs/photo_action_model_click_all.vocab.json`: Chinese prompt vocabulary
- `outputs/photo_action_model_click_all.action_vocab.json`: action token vocabulary
- `outputs/photo_action_model_click_all.object_vocab.json`: object-class vocabulary

Download at least the checkpoint and JSON logs after training.

## 7. Visualize One Prediction

After training, run:

```bash
python scripts/predict_action_sequence.py \
  --image data/photo_action_click_all_20k/images/val_photo_action_00000.jpg \
  --prompt "请点击所有汽车" \
  --checkpoint outputs/photo_action_model_click_all.pt \
  --output outputs/action_prediction.png \
  --device cuda
```

The command prints predicted cells and `move_to_cell / click / done` actions as JSON, then saves an image with red predicted-cell boxes and a blue mouse trajectory.

On Windows PowerShell, prefer Python for reading `manifest.jsonl` because the default console encoding can corrupt Chinese JSON text:

```bash
python -c "import json, pathlib; p=pathlib.Path('data/photo_action_click_all_20k/manifest.jsonl'); print(json.loads(p.read_text(encoding='utf-8').splitlines()[0])['prompt'])"
```

## 8. Evaluate a Full Split

Run the formal evaluator after training:

```bash
python scripts/evaluate_action_sequence.py \
  --data-dir data/photo_action_click_all_20k \
  --checkpoint outputs/photo_action_model_click_all.pt \
  --split test \
  --threshold auto \
  --output-dir outputs/photo_action_eval \
  --max-failures 24 \
  --device cuda
```

`--threshold auto` chooses the threshold on the validation split, then applies it to the requested split. The default grid includes low recall-friendly values such as `0.03`, `0.05`, `0.07`, and `0.1`. The evaluator writes `metrics.json` with global metrics, per-class metrics, threshold candidates, and up to `--max-failures` visualized error cases.

## 9. Metrics

The training log reports:

- `cell_exact_match`: whether all 9 selected/non-selected cells are correct
- `cell_precision`: precision over selected cells
- `cell_recall`: recall over true target cells
- `exact_match`: full derived action sequence accuracy
- `click_order_accuracy`: whether the derived clicked cell sequence is correct

For this task, `cell_exact_match` and `click_order_accuracy` are the main metrics. They should usually match because actions are derived from the selected cell set in ascending grid order.

## 10. Common Adjustments

If GPU memory is not enough:

```bash
--batch-size 64 --model-size small
```

If training is too slow:

```bash
--num-samples 5000 --epochs 10 --model-size small
```

If validation accuracy is unstable, increase data:

```bash
--num-samples 50000 --epochs 40 --patience 10
```
