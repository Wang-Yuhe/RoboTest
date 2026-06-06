from __future__ import annotations

import random
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.multimodal_captcha.baseline import color_grounding_predict
from src.multimodal_captcha.dataset import CaptchaDataset, build_vocab
from src.multimodal_captcha.generator import draw_prediction_overlay, generate_dataset
from src.multimodal_captcha.model import MultimodalGridLocator, build_model_from_checkpoint, predict_index
from src.multimodal_captcha.template_matcher import template_grounding_predict
from src.multimodal_captcha.trajectory import generate_mouse_trajectory, random_point_in_cell
from src.multimodal_captcha.visualize import draw_trajectory


DATA_DIR = Path("data/synthetic_captcha")
CHECKPOINT = Path("outputs/model.pt")


@st.cache_resource
def load_model_and_data() -> tuple[MultimodalGridLocator, CaptchaDataset, bool]:
    if not (DATA_DIR / "manifest.jsonl").exists():
        generate_dataset(DATA_DIR, 300, seed=21, image_size=192)

    trained = CHECKPOINT.exists()
    if trained:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        vocab = checkpoint["vocab"]
        object_vocab = checkpoint.get("object_vocab", {})
        model = build_model_from_checkpoint(checkpoint, len(vocab), len(object_vocab))
        model.load_state_dict(checkpoint["model"])
    else:
        vocab = build_vocab(DATA_DIR / "manifest.jsonl")
        model = MultimodalGridLocator(vocab_size=len(vocab))

    dataset = CaptchaDataset(DATA_DIR, split="val", vocab=vocab)
    return model, dataset, trained


st.set_page_config(page_title="多模态验证码定位", layout="wide")
st.title("多模态验证码定位与鼠标轨迹生成")

model, dataset, trained = load_model_and_data()
if not trained:
    st.warning("尚未发现 outputs/model.pt。当前使用未训练模型；先运行 `python3 scripts/train.py --epochs 5` 可获得更好效果。")

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = random.randrange(len(dataset))

left, right = st.columns([1.15, 1])
with left:
    mode = st.radio("定位模式", ["模板图文模型", "规则颜色基线", "神经模型"], horizontal=True)
    if st.button("换一个样本"):
        st.session_state.sample_idx = random.randrange(len(dataset))

    sample = dataset[st.session_state.sample_idx]
    record = dataset.records[st.session_state.sample_idx]
    image = Image.open(DATA_DIR / record["image"]).convert("RGB")
    if mode == "神经模型":
        pred, probs = predict_index(model, sample["image"], sample["text"])
    elif mode == "模板图文模型":
        pred, probs = template_grounding_predict(image, record["prompt"])
    else:
        pred, probs = color_grounding_predict(image, record["prompt"])
    overlay = draw_prediction_overlay(image, record["target_index"], pred)
    points = generate_mouse_trajectory(
        random_point_in_cell(pred, image.size[0], random.Random(st.session_state.sample_idx)),
        seed=st.session_state.sample_idx,
        image_size=image.size[0],
    )
    vis = draw_trajectory(overlay, points)
    st.image(vis, caption="绿色框为真实目标，红色框为模型预测，蓝线为生成轨迹。", width=520)

with right:
    st.subheader("任务")
    st.write(record["prompt"])
    st.metric("真实格子", int(record["target_index"]))
    st.metric("预测格子", int(pred))
    st.metric("是否正确", "是" if pred == record["target_index"] else "否")
    st.subheader("九宫格概率")
    for i, p in enumerate(probs):
        st.progress(float(p), text=f"格子 {i}: {float(p):.3f}")
