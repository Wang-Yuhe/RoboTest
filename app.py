from __future__ import annotations

import random
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

from src.multimodal_captcha.baseline import color_grounding_predict
from src.multimodal_captcha.dataset import CaptchaDataset, build_vocab
from src.multimodal_captcha.generator import draw_prediction_overlay, generate_dataset
from src.multimodal_captcha.model import MultimodalGridLocator, build_model_from_checkpoint, predict_index
from src.multimodal_captcha.streamlit_action_demo import (
    ACTION_PROMPT_KEY,
    DEFAULT_ACTION_CHECKPOINT,
    DEFAULT_ACTION_DATA_DIR,
    build_turnstile_widget_html,
    draw_action_demo_overlay,
    explicit_cached_request,
    load_action_demo_model,
    load_action_demo_records,
    predict_action_demo,
    resolve_first_existing_path,
    sync_action_prompt_state,
)
from src.multimodal_captcha.prompt_rewriter import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    load_supported_objects,
    merge_rewrite_results,
    rewrite_prompt,
)
from src.multimodal_captcha.query_planner import execute_query_plan, plan_prompt
from src.multimodal_captcha.template_matcher import template_grounding_predict
from src.multimodal_captcha.trajectory import generate_mouse_trajectory, random_point_in_cell
from src.multimodal_captcha.vlm_baseline import QWEN_BASE_URL, QWEN_MODEL, QwenVlmBaseline
from src.multimodal_captcha.visualize import draw_trajectory


SYNTHETIC_DATA_DIR = Path("data/synthetic_captcha")
SINGLE_TARGET_CHECKPOINT = Path("outputs/model.pt")


@st.cache_resource
def load_single_target_model_and_data() -> tuple[MultimodalGridLocator, CaptchaDataset, bool]:
    if not (SYNTHETIC_DATA_DIR / "manifest.jsonl").exists():
        generate_dataset(SYNTHETIC_DATA_DIR, 300, seed=21, image_size=192)

    trained = SINGLE_TARGET_CHECKPOINT.exists()
    if trained:
        checkpoint = torch.load(SINGLE_TARGET_CHECKPOINT, map_location="cpu", weights_only=True)
        vocab = checkpoint["vocab"]
        object_vocab = checkpoint.get("object_vocab", {})
        model = build_model_from_checkpoint(checkpoint, len(vocab), len(object_vocab))
        model.load_state_dict(checkpoint["model"])
    else:
        vocab = build_vocab(SYNTHETIC_DATA_DIR / "manifest.jsonl")
        model = MultimodalGridLocator(vocab_size=len(vocab))

    dataset = CaptchaDataset(SYNTHETIC_DATA_DIR, split="val", vocab=vocab)
    model.eval()
    return model, dataset, trained


@st.cache_data(show_spinner=False)
def load_action_records_cached(data_dir: str, split: str, limit: int) -> list[dict]:
    return load_action_demo_records(data_dir, split=split, limit=limit)


@st.cache_resource
def load_action_model_cached(checkpoint_path: str, device: str):
    return load_action_demo_model(checkpoint_path, device=device)


def choose_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        st.warning("当前环境没有可用 CUDA，已回退到 CPU。")
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def render_single_target_tab() -> None:
    model, dataset, trained = load_single_target_model_and_data()
    if not trained:
        st.warning("没有发现 outputs/model.pt，当前神经网络模式会使用未训练权重。")

    if "single_sample_idx" not in st.session_state:
        st.session_state.single_sample_idx = random.randrange(len(dataset))

    controls, image_col, result_col = st.columns([0.85, 1.15, 1])
    with controls:
        mode = st.radio(
            "定位模式",
            ["模板图文模型", "规则颜色基线", "神经网络模型"],
            horizontal=False,
        )
        if st.button("换一个单目标样本", use_container_width=True):
            st.session_state.single_sample_idx = random.randrange(len(dataset))

    sample = dataset[st.session_state.single_sample_idx]
    record = dataset.records[st.session_state.single_sample_idx]
    image = Image.open(SYNTHETIC_DATA_DIR / record["image"]).convert("RGB")
    if mode == "神经网络模型":
        pred, probs = predict_index(model, sample["image"], sample["text"])
        model_label = "神经网络模型"
    elif mode == "模板图文模型":
        pred, probs = template_grounding_predict(image, record["prompt"])
        model_label = "模板图文模型"
    else:
        pred, probs = color_grounding_predict(image, record["prompt"])
        model_label = "规则颜色基线"

    overlay = draw_prediction_overlay(image, record["target_index"], pred)
    points = generate_mouse_trajectory(
        random_point_in_cell(pred, image.size[0], random.Random(st.session_state.single_sample_idx)),
        seed=st.session_state.single_sample_idx,
        image_size=image.size[0],
    )
    vis = draw_trajectory(overlay, points)

    with image_col:
        st.image(vis, caption="绿色是真实目标，红色是预测目标，蓝线是生成轨迹。", width=420)

    with result_col:
        st.subheader("任务")
        st.write(record["prompt"])
        st.metric("真实格子", int(record["target_index"]))
        st.metric(f"{model_label}预测", int(pred))
        st.metric("是否正确", "是" if pred == record["target_index"] else "否")
        st.caption("九宫格概率")
        for index, prob in enumerate(probs):
            st.progress(float(prob), text=f"格子 {index}: {float(prob):.3f}")


def render_action_sequence_tab() -> None:
    st.caption("输入九宫格真实图片和文本指令，模型输出需要点击的所有格子，再渲染离散点击动作和鼠标轨迹。")

    default_data_dir = resolve_first_existing_path([DEFAULT_ACTION_DATA_DIR])
    default_checkpoint = resolve_first_existing_path([DEFAULT_ACTION_CHECKPOINT])

    with st.sidebar:
        st.subheader("Action sequence demo")
        data_dir_text = st.text_input(
            "数据集目录",
            value=str(default_data_dir or DEFAULT_ACTION_DATA_DIR),
        )
        checkpoint_text = st.text_input(
            "模型 checkpoint",
            value=str(default_checkpoint or DEFAULT_ACTION_CHECKPOINT),
        )
        split = st.selectbox("数据 split", ["test", "val", "train"], index=0)
        threshold = st.slider("点击阈值", min_value=0.03, max_value=0.90, value=0.50, step=0.01)
        decode_policy = st.selectbox("解码策略", ["threshold", "topk_count"], index=0)
        requested_device = st.selectbox("推理设备", ["auto", "cpu", "cuda"], index=0)
        enable_planner = st.checkbox("启用结构化 planner / 属性执行器", value=False)
        enable_rewrite = st.checkbox("启用 DeepSeek prompt 改写", value=False)
        deepseek_model = st.text_input("DeepSeek 模型", value=DEEPSEEK_MODEL)
        deepseek_base_url = st.text_input("DeepSeek base URL", value=DEEPSEEK_BASE_URL)
        deepseek_api_key = st.text_input("DeepSeek API key（可留空使用环境变量）", value="", type="password")
        enable_vlm_compare = st.checkbox("启用多模态大模型对比", value=False)
        vlm_provider = st.selectbox("多模态大模型 provider", ["qwen"], index=0)
        vlm_model = st.text_input("Qwen-VL 模型", value=QWEN_MODEL)
        vlm_base_url = st.text_input("Qwen-VL base URL", value=QWEN_BASE_URL)
        vlm_api_key = st.text_input("Qwen/DashScope API key（可留空使用环境变量）", value="", type="password")

    data_dir = Path(data_dir_text)
    checkpoint_path = Path(checkpoint_text)
    if not data_dir.exists():
        st.error(f"数据集目录不存在：{data_dir}")
        return
    if not checkpoint_path.exists():
        st.error(f"checkpoint 不存在：{checkpoint_path}")
        return

    try:
        records = load_action_records_cached(str(data_dir), split=split, limit=500)
    except Exception as exc:
        st.error(f"加载数据集失败：{exc}")
        return

    if "action_sample_idx" not in st.session_state:
        st.session_state.action_sample_idx = 0

    max_idx = max(len(records) - 1, 0)
    top_controls = st.columns([1, 1, 1.3])
    with top_controls[0]:
        if st.button("随机样本", use_container_width=True):
            st.session_state.action_sample_idx = random.randrange(len(records))
    with top_controls[1]:
        if st.button("下一个样本", use_container_width=True):
            st.session_state.action_sample_idx = (st.session_state.action_sample_idx + 1) % len(records)
    with top_controls[2]:
        st.session_state.action_sample_idx = st.number_input(
            "样本序号",
            min_value=0,
            max_value=max_idx,
            value=min(st.session_state.action_sample_idx, max_idx),
            step=1,
        )

    record = records[int(st.session_state.action_sample_idx)]
    image = Image.open(data_dir / record["image"]).convert("RGB")
    record_key = f"{split}:{record.get('image', st.session_state.action_sample_idx)}"
    sync_action_prompt_state(st.session_state, record_key=record_key, default_prompt=record["prompt"])
    prompt_col, reset_col = st.columns([4, 1])
    with prompt_col:
        prompt = st.text_input("指令 prompt（可手动修改，修改后会立即重新推理）", key=ACTION_PROMPT_KEY)
    with reset_col:
        st.write("")
        st.write("")
        if st.button("恢复样本 prompt", use_container_width=True):
            st.session_state[ACTION_PROMPT_KEY] = record["prompt"]
            st.rerun()
    paid_action_cols = st.columns(2)
    run_deepseek_request = False
    run_vlm_request = False
    if enable_rewrite:
        with paid_action_cols[0]:
            run_deepseek_request = st.button("运行 DeepSeek 改写", use_container_width=True)
    if enable_vlm_compare:
        with paid_action_cols[1]:
            run_vlm_request = st.button("运行 Qwen-VL 对比", use_container_width=True)
    target_indices = [int(index) for index in record.get("target_indices", [])]

    device = choose_device(requested_device)
    try:
        model, vocab, config = load_action_model_cached(str(checkpoint_path), device)
        planner_result = None
        planner_object_predictions = []
        rewrite = None
        effective_prompts = [prompt]
        if enable_planner:
            supported_objects = load_supported_objects(data_dir)
            plan = plan_prompt(prompt, supported_objects)

            def object_predictor(current_prompt: str) -> list[int]:
                prediction = predict_action_demo(
                    model=model,
                    vocab=vocab,
                    config=config,
                    image=image,
                    prompt=current_prompt,
                    target_indices=target_indices,
                    threshold=threshold,
                    decode_policy=decode_policy,
                    device=device,
                    seed=int(st.session_state.action_sample_idx) + len(planner_object_predictions),
                )
                planner_object_predictions.append(
                    {
                        "prompt": current_prompt,
                        "predicted_indices": prediction.predicted_indices,
                        "cell_probabilities": prediction.cell_probabilities,
                    }
                )
                return prediction.predicted_indices

            planner_result = execute_query_plan(image, plan, object_predictor=object_predictor)
            effective_prompts = planner_result.get("rewritten_prompts") or [prompt]
            predictions = []
        elif enable_rewrite:
            supported_objects = load_supported_objects(data_dir)
            try:
                rewrite = explicit_cached_request(
                    st.session_state,
                    "deepseek_rewrite_cache",
                    (record_key, prompt, deepseek_base_url, deepseek_model),
                    run_deepseek_request,
                    lambda: rewrite_prompt(
                        prompt,
                        supported_objects=supported_objects,
                        provider="deepseek",
                        api_key=deepseek_api_key.strip() or None,
                        base_url=deepseek_base_url,
                        model=deepseek_model,
                    ),
                )
            except Exception as exc:
                st.warning(f"DeepSeek prompt 改写失败，已退回原始 prompt：{exc}")
                rewrite = None
            if rewrite is not None:
                if rewrite.rewritten_prompts:
                    effective_prompts = rewrite.rewritten_prompts
                else:
                    st.warning("prompt 改写没有命中当前数据集支持的类别，继续使用原始 prompt。")
        predictions = [
            predict_action_demo(
                model=model,
                vocab=vocab,
                config=config,
                image=image,
                prompt=current_prompt,
                target_indices=target_indices,
                threshold=threshold,
                decode_policy=decode_policy,
                device=device,
                seed=int(st.session_state.action_sample_idx) + idx,
            )
            for idx, current_prompt in enumerate(effective_prompts)
        ]
    except Exception as exc:
        st.error(f"模型推理失败：{exc}")
        return

    if planner_result is not None:
        merged = None
        predicted_indices = planner_result["selected_cells"]
        merged_probs = (
            planner_result.get("color_scores")
            or planner_result.get("position_scores")
            or planner_result.get("size_scores")
            or [0.0] * 9
        )
        if planner_object_predictions:
            for object_prediction in planner_object_predictions:
                for index, score in enumerate(object_prediction["cell_probabilities"]):
                    merged_probs[index] = max(float(merged_probs[index]), float(score))
        visualization = draw_action_demo_overlay(
            image,
            predicted_indices,
            target_indices=target_indices,
            seed=int(st.session_state.action_sample_idx),
        )
        correct = sorted(predicted_indices) == sorted(target_indices)
    elif enable_rewrite and rewrite is not None and predictions:
        per_prompt_payloads = [
            {
                "prompt": prediction.prompt,
                "predicted_indices": prediction.predicted_indices,
                "cell_probabilities": prediction.cell_probabilities,
            }
            for prediction in predictions
        ]
        merged = merge_rewrite_results(prompt, rewrite, per_prompt_payloads)
        predicted_indices = merged["predicted_indices"]
        merged_probs = [
            max(prediction.cell_probabilities[index] for prediction in predictions)
            for index in range(9)
        ]
        visualization = draw_action_demo_overlay(
            image,
            predicted_indices,
            target_indices=target_indices,
            seed=int(st.session_state.action_sample_idx),
        )
        correct = sorted(predicted_indices) == sorted(target_indices)
    else:
        merged = None
        prediction = predictions[0]
        predicted_indices = prediction.predicted_indices
        merged_probs = prediction.cell_probabilities
        visualization = prediction.visualization
        correct = prediction.correct

    vlm_prediction = None
    vlm_error = None
    if enable_vlm_compare:
        try:
            if vlm_provider == "qwen":
                vlm_prediction = explicit_cached_request(
                    st.session_state,
                    "qwen_vlm_cache",
                    (record_key, prompt, vlm_base_url, vlm_model),
                    run_vlm_request,
                    lambda: QwenVlmBaseline(
                        api_key=vlm_api_key.strip() or None,
                        base_url=vlm_base_url,
                        model=vlm_model,
                    ).predict(data_dir / record["image"], prompt),
                )
            else:
                vlm_error = f"Unsupported VLM provider: {vlm_provider}"
        except Exception as exc:
            vlm_error = str(exc)

    left, right = st.columns([1.15, 1])
    with left:
        st.image(visualization, caption="绿色是真实目标，红色是模型点击目标，蓝线是动作轨迹。", width=500)

    with right:
        st.subheader("预测结果")
        st.metric("真实目标格", ", ".join(map(str, target_indices)) or "-")
        st.metric("预测点击格", ", ".join(map(str, predicted_indices)) or "-")
        if correct is not None:
            st.metric("集合是否完全匹配", "是" if correct else "否")
        if planner_result is not None:
            st.caption("结构化 planner / 属性执行器")
            st.json(
                {
                    "mode": planner_result["mode"],
                    "objects": planner_result["objects"],
                    "color": planner_result["color"],
                    "position": planner_result["position"],
                    "size": planner_result["size"],
                    "rewritten_prompts": planner_result["rewritten_prompts"],
                    "object_candidate_cells": planner_result["object_candidate_cells"],
                    "selected_cells": planner_result["selected_cells"],
                    "color_scores": planner_result.get("color_scores"),
                    "position_scores": planner_result.get("position_scores"),
                    "size_scores": planner_result.get("size_scores"),
                    "object_predictions": planner_object_predictions,
                }
            )
        if merged is not None:
            st.caption("DeepSeek 改写结果")
            st.json(
                {
                    "provider": merged["provider"],
                    "target_objects": merged["target_objects"],
                    "rewritten_prompts": merged["rewritten_prompts"],
                    "reason": merged["reason"],
                    "per_prompt_predictions": merged["per_prompt_predictions"],
                }
            )
        if enable_vlm_compare:
            st.caption("多模态大模型对比")
            if vlm_error:
                st.warning(vlm_error)
            elif vlm_prediction is not None:
                vlm_correct = sorted(vlm_prediction.predicted_indices) == sorted(target_indices)
                st.json(
                    {
                        "provider": vlm_prediction.provider,
                        "model": vlm_prediction.model,
                        "predicted_indices": vlm_prediction.predicted_indices,
                        "target_indices": target_indices,
                        "cell_exact_match": vlm_correct,
                        "raw_response": vlm_prediction.raw_response,
                    }
                )
            else:
                st.info("点击“运行 Qwen-VL 对比”后才会调用付费 API。")
        st.json(
            {
                "prompt": prompt,
                "effective_prompts": effective_prompts,
                "predicted_indices": predicted_indices,
                "target_indices": target_indices,
                "cell_probabilities": merged_probs,
                "threshold": threshold,
                "decode_policy": decode_policy,
                "device": device,
            }
        )

    st.caption("格子置信度")
    cols = st.columns(9)
    for index, prob in enumerate(merged_probs):
        cols[index].metric(str(index), f"{prob:.3f}")


def render_official_service_tab() -> None:
    st.subheader("Cloudflare Turnstile 测试接入")
    st.write(
        "这里展示的是官方 CAPTCHA 服务的合规接入方式，用于真实网站的人机验证流程。"
        "RoboTest 模型只用于我们自有九宫格 CAPTCHA 的研究和演示，不用于绕过第三方服务。"
    )
    st.info(
        "默认使用 Cloudflare 官方测试 sitekey，可在 localhost 上展示 widget。"
        "生产环境需要在 Cloudflare 控制台创建自己的 sitekey，并在服务端校验 token。"
    )
    site_key = st.text_input("Turnstile sitekey", value="1x00000000000000000000AA")
    components.html(build_turnstile_widget_html(site_key), height=110)


def main() -> None:
    st.set_page_config(page_title="RoboTest CAPTCHA Demo", layout="wide")
    st.title("RoboTest 多模态 CAPTCHA 演示")
    st.caption("把训练好的定位模型接到可交互页面，用真实九宫格样本展示点击所有同类目标的推理效果。")

    tab_single, tab_action, tab_service = st.tabs(["单目标定位", "点击所有同类目标", "官方服务接入"])
    with tab_single:
        render_single_target_tab()
    with tab_action:
        render_action_sequence_tab()
    with tab_service:
        render_official_service_tab()


if __name__ == "__main__":
    main()
