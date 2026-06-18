"""Streamlit web server for visualizing EPRBench retrieval results.

Run with:
    streamlit run src/visualize_server.py --server.port 8501 --server.address 0.0.0.0

Then access the UI via port forwarding on the host machine.
"""

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import streamlit as st
import torch
from omegaconf import OmegaConf
from PIL import Image

# Resolve project root: the script may be run from repo root or src/
try:
    script_dir = Path(__file__).resolve().parent
except NameError:  # e.g. when imported in some environments
    script_dir = Path.cwd()
project_root = script_dir.parent if script_dir.name == "src" else script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import rootutils

rootutils.setup_root(str(project_root), indicator=".project-root", pythonpath=True)

import hydra

from src.data.components.eventvpr_dataset import EventVPRDataset
from src.data.eprbench_datamodule import EPRBenchDataModule
from src.models.eprbench_module import EPRBenchLitModule


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_FOLDER = Path("data/EventVPR")
CKPT_PATH = Path("model/SGVPR.pth")
FOUNDATION_PATH = Path("model/dinov2_vitb14_pretrain.pth")
USE_TEXT = True
TEXT_FOLDER = None
FEATURES_DIM = 10752
RECALL_VALUES: List[int] = [1, 5, 10, 100]
INFER_BATCH_SIZE = 18
NUM_WORKERS = 4
TOP_K = 10


def build_cfg() -> OmegaConf:
    """Build a Hydra-like config object for inference."""
    return OmegaConf.create(
        {
            "data": {
                "_target_": "src.data.eprbench_datamodule.EPRBenchDataModule",
                "dataset_folder": str(DATASET_FOLDER),
                "train_batch_size": 18,
                "infer_batch_size": INFER_BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "pin_memory": True,
                "use_text": USE_TEXT,
                "text_folder": TEXT_FOLDER,
                "img_per_place": 5,
                "min_img_per_place": 5,
                "resize": [224, 224],
            },
            "model": {
                "_target_": "src.models.eprbench_module.EPRBenchLitModule",
                "lambda_contrast": 0.15,
                "temperature": 0.07,
                "use_text": USE_TEXT,
                "features_dim": FEATURES_DIM,
                "test_method": "hard_resize",
                "recall_values": RECALL_VALUES,
                "infer_batch_size": INFER_BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "pca_dim": None,
                "efficient_ram_testing": False,
                "majority_weight": 0.01,
                "compile": False,
                "use_freq_contrastive_loss": False,
                "freq_loss_weight": 0.0,
                "freq_num_negatives": 2,
                "use_temporal_consistency": False,
                "temporal_consistency_weight": 0.05,
                "text_dropout_prob": 0.0,
                "word_dropout_prob": 0.0,
                "sacf_residual_loss_weight": 0.0,
                "sacf_noise_loss_weight": 0.0,
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "_partial_": True,
                    "lr": 0.0001,
                    "weight_decay": 0.001,
                },
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "_partial_": True,
                    "gamma": 0.5,
                },
                "net": {
                    "_target_": "src.models.components.network.SGVPRNet",
                    "pretrained_foundation": True,
                    "foundation_model_path": str(FOUNDATION_PATH),
                    "use_frequency_enhance": False,
                    "use_freq_topk": False,
                    "freq_score_weight": 0.10,
                    "freq_low_cut": 0.25,
                    "freq_mid_cut": 0.60,
                    "freq_highpass_cut": 0.25,
                    "freq_max_gain": 0.50,
                    "freq_gamma_init": 0.0,
                    "mods": {
                        "cross_spectral": {"enabled": False, "layer_idx": 6},
                        "sacf": {"enabled": False, "variant": "original", "layer_idx": 6},
                        "scfm": {"enabled": False},
                        "igr": {"enabled": False},
                        "lgfp": {"enabled": False},
                        "fbgw": {"enabled": False},
                        "bim": {"enabled": False},
                        "spm": {"enabled": False},
                        "event_cues": {"enabled": False},
                        "event_aware_token_selector": {"enabled": False},
                        "adaptive_text_topk": {"enabled": False},
                        "temporal_fusion": {"enabled": False},
                    },
                },
            },
        }
    )


@st.cache_resource(show_spinner=True)
def load_model_and_features() -> Dict[str, Any]:
    """Load model, dataset, extract features and build FAISS index.

    Cached as a single resource to avoid reloading on every interaction.
    """
    cfg = build_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule: EPRBenchDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("test")
    test_dataset: EventVPRDataset = datamodule.data_test

    model: EPRBenchLitModule = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    model.eval()

    # Load SGVPR checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    model.net.load_state_dict(model_state, strict=False)
    if "text_encoder_state_dict" in ckpt:
        model.text_encoder.load_state_dict(ckpt["text_encoder_state_dict"], strict=False)

    # Extract features
    def extract_features(dataloader):
        all_features = []
        all_indices = []
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    imgs, indices, text_desc = batch
                    text_vec = model.text_encoder(text_desc)
                    text_tokens, text_mask = model.text_encoder.encode_batch_tokens(text_desc)
                    text_tokens = text_tokens.to(device)
                    text_mask = text_mask.to(device)
                    feats = model.net(
                        imgs.to(device),
                        text=text_vec,
                        text_tokens=text_tokens,
                        text_mask=text_mask,
                    )
                else:
                    imgs, indices = batch
                    feats = model.net(imgs.to(device))
                all_features.append(feats.cpu())
                all_indices.append(indices.cpu())
        features = torch.cat(all_features, dim=0).numpy()
        indices = torch.cat(all_indices, dim=0).numpy()
        return features, indices

    test_loader = datamodule.test_dataloader()
    all_features, all_indices = extract_features(test_loader)

    sorted_features = np.empty((len(test_dataset), FEATURES_DIM), dtype=np.float32)
    sorted_features[all_indices] = all_features

    database_features = sorted_features[: test_dataset.database_num]
    queries_features = sorted_features[test_dataset.database_num :]

    faiss_index = faiss.IndexFlatL2(FEATURES_DIM)
    faiss_index.add(database_features)

    top_k_max = max(max(RECALL_VALUES), TOP_K)
    _, predictions = faiss_index.search(queries_features, top_k_max)

    positives_per_query = test_dataset.get_positives()
    recalls_per_query = np.zeros((len(positives_per_query), len(RECALL_VALUES)), dtype=bool)
    for q_idx, (pred, pos) in enumerate(zip(predictions, positives_per_query)):
        for r_i, n in enumerate(RECALL_VALUES):
            recalls_per_query[q_idx, r_i] = np.any(np.in1d(pred[:n], pos))
    overall_recalls = recalls_per_query.mean(axis=0) * 100

    return {
        "device": device,
        "model": model,
        "test_dataset": test_dataset,
        "predictions": predictions,
        "positives_per_query": positives_per_query,
        "recalls_per_query": recalls_per_query,
        "overall_recalls": overall_recalls,
    }


def load_image(path: str, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Load and resize an image for display."""
    return Image.open(path).convert("RGB").resize(size[::-1])


def render_query_results(query_idx: int, ctx: Dict[str, Any], top_k: int = TOP_K):
    """Render a query and its top-k retrieved database images."""
    test_dataset: EventVPRDataset = ctx["test_dataset"]
    predictions: np.ndarray = ctx["predictions"]
    positives_per_query = ctx["positives_per_query"]

    query_global_idx = test_dataset.database_num + query_idx
    query_path = test_dataset.all_images_paths[query_global_idx]
    query_place_id = test_dataset._all_place_ids[query_global_idx]
    positives = positives_per_query[query_idx]
    pred = predictions[query_idx][:top_k]

    text_desc = test_dataset.text_descriptions.get(query_path, "")

    hits = {n: bool(np.any(np.in1d(pred[:n], positives))) for n in RECALL_VALUES}

    st.subheader(f"Query #{query_idx} — Place {query_place_id}")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(load_image(query_path), caption="Query", use_container_width=True)
    with col2:
        st.markdown("**Scene description**")
        st.write(text_desc if text_desc else "No description available.")
        hit_str = ", ".join([f"R@{n}: {'✓' if hits[n] else '✗'}" for n in RECALL_VALUES])
        st.markdown(f"**Recall**: {hit_str}")

    st.markdown("---")
    st.markdown("**Retrieved images**")

    cols = st.columns(top_k)
    for rank, db_idx in enumerate(pred, start=1):
        db_path = test_dataset.all_images_paths[db_idx]
        db_place_id = test_dataset._all_place_ids[db_idx]
        is_positive = db_idx in positives
        label = "GT ✓" if is_positive else "GT ✗"
        caption = f"Rank {rank}\nPlace {db_place_id}\n{label}"
        with cols[rank - 1]:
            st.image(
                load_image(db_path),
                caption=caption,
                use_container_width=True,
            )


def main():
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="EPRBench Visual Retrieval", layout="wide")
    st.title("EPRBench 可视化推理服务")
    st.markdown(
        "加载 SGVPR 模型，对 EventVPR 测试集中的 query 进行地点识别检索，"
        "并直观展示 Top-K 结果与 GT 正样本。"
    )

    with st.spinner("正在加载模型并提取特征，请稍候..."):
        ctx = load_model_and_features()

    test_dataset: EventVPRDataset = ctx["test_dataset"]
    queries_num = test_dataset.queries_num
    overall = ctx["overall_recalls"]

    st.sidebar.header("设置")
    st.sidebar.markdown(
        "**整体测试集 Recall**  \n"
        + "  \n".join([f"R@{n}: {rec:.2f}%" for n, rec in zip(RECALL_VALUES, overall)])
    )
    st.sidebar.markdown(f"**Query 总数**: {queries_num}")

    view_mode = st.sidebar.radio("选择模式", ["单张 query", "R@1 成功样例", "R@1 失败样例"])

    top_k = st.sidebar.slider("Top-K 可视化数量", min_value=1, max_value=20, value=TOP_K)

    if view_mode == "单张 query":
        query_idx = st.sidebar.number_input(
            "Query 索引",
            min_value=0,
            max_value=queries_num - 1,
            value=random.randint(0, queries_num - 1),
            step=1,
        )
        if st.sidebar.button("随机选择"):
            query_idx = random.randint(0, queries_num - 1)
            st.session_state["query_idx"] = query_idx
        render_query_results(query_idx, ctx, top_k=top_k)

    else:
        recalls_per_query = ctx["recalls_per_query"]
        pool = (
            np.where(recalls_per_query[:, 0])[0]
            if view_mode == "R@1 成功样例"
            else np.where(~recalls_per_query[:, 0])[0]
        )
        if len(pool) == 0:
            st.warning("没有符合条件的样例。")
            return

        sample_size = min(6, len(pool))
        if st.sidebar.button("重新随机抽取"):
            st.session_state["sample_seed"] = random.randint(0, 1 << 30)
        seed = st.session_state.get("sample_seed", 42)
        rng = random.Random(seed)
        samples = rng.sample(list(pool), sample_size)

        st.subheader(f"{view_mode} ({sample_size} 张)")
        for q in samples:
            render_query_results(int(q), ctx, top_k=top_k)
            st.markdown("---")


if __name__ == "__main__":
    main()
