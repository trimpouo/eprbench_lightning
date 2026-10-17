"""Dataset for NYC-Event-VPR_VPR-Bench evaluation format.

This loads query.zip / ref.zip (already extracted) and ground_truth_new.npy,
and exposes the interface expected by EPRBenchLitModule:
    - database_num, queries_num
    - get_positives()
    - __getitem__ returns (image, global_index, text_desc)
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class NYCVPRBenchDataset(Dataset):
    def __init__(
        self,
        dataset_folder: str,
        modality: str = "event",  # or "rgb"
        split: str = "test",
        transform=None,
        use_text: bool = False,
        ground_truth_name: str = "ground_truth_new.npy",
    ):
        self.dataset_folder = Path(dataset_folder)
        self.modality = modality
        self.split = split
        self.transform = transform if transform else self._default_transform()
        self.use_text = use_text

        self._database_images_paths: List[str] = []
        self._query_images_paths: List[str] = []
        self.all_images_paths: List[str] = []
        self._database_num = 0
        self._queries_num = 0
        self._positives_per_query: List[np.ndarray] = []

        self._load_dataset(ground_truth_name)

    def _default_transform(self):
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_dataset(self, ground_truth_name: str):
        root = self.dataset_folder
        ref_dir = root / "ref"
        query_dir = root / "query"
        assert ref_dir.exists(), f"ref dir not found: {ref_dir}"
        assert query_dir.exists(), f"query dir not found: {query_dir}"

        self._database_images_paths = sorted([str(p) for p in ref_dir.glob("*.jpg")])
        self._query_images_paths = sorted([str(p) for p in query_dir.glob("*.jpg")])
        self._database_num = len(self._database_images_paths)
        self._queries_num = len(self._query_images_paths)
        self.all_images_paths = self._database_images_paths + self._query_images_paths

        gt_path = root / ground_truth_name
        if not gt_path.exists():
            # Fallback names used by RGB subset
            for alt in ["ground_truth_new.npy", "ground_truth.npy"]:
                gt_path = root / alt
                if gt_path.exists():
                    break

        assert gt_path.exists(), f"Ground truth not found in {root}"
        gt = np.load(gt_path, allow_pickle=True)
        # gt shape: (num_queries, 2) with rows [query_idx, list_of_ref_indices]
        self._positives_per_query = []
        for row in gt:
            positives = np.array(row[1], dtype=np.int64)
            self._positives_per_query.append(positives)

        assert len(self._positives_per_query) == self._queries_num, (
            f"GT has {len(self._positives_per_query)} queries, "
            f"but found {self._queries_num} query images"
        )

    def __getitem__(self, idx):
        img_path = self.all_images_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.use_text:
            return img, idx, "Scene description"
        return img, idx, ""

    def __len__(self):
        return len(self.all_images_paths)

    @property
    def database_num(self):
        return self._database_num

    @property
    def queries_num(self):
        return self._queries_num

    def get_positives(self):
        return self._positives_per_query
