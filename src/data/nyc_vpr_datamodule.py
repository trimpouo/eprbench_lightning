"""LightningDataModule for NYC-Event-VPR.

- Train/val use converted VG data (place-based) via EventVPRDataset.
- Test uses VPR-Bench data via NYCVPRBenchDataset.
- Text is disabled by default because NYC HF data has no scene descriptions.
"""
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.eventvpr_dataset import EventVPRDataset
from src.data.components.nyc_vpr_bench_dataset import NYCVPRBenchDataset


class NYCVPRDataModule(LightningDataModule):
    def __init__(
        self,
        converted_vg_root: str,
        vpr_bench_root: str,
        modality: str = "event",  # "event" or "rgb"
        train_batch_size: int = 8,
        infer_batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_text: bool = False,
        text_folder: Optional[str] = None,
        img_per_place: int = 5,
        min_img_per_place: int = 5,
        resize: tuple = (224, 224),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[EventVPRDataset] = None
        self.data_val: Optional[EventVPRDataset] = None
        self.data_test: Optional[NYCVPRBenchDataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        import os

        vg_root = self.hparams.converted_vg_root
        modality = self.hparams.modality
        # Subfolder names used by convert_nyc_vg_to_eprbench.py
        mod_folder = modality.lower()  # "event" or "rgb"

        text_folder = self.hparams.text_folder or os.path.join(
            vg_root, mod_folder, "scene_descriptions"
        )

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = EventVPRDataset(
                dataset_folder=os.path.join(vg_root, mod_folder),
                split="train",
                use_text=self.hparams.use_text,
                text_folder=text_folder,
                img_per_place=self.hparams.img_per_place,
                min_img_per_place=self.hparams.min_img_per_place,
            )
            self.data_val = EventVPRDataset(
                dataset_folder=os.path.join(vg_root, mod_folder),
                split="val",
                use_text=self.hparams.use_text,
                text_folder=text_folder,
            )
            self.data_test = NYCVPRBenchDataset(
                dataset_folder=os.path.join(
                    self.hparams.vpr_bench_root,
                    "NYC-Event-VPR_VPR-Bench",
                    f"{mod_folder}_25m_0sobel_1fps" if modality == "event" else "rgb_25m_1fps",
                ),
                modality=modality,
                use_text=self.hparams.use_text,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.infer_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.infer_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
