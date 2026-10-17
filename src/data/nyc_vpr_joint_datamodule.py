"""Joint LightningDataModule for NYC-Event-VPR Event + RGB.

Trains a single model on both modalities simultaneously and evaluates on each
modality separately. The converted VG data is used for training/validation and
the VPR-Bench splits are used for testing.
"""
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.data.components.eventvpr_dataset import EventVPRDataset
from src.data.components.nyc_vpr_bench_dataset import NYCVPRBenchDataset


class _PlaceOffsetDataset(Dataset):
    """Wrapper that adds a fixed offset to place IDs.

    This prevents collisions between Event and RGB place IDs when both datasets
    are concatenated for joint training.
    """

    def __init__(self, base_dataset: Dataset, offset: int) -> None:
        self.base = base_dataset
        self.offset = offset

    def __getitem__(self, idx: int):
        imgs, pids, texts = self.base[idx]
        return imgs, pids + self.offset, texts

    def __len__(self) -> int:
        return len(self.base)


class NYCVPRJointDataModule(LightningDataModule):
    """Joint Event + RGB data module for NYC-Event-VPR experiments."""

    def __init__(
        self,
        converted_vg_root: str,
        vpr_bench_root: str,
        train_batch_size: int = 8,
        infer_batch_size: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True,
        use_text: bool = False,
        text_folder: Optional[str] = None,
        img_per_place: int = 5,
        min_img_per_place: int = 5,
        resize: tuple = (224, 224),
        rgb_place_offset: int = 1_000_000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val_event: Optional[EventVPRDataset] = None
        self.data_val_rgb: Optional[EventVPRDataset] = None
        self.data_test_event: Optional[NYCVPRBenchDataset] = None
        self.data_test_rgb: Optional[NYCVPRBenchDataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        import os

        vg_root = self.hparams.converted_vg_root
        vpr_root = self.hparams.vpr_bench_root
        offset = self.hparams.rgb_place_offset

        text_folder_event = self.hparams.text_folder or os.path.join(
            vg_root, "event", "scene_descriptions"
        )
        text_folder_rgb = self.hparams.text_folder or os.path.join(
            vg_root, "rgb", "scene_descriptions"
        )

        if stage in (None, "fit"):
            event_train = EventVPRDataset(
                dataset_folder=os.path.join(vg_root, "event"),
                split="train",
                use_text=self.hparams.use_text,
                text_folder=text_folder_event,
                img_per_place=self.hparams.img_per_place,
                min_img_per_place=self.hparams.min_img_per_place,
            )
            rgb_train = EventVPRDataset(
                dataset_folder=os.path.join(vg_root, "rgb"),
                split="train",
                use_text=self.hparams.use_text,
                text_folder=text_folder_rgb,
                img_per_place=self.hparams.img_per_place,
                min_img_per_place=self.hparams.min_img_per_place,
            )
            self.data_train = ConcatDataset(
                [
                    event_train,  # keep original IDs
                    _PlaceOffsetDataset(rgb_train, offset=offset),
                ]
            )

            self.data_val_event = EventVPRDataset(
                dataset_folder=os.path.join(vg_root, "event"),
                split="val",
                use_text=self.hparams.use_text,
                text_folder=text_folder_event,
            )
            self.data_val_rgb = EventVPRDataset(
                dataset_folder=os.path.join(vg_root, "rgb"),
                split="val",
                use_text=self.hparams.use_text,
                text_folder=text_folder_rgb,
            )

        if stage in (None, "test"):
            self.data_test_event = NYCVPRBenchDataset(
                dataset_folder=os.path.join(
                    vpr_root, "NYC-Event-VPR_VPR-Bench", "event_25m_0sobel_1fps"
                ),
                modality="event",
                use_text=self.hparams.use_text,
            )
            self.data_test_rgb = NYCVPRBenchDataset(
                dataset_folder=os.path.join(
                    vpr_root, "NYC-Event-VPR_VPR-Bench", "rgb_25m_1fps"
                ),
                modality="rgb",
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

    def val_dataloader(self) -> list[DataLoader[Any]]:
        return [
            DataLoader(
                dataset=self.data_val_event,
                batch_size=self.hparams.infer_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.data_val_rgb,
                batch_size=self.hparams.infer_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
        ]

    def test_dataloader(self) -> list[DataLoader[Any]]:
        return [
            DataLoader(
                dataset=self.data_test_event,
                batch_size=self.hparams.infer_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
            DataLoader(
                dataset=self.data_test_rgb,
                batch_size=self.hparams.infer_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            ),
        ]

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
