from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.eventvpr_dataset import EventVPRDataset


class EPRBenchDataModule(LightningDataModule):
    """LightningDataModule for the EPRBench EventVPR dataset.

    A LightningDataModule implements the standard data pipeline:
    ```python
        def prepare_data(self):
            # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
            # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
            # Things to do on every process in DDP.
            # Load data, set variables, etc...

        def train_dataloader(self):
            # return train dataloader

        def val_dataloader(self):
            # return validation dataloader

        def test_dataloader(self):
            # return test dataloader
    ```

    This module replicates the original EPRBench data loading logic where:
    - Training uses the "train" split of EventVPR.
    - Validation uses the "test" split (mirroring original behavior).
    - Testing uses the "test" split.
    """

    def __init__(
        self,
        dataset_folder: str,
        train_batch_size: int = 8,
        infer_batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_text: bool = True,
        text_folder: Optional[str] = None,
        img_per_place: int = 5,
        min_img_per_place: int = 5,
        resize: tuple = (224, 224),
    ) -> None:
        """Initialize the EPRBenchDataModule.

        :param dataset_folder: Path to the dataset folder.
        :param train_batch_size: Batch size for training.
        :param infer_batch_size: Batch size for inference (validation/testing).
        :param num_workers: Number of workers for dataloaders.
        :param pin_memory: Whether to pin memory in dataloaders.
        :param use_text: Whether to use text descriptions.
        :param text_folder: Path to text descriptions folder.
        :param img_per_place: Number of images to sample per place during training.
        :param min_img_per_place: Minimum number of images required per place.
        :param resize: Image resize dimensions (H, W).
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[EventVPRDataset] = None
        self.data_val: Optional[EventVPRDataset] = None
        self.data_test: Optional[EventVPRDataset] = None

    def prepare_data(self) -> None:
        """Download data if needed.

        Lightning ensures that ``self.prepare_data()`` is called only within a single process on CPU,
        so you can safely add your downloading logic within. In case of multi-node training, the
        execution of this hook depends upon ``self.prepare_data_per_node()``.

        Do not use it to assign state (self.x = y).
        """
        # Data is assumed to be already downloaded and prepared.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: ``self.data_train``, ``self.data_val``, ``self.data_test``.

        This method is called by Lightning before ``trainer.fit()``, ``trainer.validate()``,
        ``trainer.test()``, and ``trainer.predict()``.

        :param stage: The stage to setup. Either ``"fit"``, ``"validate"``, ``"test"``, or ``"predict"``.
        """
        import os

        text_folder = self.hparams.text_folder or os.path.join(
            self.hparams.dataset_folder, "scene_descriptions"
        )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = EventVPRDataset(
                dataset_folder=self.hparams.dataset_folder,
                split="train",
                use_text=self.hparams.use_text,
                text_folder=text_folder,
                img_per_place=self.hparams.img_per_place,
                min_img_per_place=self.hparams.min_img_per_place,
            )
            # Original EPRBench uses test split for validation
            self.data_val = EventVPRDataset(
                dataset_folder=self.hparams.dataset_folder,
                split="test",
                use_text=self.hparams.use_text,
                text_folder=text_folder,
            )
            self.data_test = EventVPRDataset(
                dataset_folder=self.hparams.dataset_folder,
                split="test",
                use_text=self.hparams.use_text,
                text_folder=text_folder,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.infer_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.infer_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after ``fit()``, ``validate()``, ``test()``, and ``predict()``.

        :param stage: The stage being torn down.
        """
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint.

        :return: A dictionary containing the datamodule state.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint.

        :param state_dict: The datamodule state.
        """
        pass
