from pathlib import Path

import pytest
import torch

from src.data.eprbench_datamodule import EPRBenchDataModule


@pytest.mark.parametrize("batch_size", [2, 4])
def test_eprbench_datamodule(batch_size: int) -> None:
    """Tests `EPRBenchDataModule` to verify that it can be set up correctly,
    that the necessary attributes were created (e.g., the dataloader objects),
    and that dtypes and batch sizes correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/EventVPR"

    dm = EPRBenchDataModule(
        dataset_folder=data_dir,
        train_batch_size=batch_size,
        infer_batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # Check train batch (may be smaller than batch_size if dataset is tiny)
    batch = next(iter(dm.train_dataloader()))
    images, place_id, text = batch
    assert len(images) <= batch_size
    assert images.dtype == torch.float32

    # Check val batch
    batch = next(iter(dm.val_dataloader()))
    img, idx, text_desc = batch
    assert len(img) <= batch_size
    assert img.dtype == torch.float32
