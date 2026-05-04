from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components.backbone.text_encoder_clip import CLIPTextEncoder
from src.models.components.loss import loss_function
from src.models.components.network import SGVPRNet


class EPRBenchLitModule(LightningModule):
    """LightningModule for EPRBench (SGVPR) visual place recognition.

    This module replicates the original EPRBench training and evaluation logic
    using PyTorch Lightning and Hydra configuration.
    """

    def __init__(
        self,
        net: SGVPRNet,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lambda_contrast: float = 0.1,
        temperature: float = 0.07,
        use_text: bool = True,
        features_dim: int = 10752,
        test_method: str = "hard_resize",
        recall_values: List[int] = [1, 5, 10, 100],
        infer_batch_size: int = 8,
        num_workers: int = 4,
        pca_dim: Optional[int] = None,
        efficient_ram_testing: bool = False,
        majority_weight: float = 0.01,
        compile: bool = False,
    ) -> None:
        """Initialize the EPRBenchLitModule.

        :param net: The SGVPRNet model.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler.
        :param lambda_contrast: Weight for contrastive learning loss.
        :param temperature: Temperature parameter for contrastive learning.
        :param use_text: Whether to use text descriptions.
        :param features_dim: Dimension of output features (14*768).
        :param test_method: Inference test method.
        :param recall_values: List of recall values to compute.
        :param infer_batch_size: Batch size for inference.
        :param num_workers: Number of workers for dataloaders.
        :param pca_dim: PCA dimension, None to disable.
        :param efficient_ram_testing: Use RAM-efficient testing.
        :param majority_weight: Weight for majority voting.
        :param compile: Whether to compile the model with torch.compile.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.text_encoder = CLIPTextEncoder(
            model_name="ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            output_dim=768,
            freeze_clip=True,
        )

        # Freeze backbone parameters except adapter
        for name, param in self.net.backbone.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

        # Initialize Adapter weights
        for n, m in self.net.named_modules():
            if "adapter" in n:
                for n2, m2 in m.named_modules():
                    if "D_fc2" in n2 and isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.0)
                        nn.init.constant_(m2.bias, 0.0)
                for n2, m2 in m.named_modules():
                    if "conv" in n2 and isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)

        # loss function is imported from components.loss
        self.criterion = loss_function

        # metric objects for calculating and averaging losses across batches
        self.train_loss_total = MeanMetric()
        self.train_loss_tri = MeanMetric()
        self.train_loss_i2t = MeanMetric()
        self.train_loss_t2i = MeanMetric()

        # for tracking best so far validation R@5
        self.val_r5_best = MaxMetric()

        # validation/test feature storage
        self._val_features: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_features: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []

    def forward(
        self,
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: Input images.
        :param text: Sentence-level text embeddings.
        :param text_tokens: Token-level text embeddings.
        :param text_mask: Attention mask for text tokens.
        :param return_aux: Whether to return auxiliary outputs.
        :return: Model output descriptors.
        """
        return self.net(
            x,
            text=text,
            text_tokens=text_tokens,
            text_mask=text_mask,
            return_aux=return_aux,
        )

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_r5_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data.

        :param batch: A batch of data containing images, place_ids, and optional text.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses.
        """
        images, place_id, text = batch

        BS, N, C, H, W = images.shape
        # Flatten batch and place dimensions
        images = images.view(BS * N, C, H, W)
        labels = place_id.view(BS * N)

        if text is not None and self.hparams.use_text:
            # Flatten nested text list
            text = [t for sublist in text for t in sublist]
            # Sentence embeddings (BS*N, 768)
            text_vec = self.text_encoder(text)
            # Token embeddings and mask
            text_tokens, attn_mask = self.text_encoder.encode_batch_tokens(text)
            # Forward pass with text conditioning
            descriptors, image_global_raw = self.forward(
                images,
                text=text_vec,
                text_tokens=text_tokens,
                text_mask=attn_mask,
                return_aux=True,
            )
            # Aggregate per-place features for contrastive loss
            image_global_raw_place = image_global_raw.view(BS, N, -1).mean(dim=1)
            text_vec_place = text_vec.view(BS, N, -1).mean(dim=1)
        else:
            descriptors = self.forward(images)
            image_global_raw_place = None
            text_vec_place = None

        # Compute combined loss
        loss, tri_loss, loss_i2t, loss_t2i = self.criterion(
            descriptors,
            labels,
            image_global_raw=image_global_raw_place,
            text_global_raw=text_vec_place,
            lambda_contrast=self.hparams.lambda_contrast,
            temperature=self.hparams.temperature,
        )

        # update and log metrics
        self.train_loss_total(loss)
        self.train_loss_tri(tri_loss)
        self.train_loss_i2t(loss_i2t)
        self.train_loss_t2i(loss_t2i)

        self.log(
            "train/loss",
            self.train_loss_total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/tri_loss",
            self.train_loss_tri,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/i2t_loss",
            self.train_loss_i2t,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/t2i_loss",
            self.train_loss_t2i,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def on_validation_epoch_start(self) -> None:
        """Lightning hook called at the start of validation epoch."""
        self._val_features.clear()
        self._val_indices.clear()

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data.

        :param batch: A batch of data containing images, indices, and optional text.
        :param batch_idx: The index of the current batch.
        """
        inputs, indices, text_desc = batch

        if text_desc is not None and self.hparams.use_text:
            text_list = list(text_desc)
            text_vec = self.text_encoder(text_list)
            text_tokens, attn_mask = self.text_encoder.encode_batch_tokens(text_list)
            features = self.forward(
                inputs,
                text=text_vec,
                text_tokens=text_tokens,
                text_mask=attn_mask,
            )
        else:
            features = self.forward(inputs)

        features = features.cpu()
        self._val_features.append(features)
        self._val_indices.append(indices.cpu())

    def on_validation_epoch_end(self) -> None:
        """Lightning hook called at the end of validation epoch.

        Computes recalls using FAISS search over the accumulated validation features.
        """
        if len(self._val_features) == 0:
            return

        all_features = torch.cat(self._val_features, dim=0).numpy()
        all_indices = torch.cat(self._val_indices, dim=0).numpy()

        eval_ds = self.trainer.datamodule.data_val
        database_num = eval_ds.database_num
        queries_num = eval_ds.queries_num

        # Sort features by their original dataset indices
        sorted_features = np.empty(
            (len(eval_ds), self.hparams.features_dim), dtype=np.float32
        )
        sorted_features[all_indices] = all_features

        # Split into database and queries
        database_features = sorted_features[:database_num]
        queries_features = sorted_features[database_num:]

        # FAISS search
        faiss_index = faiss.IndexFlatL2(self.hparams.features_dim)
        faiss_index.add(database_features)

        _, predictions = faiss_index.search(
            queries_features, max(self.hparams.recall_values)
        )

        # Compute recalls
        positives_per_query = eval_ds.get_positives()
        recalls = np.zeros(len(self.hparams.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(self.hparams.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        recalls = recalls / queries_num * 100

        # Log recalls
        for val, rec in zip(self.hparams.recall_values, recalls):
            self.log(f"val/R@{val}", float(rec), on_epoch=True, prog_bar=(val == 5))

        r5 = float(recalls[1]) if len(recalls) > 1 else 0.0
        self.val_r5_best(r5)
        self.log(
            "val/R@5_best", self.val_r5_best.compute(), on_epoch=True, prog_bar=True
        )

    def on_test_epoch_start(self) -> None:
        """Lightning hook called at the start of test epoch."""
        self._test_features.clear()
        self._test_indices.clear()

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Perform a single test step on a batch of data.

        :param batch: A batch of data containing images, indices, and optional text.
        :param batch_idx: The index of the current batch.
        """
        inputs, indices, text_desc = batch

        if text_desc is not None and self.hparams.use_text:
            text_list = list(text_desc)
            text_vec = self.text_encoder(text_list)
            text_tokens, attn_mask = self.text_encoder.encode_batch_tokens(text_list)
            features = self.forward(
                inputs,
                text=text_vec,
                text_tokens=text_tokens,
                text_mask=attn_mask,
            )
        else:
            features = self.forward(inputs)

        features = features.cpu()
        self._test_features.append(features)
        self._test_indices.append(indices.cpu())

    def on_test_epoch_end(self) -> None:
        """Lightning hook called at the end of test epoch.

        Computes recalls using FAISS search over the accumulated test features.
        """
        if len(self._test_features) == 0:
            return

        all_features = torch.cat(self._test_features, dim=0).numpy()
        all_indices = torch.cat(self._test_indices, dim=0).numpy()

        eval_ds = self.trainer.datamodule.data_test
        database_num = eval_ds.database_num
        queries_num = eval_ds.queries_num

        # Sort features by their original dataset indices
        sorted_features = np.empty(
            (len(eval_ds), self.hparams.features_dim), dtype=np.float32
        )
        sorted_features[all_indices] = all_features

        # Split into database and queries
        database_features = sorted_features[:database_num]
        queries_features = sorted_features[database_num:]

        # FAISS search
        faiss_index = faiss.IndexFlatL2(self.hparams.features_dim)
        faiss_index.add(database_features)

        _, predictions = faiss_index.search(
            queries_features, max(self.hparams.recall_values)
        )

        # Compute recalls
        positives_per_query = eval_ds.get_positives()
        recalls = np.zeros(len(self.hparams.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(self.hparams.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        recalls = recalls / queries_num * 100

        # Log recalls
        for val, rec in zip(self.hparams.recall_values, recalls):
            self.log(f"test/R@{val}", float(rec), on_epoch=True, prog_bar=(val == 5))

        recalls_str = ", ".join(
            [
                f"R@{val}: {rec:.1f}"
                for val, rec in zip(self.hparams.recall_values, recalls)
            ]
        )
        print(f"Test recalls: {recalls_str}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Lightning hook called at the beginning of fit, validate, test, or predict.

        :param stage: Either ``"fit"``, ``"validate"``, ``"test"``, or ``"predict"``.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use.

        :return: A dict containing the configured optimizers and learning-rate schedulers.
        """
        # Replicate original param grouping: model params and text encoder params
        model_params = [p for p in self.net.parameters() if p.requires_grad]
        text_params = [p for p in self.text_encoder.parameters() if p.requires_grad]

        optimizer = self.hparams.optimizer(
            params=[{"params": model_params}, {"params": text_params}]
        )

        if self.hparams.scheduler is not None:
            # Compute step_size based on train dataloader length
            # scheduler step is per batch (interval="step")
            train_ds = self.trainer.datamodule.data_train
            train_batch_size = self.trainer.datamodule.hparams.train_batch_size
            steps_per_epoch = (len(train_ds) + train_batch_size - 1) // train_batch_size
            step_size = steps_per_epoch * 3

            scheduler = self.hparams.scheduler(optimizer=optimizer, step_size=step_size)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
