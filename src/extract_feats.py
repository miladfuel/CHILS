import os
from typing import List, Optional
import numpy as np
import torch
import pytorch_lightning as pl
from src.model_utils import get_model
from tqdm import tqdm
class EvalNet(pl.LightningModule):
    def __init__(
        self,
        arch: str = "ClipViTL14",
        retrain: bool = False,
        base_task: str = "food-101",
        pretrained: bool = True,
        target_dataset: List[str] = [],
        work_dir: str = ".",
        max_epochs: int = 1,
        hash: Optional[str] = None
    ):
        super().__init__()

        self.arch = arch
        self.base_task = base_task
        self.model = get_model(arch=arch, dataset=base_task, pretrained=pretrained, retrain=False, extract_features=True, work_dir=work_dir)

        self.model.init_text(base_task.lower())
        self.target_dataset = target_dataset

        self.work_dir = work_dir
        self.hash = hash
        self.pretrained = pretrained
        self.test_outputs = []
        self.current_batch = 0
        self.total_batches = None

        print("target_dataset:", target_dataset)
        print("work_dir:", work_dir)

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, _ = batch[:2]
        return self.model(x)

    def process_batch(self, batch, stage="train", dataloader_idx=0):
        x, y, idx = batch[:3]
        output = self.forward(x)
        logits = output["logits"]
        features = torch.flatten(output["features"], 1)
        
        _, pred_idx = torch.max(logits, dim=1)
        return  y, pred_idx, idx, features

    def training_step(self, batch, batch_idx: int):
        _ = self.process_batch(batch, "pred")
        return None

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        labels, outputs, idx, features = self.process_batch(batch, "pred", dataloader_idx)
        self.test_outputs.append((labels, outputs, idx, features))  # Append to test_outputs
        return labels, outputs, idx, features

    def on_test_epoch_end(self):
        print("on_test_epoch_end called")
        self.gen_confidence_files(self.test_outputs, "test")
        self.test_outputs = []

    def gen_confidence_files(self, outputs_list, stage):
        print(f"Generating .npz file for {stage} stage")

        if not outputs_list:
            print("No outputs to save. outputs_list is empty.")
            return

        print(f"Contents of outputs_list: {outputs_list}")

        labels = torch.cat([x[0] for x in outputs_list])
        outputs = torch.cat([x[1] for x in outputs_list])
        idx = torch.cat([x[2] for x in outputs_list])
        features = torch.cat([x[3] for x in outputs_list])

        if not os.path.exists(self.work_dir + f"/{self.arch}"):
            os.mkdir(self.work_dir + f"/{self.arch}")
        np.savez(self.work_dir + f"/{self.arch}/conf_" + self.base_task.lower() + ".npz",
                 labels=labels.detach().cpu().numpy(),
                 outputs=outputs.detach().cpu().numpy(),
                 indices=idx.detach().cpu().numpy(),
                 features=features.detach().cpu().numpy())

    def configure_optimizers(self):
        pass

