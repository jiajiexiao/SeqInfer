import lightning as L
import pytest
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from seqinfer.infer.components.ligntning_modules import BaseLitModule


class TestBaseLitModule:
    """Unit test class for BaseLitModule"""

    def setup_method(self):
        """Define dummy data and model for testing"""
        self.out_dim = 5
        self.model = nn.Linear(10, self.out_dim)
        self.loss = nn.CrossEntropyLoss()
        self.l1_loss_coef = 0.01
        self.lr_scheduler_path = "torch.optim.lr_scheduler.LinearLR"
        self.lit_model = BaseLitModule(
            model=self.model,
            loss=self.loss,
            l1_loss_coef=self.l1_loss_coef,
            metrics=None,
            lr_scheduler_path=self.lr_scheduler_path,
        )
        self.trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)

    @pytest.fixture
    def feat(self):
        """Pytest fixture for usable feat"""
        return torch.randn(16, 10)  # Batch size 16, 10 features

    @pytest.fixture
    def target(self):
        """Pytest fixture for usable target"""
        return torch.randint(0, self.out_dim, (16,))  # Batch size 16, random integer labels

    def test_forward(self, feat):
        """Test forward method"""
        output = self.lit_model(feat)
        assert output.shape == (16, self.out_dim)

    def test_configure_optimizers(self):
        """Test configure_optimizers method"""
        optimizers, lr_schedulers = self.lit_model.configure_optimizers()
        assert len(optimizers) == 1, f"Expected 1 optimizer but got {len(optimizers)}"
        assert isinstance(optimizers[0], torch.optim.Optimizer), f"Not {type(optimizers[0])}"
        assert len(lr_schedulers) == 1, f"Expected 1 lr_scheduler but got {len(lr_schedulers)}"
        assert isinstance(
            lr_schedulers[0], torch.optim.lr_scheduler.LRScheduler
        ), f"Not {type(lr_schedulers[0])}"

    def test_training_step(self, feat, target):
        """Test training_step method"""
        batch = (feat, target)
        loss = self.lit_model.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_validation_step(self, feat, target):
        """Test validation_step method"""
        batch = (feat, target)
        loss = self.lit_model.validation_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_test_step(self, feat, target):
        """Test test_step method"""
        batch = (feat, target)
        loss = self.lit_model.test_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_predict_step(self, feat):
        """Test predict_step method"""
        batch = (feat, None)  # Since we don't need targets for prediction
        output = self.lit_model.predict_step(batch, batch_idx=0)
        assert output.shape == (16, self.out_dim)
