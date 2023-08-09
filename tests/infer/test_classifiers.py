import lightning as L
import pytest
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from seqinfer.infer.classifiers import (
    DEFAULT_BINARY_CLASSIFICATION_METRICS,
    LitClassifier,
)


class TestLitClassifier:
    """Unit test class for LitClassifier"""

    def setup_method(self):
        """Define dummy data and model for testing"""
        self.num_classes = 5
        self.model = nn.Linear(10, self.num_classes)
        self.is_output_logits = True
        self.loss = nn.CrossEntropyLoss()
        self.metrics = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)]
        )
        self.lr_scheduler_path = "torch.optim.lr_scheduler.LinearLR"
        self.lit_classifier = LitClassifier(
            num_classes=self.num_classes,
            model=self.model,
            is_output_logits=self.is_output_logits,
            loss=self.loss,
            metrics=self.metrics,
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
        return torch.randint(0, self.num_classes, (16,))  # Batch size 16, random integer labels

    def test_forward(self, feat):
        """Test forward method"""
        output = self.lit_classifier(feat)
        assert output.shape == (16, self.num_classes)

    def test_configure_optimizers(self):
        """Test configure_optimizers method"""
        optimizers, lr_schedulers = self.lit_classifier.configure_optimizers()
        assert len(optimizers) == 1, f"Expected 1 optimizer but got {len(optimizers)}"
        assert isinstance(optimizers[0], torch.optim.Optimizer), f"Not {type(optimizers[0])}"
        assert len(lr_schedulers) == 1, f"Expected 1 lr_scheduler but got {len(lr_schedulers)}"
        assert isinstance(
            lr_schedulers[0], torch.optim.lr_scheduler.LRScheduler
        ), f"Not {type(lr_schedulers[0])}"

    def test_training_step(self, feat, target):
        """Test training_step method"""
        batch = (feat, target)
        loss = self.lit_classifier.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_validation_step(self, feat, target):
        """Test validation_step method"""
        batch = (feat, target)
        loss = self.lit_classifier.validation_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_test_step(self, feat, target):
        """Test test_step method"""
        batch = (feat, target)
        loss = self.lit_classifier.test_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_predict_step(self, feat):
        """Test predict_step method"""
        batch = (feat, None)  # Since we don't need targets for prediction
        output = self.lit_classifier.predict_step(batch, batch_idx=0)
        assert output.shape == (16, self.num_classes)

    def test_metrics_logging(self, feat, target):
        """Test if the metrics are correctly logged during validation and test steps."""
        dummy_dataset = TensorDataset(feat, target)
        dummy_train_dataloader = DataLoader(Subset(dummy_dataset, range(10)), batch_size=4)
        dummy_val_dataloader = DataLoader(Subset(dummy_dataset, range(10, 13)), batch_size=4)
        dummy_test_dataloader = DataLoader(Subset(dummy_dataset, range(13, 16)), batch_size=4)

        # Train
        self.trainer.fit(self.lit_classifier, dummy_train_dataloader)

        # Validation
        self.trainer.validate(self.lit_classifier, dummy_val_dataloader)
        with torch.no_grad():
            val_output = self.lit_classifier.model(feat[range(10, 13)])
            val_loss = self.loss(val_output, target[range(10, 13)])
        val_accuracy = (val_output.argmax(dim=1) == target[range(10, 13)]).float().mean()
        torch.testing.assert_close(self.trainer.logged_metrics["val_loss"], val_loss)
        torch.testing.assert_close(
            self.trainer.logged_metrics["val_MulticlassAccuracy"], val_accuracy
        )

        # Test
        self.trainer.test(self.lit_classifier, dummy_test_dataloader)
        with torch.no_grad():
            test_output = self.lit_classifier.model(feat[range(13, 16)])
            test_loss = self.loss(test_output, target[range(13, 16)])
        test_accuracy = (test_output.argmax(dim=1) == target[range(13, 16)]).float().mean()
        torch.testing.assert_close(self.trainer.logged_metrics["test_loss"], test_loss)
        torch.testing.assert_close(
            self.trainer.logged_metrics["test_MulticlassAccuracy"], test_accuracy
        )


class TestBinaryClassificationMetrics:
    """Tests for the default binary classification metrics."""

    def setup_method(self):
        """Setup with some dummy predictions and targets."""
        self.metrics = DEFAULT_BINARY_CLASSIFICATION_METRICS
        self.target = torch.randint(0, 2, (16,))  # Batch size 16, random integer labels
        self.pred_logits = torch.randn(16)
        self.prob = self.pred_logits.sigmoid()

    def test_default_binary_classification_metrics_key(self):
        """Test that default metrics contains Accuracy, AUROC, AveragePrecision."""
        classification_metrics_prob = self.metrics(self.pred_logits, self.target)

        assert "BinaryAccuracy" in classification_metrics_prob
        assert "BinaryAUROC" in classification_metrics_prob
        assert "BinaryAveragePrecision" in classification_metrics_prob

    def test_equality_logits_prob(self):
        """Test that metrics are same for logit and probability predictions. This is due to the
        functionality of torchmetrics"""
        classification_metrics_logits = self.metrics(self.prob, self.target)
        classification_metrics_prob = self.metrics(self.pred_logits, self.target)

        assert classification_metrics_logits == classification_metrics_prob
