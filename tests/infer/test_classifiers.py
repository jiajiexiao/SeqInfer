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
        self.l1_loss_coef = 0.01
        self.metrics = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)]
        )
        self.lr_scheduler_path = "torch.optim.lr_scheduler.LinearLR"
        self.lit_classifier = LitClassifier(
            num_classes=self.num_classes,
            model=self.model,
            is_output_logits=self.is_output_logits,
            loss=self.loss,
            l1_loss_coef=self.l1_loss_coef,
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

    def test_multi_class_probs(self, feat):
        """Test probability output for multi-class case"""
        output = self.lit_classifier.predict_prob(feat)
        batch_size = feat.shape[0]
        assert output.shape == (
            batch_size,
            self.num_classes,
        )  # output probabilities for all classes
        torch.testing.assert_close(output.sum(dim=1), torch.ones(batch_size))  # probs sum to 1

    def test_binary_class_probs(self, feat):
        """Test probability output for binary class case"""
        self.lit_classifier.num_classes = 2
        self.lit_classifier.model = nn.Linear(10, 1)
        output = self.lit_classifier.predict_prob(feat)
        torch.testing.assert_close(output, torch.sigmoid(self.lit_classifier.model(feat)))

    def test_probs_from_probs(self, feat):
        """Test probability method when model already outputs probabilities"""
        self.lit_classifier.is_output_logits = False
        output = self.lit_classifier.predict_prob(feat)
        assert output.shape == (feat.shape[0], self.num_classes)
        torch.testing.assert_close(output, self.lit_classifier.model(feat))

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
            l1_reg = self.l1_loss_coef * torch.mean(
                torch.concatenate(
                    [
                        torch.abs(param)
                        for name, param in self.lit_classifier.model.named_parameters()
                        if "bias" not in name and param.requires_grad
                    ]
                )
            )
            val_output = self.lit_classifier.model(feat[range(10, 13)])
            val_loss = self.loss(val_output, target[range(10, 13)]) + l1_reg
        val_accuracy = (val_output.argmax(dim=1) == target[range(10, 13)]).float().mean()
        torch.testing.assert_close(self.trainer.logged_metrics["val_loss"], val_loss)
        torch.testing.assert_close(
            self.trainer.logged_metrics["val_MulticlassAccuracy"], val_accuracy
        )

        # Test
        self.trainer.test(self.lit_classifier, dummy_test_dataloader)
        with torch.no_grad():
            test_output = self.lit_classifier.model(feat[range(13, 16)])
            test_loss = self.loss(test_output, target[range(13, 16)]) + l1_reg
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
        """Test that default metrics contains Accuracy, AUROC."""
        classification_metrics_prob = self.metrics(self.pred_logits, self.target)

        assert "BinaryAccuracy" in classification_metrics_prob
        assert "BinaryAUROC" in classification_metrics_prob
        # assert "BinaryAveragePrecision" in classification_metrics_prob

    def test_equality_logits_prob(self):
        """Test that metrics are same for logit and probability predictions. This is due to the
        functionality of torchmetrics"""
        classification_metrics_logits = self.metrics(self.prob, self.target)
        classification_metrics_prob = self.metrics(self.pred_logits, self.target)

        assert classification_metrics_logits == classification_metrics_prob
