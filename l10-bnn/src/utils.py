import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm

FLOAT_EPS = torch.finfo(torch.float32).eps
CLASSES = np.arange(10)

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


class Analyzer:
    def __init__(self, model: nn.Module, dataset: Dataset, num_samplings: int):
        self.model = model
        self.dataset = dataset
        self.num_samplings = num_samplings

        self.model.eval()

        self._preds: Optional[torch.Tensor] = None
        self._trues: Optional[torch.Tensor] = None
        self._images: Optional[torch.Tensor] = None

        self._retrieve_predictions()

    def reset_state(self):
        self._preds = None
        self._trues = None

    def _retrieve_predictions(self):
        loader = DataLoader(
            self.dataset, batch_size=32, shuffle=False, drop_last=False
        )

        trues = []
        preds = []
        images = []
        for inputs, targets in loader:
            images.append(inputs[:, 0])
            inputs = inputs.view((-1, 28 * 28))
            trues.append(targets)
            if self.num_samplings <= 1:
                preds.append(self.model(inputs).detach())
            else:
                loc_preds = []
                for _ in range(self.num_samplings):
                    loc_preds.append(self.model(inputs).detach())
                preds.append(torch.stack(loc_preds, dim=-1))

        self._preds = torch.cat(preds, dim=0)
        self._trues = torch.cat(trues, dim=0)
        self._images = torch.cat(images, dim=0)

    @classmethod
    def _get_entropy(cls, matrix: torch.Tensor) -> torch.Tensor:
        return -(matrix * matrix.clamp_min(FLOAT_EPS).log()).sum(dim=-1)

    def get_top_k_high_confidence_mistakes(
        self, k: int = 10
    ) -> Tuple[torch.Tensor, ...]:
        if len(self._preds.shape) == 3:
            mean_preds = self._preds.mean(dim=-1)
        else:
            mean_preds = self._preds
        where_mistakes = mean_preds.argmax(dim=-1) != self._trues.argmax(
            dim=-1
        )
        preds_with_mistakes = mean_preds[where_mistakes]
        images = self._images[where_mistakes]
        trues = self._trues[where_mistakes]

        top_high_confidence_indices = self._get_entropy(
            preds_with_mistakes
        ).argsort(descending=False)[:k]
        return (
            images[top_high_confidence_indices],
            self._preds[where_mistakes][top_high_confidence_indices],
            trues[top_high_confidence_indices],
        )

    def get_top_k_low_confidence_mistakes(
        self, k: int = 10
    ) -> Tuple[torch.Tensor, ...]:
        if len(self._preds.shape) == 3:
            mean_preds = self._preds.mean(dim=-1)
        else:
            mean_preds = self._preds
        where_mistakes = mean_preds.argmax(dim=-1) != self._trues.argmax(
            dim=-1
        )
        preds_with_mistakes = mean_preds[where_mistakes]
        images = self._images[where_mistakes]
        trues = self._trues[where_mistakes]

        top_low_confidence_indices = self._get_entropy(
            preds_with_mistakes
        ).argsort(descending=True)[:k]
        return (
            images[top_low_confidence_indices],
            self._preds[where_mistakes][top_low_confidence_indices],
            trues[top_low_confidence_indices],
        )

    def get_top_k_high_confidence_correct(
        self, k: int = 10
    ) -> Tuple[torch.Tensor, ...]:
        if len(self._preds.shape) == 3:
            mean_preds = self._preds.mean(dim=-1)
        else:
            mean_preds = self._preds
        where_correct = mean_preds.argmax(dim=-1) == self._trues.argmax(dim=-1)
        preds_with_mistakes = mean_preds[where_correct]
        images = self._images[where_correct]
        trues = self._trues[where_correct]

        top_high_confidence_indices = self._get_entropy(
            preds_with_mistakes
        ).argsort(descending=False)[:k]
        return (
            images[top_high_confidence_indices],
            self._preds[where_correct][top_high_confidence_indices],
            trues[top_high_confidence_indices],
        )

    def get_top_k_low_confidence_correct(
        self, k: int = 10
    ) -> Tuple[torch.Tensor, ...]:
        if len(self._preds.shape) == 3:
            mean_preds = self._preds.mean(dim=-1)
        else:
            mean_preds = self._preds
        where_correct = mean_preds.argmax(dim=-1) == self._trues.argmax(dim=-1)
        preds_with_mistakes = mean_preds[where_correct]
        images = self._images[where_correct]
        trues = self._trues[where_correct]

        top_low_confidence_indices = self._get_entropy(
            preds_with_mistakes
        ).argsort(descending=True)[:k]
        return (
            images[top_low_confidence_indices],
            self._preds[where_correct][top_low_confidence_indices],
            trues[top_low_confidence_indices],
        )

    def get_weight_vector(self) -> np.ndarray:
        weights = []
        for params in self.model.parameters():
            if params.requires_grad:
                weights.append(params.view((-1,)))
        weights = torch.cat(weights).detach().numpy()
        return weights


def visualize_samples(
    images: torch.Tensor,
    preds: torch.Tensor,
    trues: torch.Tensor,
    num_cols: int = 10,
) -> plt.Figure:
    num_rows = len(images) // num_cols
    if num_rows == 0:
        num_rows = 1
    num_rows *= 2

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(3 * num_cols, 2 * num_rows * 2)
    )
    axes = np.array(axes)

    is_with_multiple_samplings = len(preds.shape) == 3

    x = 0
    y = 0

    for image, pred, true in zip(
        images, preds, trues
    ):  # type: torch.Tensor, torch.Tensor, torch.Tensor
        ax_hist: plt.Axes = axes[y, x]
        ax_image: plt.Axes = axes[y + 1, x]

        if is_with_multiple_samplings:
            pred_mean = pred.mean(dim=-1)
            pred_std = pred.std(dim=-1)
        else:
            pred_mean = pred
            pred_std = 0

        pred_class = pred_mean.argmax().item()
        true_class = true.argmax().item()
        ax_hist.set_title(f"Pred: {pred_class} | True: {true_class}")
        ax_hist.bar(CLASSES, pred_mean, yerr=pred_std)

        if pred_class == true_class:
            ax_hist.bar([pred_class], pred_mean[[pred_class]], color="green")
        else:
            ax_hist.bar([true_class], pred_mean[[true_class]], color="yellow")
            ax_hist.bar([pred_class], pred_mean[[pred_class]], color="red")

        ax_image.imshow((image.numpy() * 255).astype(np.uint8), cmap="gray")
        ax_image.set_axis_off()
        x += 1
        if x == num_cols:
            x = 0
            y += 2
    return fig


def visualize_weights(params: torch.Tensor, name: str) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.kdeplot(params.view((-1,)).detach().numpy(), ax=ax)
    ax.set_title(f"Kernel density plot of {name}")
    return fig


def fit_elbo(
    model: nn.Module,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    loss_function: nn.Module,
    batch_size: int,
    epochs: int,
    optimizer: Optimizer,
) -> Tuple[Dict[str, List[float]], ...]:
    train_metrics = {"loss": [], "acc": [], "step": []}
    test_metrics = {"loss": [], "acc": [], "step": []}

    global_step = 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )

    test_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1} / {epochs}")

        # training step
        model.train()  # enable training mode
        # this method sets `model.training = True`
        pbar = tqdm(train_loader)
        for inputs, targets in pbar:
            optimizer.zero_grad()  # zeroing any existing gradients
            inputs = inputs.view((-1, 28 * 28))  # reshaping to vector

            loss, y_predictions = loss_function(
                model, inputs, targets, return_predictions=True
            )
            loss.backward()  # backpropagation
            optimizer.step()  # applying gradients (partial derivatives)

            y_predictions = y_predictions.mean(dim=-1)
            accuracy = (
                (y_predictions.argmax(dim=1) == targets.argmax(dim=1))
                .float()
                .mean()
            )

            train_metrics["loss"].append(loss.item())
            train_metrics["acc"].append(accuracy.item())
            train_metrics["step"].append(global_step)
            global_step += 1
            pbar.update(1)
        pbar.close()

        # validating step
        model.eval()  # enable training mode
        # this method sets `model.training = False`

        preds = []
        trues = []
        total_batches = 0
        total_loss = 0.0
        for inputs, targets in test_loader:
            inputs = inputs.view((-1, 28 * 28))  # reshaping to vector
            loss, y_predictions = loss_function(
                model, inputs, targets, return_predictions=True
            )
            y_predictions = y_predictions.mean(dim=-1)
            total_batches += 1

            total_loss += loss.item()

            trues.append(targets)
            preds.append(y_predictions)

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        val_acc = (
            (preds.argmax(dim=1) == trues.argmax(dim=1)).float().mean().item()
        )

        test_metrics["loss"].append(total_loss / total_batches)
        test_metrics["acc"].append(val_acc)
        test_metrics["step"].append(global_step)

    return train_metrics, test_metrics


class IntegerToOneHotConverter:
    def __init__(self, num_classes: int):
        self._code = torch.eye(num_classes)

    def __call__(self, class_num: int):
        return self._code[class_num]


def load_mnist_datasets(
    limit_train_samples_to: int = 10_000
) -> Tuple[Dataset, Dataset]:
    train_dataset = MNIST(
        "data",  # folder where data should be saved
        download=True,
        transform=ToTensor(),  # transform to convert images to torch tensors
        target_transform=IntegerToOneHotConverter(10),
    )
    test_dataset = MNIST(
        "data",  # folder where data should be saved
        download=True,
        train=False,
        transform=ToTensor(),  # transform to convert to torch tensors
        target_transform=IntegerToOneHotConverter(10),
    )

    # limiting for faster training
    indices = np.random.permutation(len(train_dataset.data))[
        :limit_train_samples_to
    ]
    train_dataset.data = train_dataset.data[indices]
    train_dataset.targets = train_dataset.targets[indices]
    return train_dataset, test_dataset


def show_learning_curve(
    train_metrics: Dict[str, List[float]], test_metrics: Dict[str, List[float]]
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(train_metrics["step"], train_metrics["loss"], label="train")
    ax.plot(test_metrics["step"], test_metrics["loss"], label="test")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curve")
    plt.legend()


def show_accuracy_curve(
    train_metrics: Dict[str, List[float]], test_metrics: Dict[str, List[float]]
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plt.plot(train_metrics["step"], train_metrics["acc"], label="train")
    ax.plot(test_metrics["step"], test_metrics["acc"], label="test")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy curve")
    plt.legend()


def show_log_prob_curve(
    train_metrics: Dict[str, List[float]], test_metrics: Dict[str, List[float]]
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plt.plot(train_metrics["step"], train_metrics["log prob"], label="train")
    ax.plot(test_metrics["step"], test_metrics["log prob"], label="test")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Log prob")
    ax.set_title("Log prob curve")
    plt.legend()
