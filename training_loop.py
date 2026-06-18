import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

class WarmupSchedule(nn.Module):
    """Learning rate schedule with a warmup period."""
    def __init__(self, d_model: int, warmup_steps: int):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def forward(self) -> float:
        """Returns the current learning rate."""
        self.step_num += 1
        return self.d_model ** -0.5 * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)


class TrainingLoop:
    """Class responsible for training a model using a variant of stochastic gradient descent."""
    def __init__(self, model: nn.Module, dataloader: DataLoader, d_model: int, warmup_steps: int, max_grad_norm: float = 1.0):
        """
        Initializes the training loop.

        Args:
        - model (nn.Module): The model to be trained.
        - dataloader (DataLoader): The data loader for the training data.
        - d_model (int): The dimensionality of the model.
        - warmup_steps (int): The number of warmup steps.
        - max_grad_norm (float): The maximum gradient norm. Defaults to 1.0.
        """
        self.model = model
        self.dataloader = dataloader
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = WarmupSchedule(self.d_model, self.warmup_steps)

    def train_one_epoch(self) -> float:
        """
        Trains the model for one epoch.

        Returns:
        - float: The average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in self.dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloader)

    def update_learning_rate(self) -> None:
        """Updates the learning rate using the warmup schedule."""
        self.optimizer.param_groups[0]['lr'] = self.scheduler()


if __name__ == "__main__":
    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    # Create a dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size: int):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.randn(10), torch.randint(0, 10, (1,)).squeeze()

    # Create a dummy data loader
    dataloader = DataLoader(DummyDataset(100), batch_size=10, shuffle=True)

    # Create a training loop
    model = DummyModel()
    training_loop = TrainingLoop(model, dataloader, d_model=10, warmup_steps=100)

    # Train the model for one epoch
    loss = training_loop.train_one_epoch()
    print(f"Loss: {loss}")

    # Update the learning rate
    training_loop.update_learning_rate()