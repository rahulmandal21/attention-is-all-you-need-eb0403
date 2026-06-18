import torch
import torch.nn as nn

class CrossEntropyLossFunction(nn.Module):
    """
    A PyTorch module that calculates the cross-entropy loss between predictions and targets.
    """

    def __init__(self, reduction: str = 'mean') -> None:
        """
        Initializes the CrossEntropyLossFunction module.

        Args:
        reduction (str): The reduction method to use. Defaults to 'mean'.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cross-entropy loss between predictions and targets.

        Args:
        predictions (torch.Tensor): The predicted values.
        targets (torch.Tensor): The target values.

        Returns:
        torch.Tensor: The calculated cross-entropy loss.
        """
        return self.criterion(predictions, targets)


class ModelOptimizer:
    """
    A class that optimizes a PyTorch model using a given optimizer and loss function.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_function: nn.Module) -> None:
        """
        Initializes the ModelOptimizer class.

        Args:
        model (nn.Module): The PyTorch model to optimize.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        loss_function (nn.Module): The loss function to use.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def optimize(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Optimizes the model using the given inputs and targets.

        Args:
        inputs (torch.Tensor): The input values.
        targets (torch.Tensor): The target values.

        Returns:
        torch.Tensor: The calculated loss.
        """
        self.optimizer.zero_grad()
        predictions = self.model(inputs)
        loss = self.loss_function(predictions, targets)
        loss.backward()
        self.optimizer.step()
        return loss


if __name__ == "__main__":
    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(5, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    # Create a dummy model instance
    model = DummyModel()

    # Create a loss function
    loss_function = CrossEntropyLossFunction()

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create a model optimizer
    model_optimizer = ModelOptimizer(model, optimizer, loss_function)

    # Create dummy inputs and targets
    inputs = torch.randn(10, 5)
    targets = torch.randint(0, 3, (10,))

    # Optimize the model
    loss = model_optimizer.optimize(inputs, targets)
    print(loss.item())