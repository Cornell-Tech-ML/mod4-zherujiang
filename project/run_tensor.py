"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

def RParam(*shape):
    """
    Create a random parameter tensor.

    Args:
        *shape: The shape of the parameter tensor.

    Returns:
        minitorch.Parameter: A randomly initialized parameter.
    """
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# Task 2.5.
class Network(minitorch.Module):
    """
    A neural network with three linear layers.
    """

    def __init__(self, hidden_layers):
        """
        Initialize the network.

        Args:
            hidden_layers (int): Number of neurons in the hidden layers.
        """
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output of the network.
        """
        h1 = self.layer1.forward(x).relu()
        h2 = self.layer2.forward(h1).relu()
        return self.layer3.forward(h2).sigmoid()

class Linear(minitorch.Module):
    """
    A linear (fully connected) layer.
    """

    def __init__(self, in_size, out_size):
        """
        Initialize the linear layer.

        Args:
            in_size (int): Number of input features.
            out_size (int): Number of output features.
        """
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        """
        Forward pass through the linear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output of the linear layer.
        """
        x = x.contiguous().view(*x.shape, 1)
        return (
            (x * self.weights.value)
            .sum(dim=1)
            .view(x.shape[0], self.out_size)
        ) + self.bias.value


def default_log_fn(epoch, total_loss, correct, losses):
    """
    Default logging function for training progress.

    Args:
        epoch (int): Current epoch number.
        total_loss (float): Total loss for the epoch.
        correct (int): Number of correct predictions.
        losses (list): List of losses for all epochs.
    """
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    """
    Class for training a neural network using tensors.
    """

    def __init__(self, hidden_layers):
        """
        Initialize the TensorTrain object.

        Args:
            hidden_layers (int): Number of neurons in the hidden layers.
        """
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        """
        Run a single input through the model.

        Args:
            x (list): Single input sample.

        Returns:
            Tensor: Model output for the input.
        """
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        """
        Run multiple inputs through the model.

        Args:
            X (list): List of input samples.

        Returns:
            Tensor: Model outputs for the inputs.
        """
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        """
        Train the model.

        Args:
            data: Dataset object containing training data.
            learning_rate (float): Learning rate for optimization.
            max_epochs (int): Maximum number of training epochs.
            log_fn (function): Logging function for training progress.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)

if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
