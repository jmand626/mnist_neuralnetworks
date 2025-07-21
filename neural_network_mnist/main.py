# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        # This model has W0, b0, W1, and b0, so we need to sample from the uniform dist for all 4 of these pieces
        # Create through the use of nn.parameter
        alpha_l1 = 1 /math.sqrt(d)
        # In both layers of the neural network, we plug in the input dimension to calculate the alpha, but the value of
        # the input is NOT the same
        dim_f1 = Uniform(-alpha_l1, alpha_l1)
        self.W0 = Parameter(dim_f1.sample((d, h)))
        # Remember the hidden layer! We go from d to h to k, not from d to k, so h is the output dim for the first layer!

        self.b0 = Parameter(dim_f1.sample((h, )))
        # Remember that b is a vector!

        # For the next layer
        alpha_l2 = 1 / math.sqrt(h)
        dim_f2 = Uniform(-alpha_l2, alpha_l2)
        self.W1 = Parameter(dim_f2.sample((h, k)))
        self.b1 = Parameter(dim_f2.sample((k, )))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        # Thankfully this is just so easy, especially compared to some of the garbage from A3 and A4!
        return relu(x@self.W0 + self.b0) @ self.W1 + self.b1


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        # Since its the same kind of code like the forward for f1, there will be less comments here. The main difference
        # is more hidden layers!!
        alpha_l1 = 1 / math.sqrt(d)
        dim_f1 = Uniform(-alpha_l1, alpha_l1)
        self.W0 = Parameter(dim_f1.sample((d, h0)))
        # Remember the hidden layer! We go from d to h0 to h1 to k, not from d to k, so h0 is the output dim for the first layer!
        self.b0 = Parameter(dim_f1.sample((h0,)))
        # Remember that b is a vector!

        # For the next layer
        alpha_l2 = 1 / math.sqrt(h0)
        dim_f2 = Uniform(-alpha_l2, alpha_l2)
        self.W1 = Parameter(dim_f2.sample((h0, h1)))
        self.b1 = Parameter(dim_f2.sample((h1,)))

        # and the final one
        alpha_l3 = 1 / math.sqrt(h1)
        dim_f3 = Uniform(-alpha_l3, alpha_l3)
        self.W2 = Parameter(dim_f3.sample((h1, k)))
        self.b2 = Parameter(dim_f3.sample((k,)))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        # Since its the same kind of code like the forward for f1, there will be less comments here.
        return relu(relu(x @ self.W0 + self.b0) @ self.W1 + self.b1) @ self.W2 + self.b2


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    # This is similar to so many of the previous train pieces! If you are looking for me to do comments, look there.
    epoch = 0
    losses = []

    # Train until .99 threshold
    accuracies = 0

    while accuracies < .99:
        loss_train = 0
        total = 0

        true = 0
        model.train()

        for x, y in train_loader:

            optimizer.zero_grad()
            output_vals = model(x)

            # Pretty standard run from all of these train functions!
            loss = cross_entropy(output_vals, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

            pred_class, class_index = torch.max(output_vals, dim=1)
            true += (class_index == y).sum().item()

            total += y.size(0)
            # Total is all indices, true is just the ones that match

        losses.append(loss_train / len(train_loader))
        # Get loss and accuracy for epoch
        accuracies = true / total
        # OMFG I WROTE ACCURACY INSTEAD OF ACCURACIES HOLY SHIT I AM GOING TO DEFENSTRATE SOMEONE

        epoch+=1
        # Why the hell does python not have ++....

    return losses


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()


    # Start training!!
    train_data = TensorDataset(x, y)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    input_dim = 784
    output_dim = 10
    hidden_dim_f1 = 64
    hidden_dim_h0_f2 = 32
    hidden_dim_h1_f2 = 32

    # Train F1!
    model_f1 = F1(hidden_dim_f1, input_dim, output_dim)
    # Now we finally get to use the adam optimizer!
    optimizer_f1 = Adam(model_f1.parameters(), lr=0.001)
    loss_f1 = train(model_f1, optimizer_f1, train_loader)

    # Plot per epoch losses for F1
    plt.figure(figsize=(12,8))
    plt.plot(loss_f1)
    plt.title('F1 training Loss vs epoch')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.show()

    # Report accuracy and loss on test set for F1
    # Switch back to eval from train mode
    model_f1.eval()

    with torch.no_grad():
        f1_labels = model_f1(x_test)
        loss_test = cross_entropy(f1_labels, y_test).item()

        predict, predict_index = torch.max(f1_labels.data,dim=1)
        avgf1_accuracy = ((predict_index == y_test).sum().item())/y_test.size(0)

    print(f"Loss - Model F1: {loss_test:.3f}")
    print(f"Accuracy - Model F1: {avgf1_accuracy:.3f}")

    # Report total number of parameters for F1
    # Mentioned online in docs and multiple times in an OH - numel gives the num of elements in a tensor, and it turns
    # out we have to use it here
    print(f"F1 Number of Parameters: {sum(k.numel() for k in model_f1.parameters() if k.requires_grad)}")


    # Now repeat the same thing with F2 residentSleeper
    # Train F2!
    model_f2 = F2(hidden_dim_h0_f2, hidden_dim_h1_f2, input_dim, output_dim)
    # Now we finally get to use the adam optimizer!
    optimizer_f2 = Adam(model_f2.parameters(), lr=0.001)
    loss_f2 = train(model_f2, optimizer_f2, train_loader)

    # Plot per epoch losses for F2
    plt.figure(figsize=(12, 8))
    plt.plot(loss_f2)
    plt.title('F2 training Loss vs epoch')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.show()

    # Report accuracy and loss on test set for F2
    # Switch back to eval from train mode
    model_f2.eval()

    with torch.no_grad():
        f2_labels = model_f2(x_test)
        loss_test = cross_entropy(f2_labels,y_test).item()

        predict, predict_index = torch.max(f2_labels.data, dim=1)
        avgf2_accuracy = ((predict_index == y_test).sum().item()) /y_test.size(0)

    print(f"Loss - Model F2: {loss_test:.3f}")
    print(f"Accuracy - Model F2: {avgf2_accuracy:.3f}")

    # Report total number of parameters for F2
    # Mentioned online in docs and multiple times in an OH - numel gives the num of elements in a tensor, and it turns
    # out we have to use it here
    print(f"F2 Number of Parameters: {sum(k.numel() for k in model_f2.parameters() if k.requires_grad)}")


if __name__ == "__main__":
    main()
