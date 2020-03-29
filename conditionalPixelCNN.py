import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from masked_cnn_layer import MaskedConv2d, ResidualMaskedConv2d, ConditionalMaskedConv2d


class ConditionalPixelCNN(nn.Module):
    """ Class condtitional PixelCNN-class """

    def __init__(self, in_channels, conv_filters, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            ConditionalMaskedConv2d('A', num_classes, in_channels=in_channels, out_channels=conv_filters, kernel_size=7,
                                    padding=3), nn.ReLU(),

            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3), nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3), nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3), nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3), nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=7, padding=3), nn.ReLU(),

            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=conv_filters,
                                    kernel_size=1), nn.ReLU(),
            ConditionalMaskedConv2d('B', num_classes, in_channels=conv_filters, out_channels=in_channels,
                                    kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, class_condition):
        out = x
        for layer in self.net:
            out = layer(out, class_condition) if isinstance(layer, ConditionalMaskedConv2d) else layer(out)
        return out


def main(train_data, train_labels, test_data, test_labels, image_shape, n_classes, dset_id):
    """
    train_data: A (n_train, H, W, 1) numpy array of binary images with values in {0, 1}
    train_labels: A (n_train,) numpy array of class labels
    test_data: A (n_test, H, W, 1) numpy array of binary images with values in {0, 1}
    test_labels: A (n_test,) numpy array of class labels
    image_shape: (H, W), height and width
    n_classes: number of classes (4 or 10)
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, 1) of samples with values in {0, 1}
    where an even number of images of each class are sampled with 100 total
    """

    def one_hot(labels):
        labels_oh = np.zeros((labels.size, n_classes))
        labels_oh[np.arange(labels.size), labels] = 1
        return torch.tensor(labels_oh).float().cuda()

    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)

    def normalize(x):
        return (x - 0.5) / 0.5

    def cross_entropy_loss(batch, output):
        return F.binary_cross_entropy(output, batch)

    def get_batched_loss(data, labels, model):
        test_loss = []
        for batch, label in zip(torch.split(data, 128), torch.split(labels, 128)):
            out = model(normalize(batch), label)
            loss = cross_entropy_loss(batch, out)
            test_loss.append(loss.item())
        return np.mean(np.array(test_loss))

    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])).float().cuda()
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float().cuda()

    epochs = 25 if dset_id == 1 else 10
    lr = 1e-3
    no_channels, convolution_filters = 1, 64

    cpixelcnn = ConditionalPixelCNN(no_channels, convolution_filters, num_classes=n_classes).cuda()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False)
    train_label_loader = torch.utils.data.DataLoader(train_labels, batch_size=128, shuffle=False)
    optimizer = torch.optim.Adam(cpixelcnn.parameters(), lr=lr)

    train_losses = []
    test_losses = [get_batched_loss(test_data, test_labels, cpixelcnn)]

    # Training
    for epoch in range(epochs):
        for batch_x, batch_y in zip(train_loader, train_label_loader):
            optimizer.zero_grad()
            output = cpixelcnn(normalize(batch_x), batch_y)
            loss = cross_entropy_loss(batch_x, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = get_batched_loss(test_data, test_labels, cpixelcnn)
        test_losses.append(test_loss)
        print(f'{epoch + 1}/{epochs} epochs')

    torch.cuda.empty_cache()
    cpixelcnn.eval()

    # Sampling
    H, W = image_shape
    samples = torch.zeros(size=(100, 1, H, W)).cuda()
    sample_classes = one_hot(np.sort(np.array([np.arange(n_classes)] * (100 // n_classes)).flatten()))
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                out = cpixelcnn(normalize(samples), sample_classes)
                torch.bernoulli(out[:, :, i, j], out=samples[:, :, i, j])

    return np.array(train_losses), np.array(test_losses), np.transpose(samples.detach().cpu().numpy(), [0, 2, 3, 1])