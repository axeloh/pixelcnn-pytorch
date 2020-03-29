import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from masked_cnn_layer import MaskedConv2d, AutoregressiveMaskedConv2d, AutoregressiveResidualMaskedConv2d


class AutoregressiveColorPixelRCNN(nn.Module):
    """ Pixel Residual-CNN-class using residual blocks as shown in figure 5 from "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al. """
    def __init__(self, in_channels, out_channels, conv_filters):
        super().__init__()
        self.net = nn.Sequential(
            # A 7x7 A-type convolution
            MaskedConv2d('A', in_channels, conv_filters, kernel_size=7, padding=3), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            # 8 type-B residual convolutons
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            AutoregressiveMaskedConv2d('B', conv_filters, out_channels, kernel_size=1, padding=0)
        ).cuda()

    def forward(self, x):
        return self.net(x)


def main(train_data, test_data, image_shape):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, W) of samples with values in {0, 1, 2, 3}
    """
    H, W, C = image_shape
    output_bits = 4

    def normalize(x):
        """ Values in [0, 3] normalizing to [-1, 1] """
        return (x - 1.5) / 1.5

    def get_proba(output):
        return torch.nn.functional.softmax(output.reshape(output.shape[0], output_bits, C, H, W), dim=1)

    def cross_entropy_loss(batch, output):
        per_bit_output = output.reshape(batch.shape[0], output_bits, C, H, W)
        return torch.nn.CrossEntropyLoss()(per_bit_output, batch.long())

    def get_test_loss(dataset, model):
        test_loss = []
        for batch in torch.split(dataset, 128):
            out = model(normalize(batch))
            loss = cross_entropy_loss(batch, out)
            test_loss.append(loss.item())

        return np.mean(np.array(test_loss))

    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])).float().cuda()
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float().cuda()

    epochs = 10
    lr = 1e-3
    no_channels, out_channels, convolution_filters = C, C * output_bits, 120

    pixelrcnn_auto = AutoregressiveColorPixelRCNN(no_channels, out_channels, convolution_filters).cuda()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(pixelrcnn_auto.parameters(), lr=lr)

    train_losses = []
    test_losses = [get_test_loss(test_data, pixelrcnn_auto)]

    # Training
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = pixelrcnn_auto(normalize(batch))
            loss = cross_entropy_loss(batch, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = get_test_loss(test_data, pixelrcnn_auto)
        test_losses.append(test_loss)
        print(f'{epoch + 1}/{epochs} epochs')

    torch.cuda.empty_cache()
    pixelrcnn_auto.eval()

    # Sampling
    samples = torch.zeros(size=(100, C, H, W)).cuda()
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    out = pixelrcnn_auto(normalize(samples))
                    proba = get_proba(out)
                    samples[:, c, i, j] = torch.multinomial(proba[:, :, c, i, j], 1).squeeze().float()

    return np.array(train_losses), np.array(test_losses), np.transpose(samples.detach().cpu().numpy(), [0, 2, 3, 1])