
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from masked_cnn_layer import MaskedConv2d, ResidualMaskedConv2d


class PixelRCNN(nn.Module):
    """PixelCNN that supports RGB color channels
    Assumes color channels as independent.
    """
    def __init__(self, in_channels):
        super().__init__()
        num_filters = 120
        self.net = nn.Sequential(
            MaskedConv2d('A', in_channels, num_filters, 7 , 1, 3), nn.ReLU(),
            ResidualMaskedConv2d(num_filters), nn.ReLU(), nn.BatchNorm2d(num_filters),
            ResidualMaskedConv2d(num_filters), nn.ReLU(), nn.BatchNorm2d(num_filters),
            ResidualMaskedConv2d(num_filters), nn.ReLU(), nn.BatchNorm2d(num_filters),
            ResidualMaskedConv2d(num_filters), nn.ReLU(), nn.BatchNorm2d(num_filters),
            ResidualMaskedConv2d(num_filters), nn.ReLU(), nn.BatchNorm2d(num_filters),
            ResidualMaskedConv2d(num_filters), nn.ReLU(), nn.BatchNorm2d(num_filters),
            ResidualMaskedConv2d(num_filters), nn.ReLU(), nn.BatchNorm2d(num_filters),
            ResidualMaskedConv2d(num_filters), nn.ReLU(), nn.BatchNorm2d(num_filters),
            MaskedConv2d('B', num_filters, 12, 1)
        )

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
    out_channels = 4

    def normalize(data):
        """Normalizing from [0-3] to [-1, 1]"""
        return (data - 1.5) / 1.5

    def model_proba(input):
        logits = input.view(input.shape[0], out_channels, C, H, W)
        return F.softmax(logits, dim=1)

    def get_loss(output, batch):
        loss = nn.CrossEntropyLoss()

        output_reshaped = output.reshape(batch.shape[0], out_channels, C, H, W)
        return loss(output_reshaped, batch.long())

    def get_test_loss(data):
        tot_loss = []
        for batch in torch.split(data, 128):
            output = net(normalize(batch))
            tot_loss.append(get_loss(output, batch).item())
        return np.mean(np.array(tot_loss))

    use_cuda = True
    device = torch.device('cuda') if use_cuda else None

    net = PixelRCNN(C)
    if use_cuda:
        net.cuda()

    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])).float().to(device)
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float().to(device)
    epochs = 20
    batch_size = 128
    batches = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_losses = []
    init_test_loss = get_test_loss(test_data)
    test_losses = [init_test_loss]

    # Training
    for epoch in range(epochs):
        for batch in batches:
            optimizer.zero_grad()
            output = net(normalize(batch))
            loss = get_loss(output, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        test_losses.append(get_test_loss(test_data))
        print(f'{epoch + 1}/{epochs} epochs')

    net.eval()

    # Sampling
    samples = torch.zeros(size=(100, C, H, W)).to(device)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    probs = model_proba(net(normalize(samples)))
                    levels = torch.multinomial(probs[:, :, c, i, j], 1).squeeze().float()
                    samples[:, c, i, j] = levels

    return np.array(train_losses), np.array(test_losses), np.transpose(samples.detach().cpu().numpy(), [0, 2, 3, 1])