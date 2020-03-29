import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pixelcnn import PixelCNN

def train(train_data, test_data, image_shape):
    """
        train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
        test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
        image_shape: (H, W), height and width of the image

        Returns
        - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
        - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
        - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
        """

    def get_loss(output, batch):
        return F.binary_cross_entropy(output, batch)

    def get_test_loss(data):
        tot_loss = []
        for batch in torch.split(data, 128):
            output = net((batch - .5) / .5)
            tot_loss.append(get_loss(output, batch).item())
        return np.mean(np.array(tot_loss))

    # Model (forward(x)) expects shape to be (batch_size, channels, height, width)
    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])).float().cuda()
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float().cuda()

    net = PixelCNN()

    epochs = 10
    batch_size = 128
    batches = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_losses = []
    init_test_loss = get_test_loss(test_data)
    test_losses = [init_test_loss]

    # Training
    for epoch in range(epochs):
        for batch in batches:
            optimizer.zero_grad()
            output = net((batch - .5) / .5)
            loss = get_loss(output, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        test_losses.append(get_test_loss(test_data))
        print(f'{epoch + 1}/{epochs} epochs')


    net.eval()

    # Sampling
    H, W = image_shape
    samples = torch.zeros(size=(100, 1, H, W)).cuda()
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                out = net((samples - .5) / .5)
                samples[:, :, i, j] = torch.bernoulli(out[:, :, i, j], out=samples[:, :, i, j])

    return np.array(train_losses), np.array(test_losses), np.transpose(samples.detach().cpu().numpy(), [0, 2, 3, 1])