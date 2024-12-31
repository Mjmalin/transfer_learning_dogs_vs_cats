import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# Define the path to your new dataset
data_dir = '/Users/maxwellmalinofsky/Desktop/portfolio/dataset'

# Use ImageFolder for the new dataset (assuming structure like: train/cats, train/dogs, etc.)
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# Display some sample images from the dataset
dogs = [train_imgs[i][0] for i in range(8)]
cats = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(dogs + cats, 2, 8, scale=1.4)

# Specify normalization parameters for the ImageNet pretrained model
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Define image augmentations for training and testing
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# Load the pretrained ResNet-18 model
pretrained_net = torchvision.models.resnet18(pretrained=True)
pretrained_net.fc

# Modify the output layer for binary classification (2 output classes)
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)  # 2 classes (e.g., cats vs dogs)
nn.init.xavier_uniform_(finetune_net.fc.weight)

# Define the training function
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")  # Use CrossEntropyLoss for binary classification
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# Call the training function with a learning rate and number of epochs
train_fine_tuning(finetune_net, 5e-5)
