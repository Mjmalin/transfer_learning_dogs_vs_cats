import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# Path to the dataset
data_dir = '/Users/maxwellmalinofsky/Desktop/portfolio/dataset'

# Split into training and test images
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# Display 8 images of cats and dogs each
dogs = [train_imgs[i][0] for i in range(8)]
cats = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(dogs + cats, 2, 8, scale=1.4)

# Specify the means and standard deviations of the three RGB channels to
# standardize each channelnormalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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

# Download ResNet-18 weights, trained on ImageNet dataset
pretrained_net = torchvision.models.resnet18(pretrained=True)
pretrained_net.fc

# Modify output layer for 2 features, cats and dogs
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greaterdef train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
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

# Call the function with a very small learning rate
train_fine_tuning(finetune_net, 5e-5)
