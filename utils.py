'''
    Utility functions for the Forward Forward Network implementation.
'''

import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from IPython.display import clear_output

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


class config:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_batch_size = 50000
    test_batch_size = 10000
    
    threshold = 0  # threshold for the FF layer
    num_epochs = 1000  # number of epochs for training the FF layer


# load mnizt dataset (train and test loaders)
def mnist_loaders(train_batch_size:int = config.train_batch_size, test_batch_size:int = config.test_batch_size) -> Tuple:

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

# make overlay for positive and negative samples (labels 0-9) in the first 10 pixels
def generate_sample_overlay(img : torch.Tensor , lbl : torch.Tensor , state : str ='positive') -> torch.Tensor:

    if(state == 'positive'):
        pos_img = img.clone()

        pos_img[:, :10] *= 0.0
        pos_img[range(pos_img.shape[0]), lbl] = pos_img.max()

        return pos_img

    elif(state == 'negative'):
        neg_img = img.clone()

        rnd = torch.randperm(neg_img.size(0))

        neg_img[:, :10] *= 0.0
        neg_img[range(neg_img.shape[0]), lbl[rnd]] = neg_img.max()

        return neg_img

# plot sample image 
def show_sample(x : torch.Tensor, pos : torch.Tensor,neg: torch.Tensor) -> None:
    for i in range(3):
        fig, axs = plt.subplots(1, 3)
        
        # Original images
        axs[0].imshow(x[i].reshape(28,28), cmap='binary')
        axs[0].set_title('Original')
        axs[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Positive images
        axs[1].imshow(pos[i].reshape(28,28), cmap='binary')
        axs[1].set_title('Positive')
        axs[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Negative images
        axs[2].imshow(neg[i].reshape(28,28), cmap='binary')
        axs[2].set_title('Negative')
        axs[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        plt.show()

# softplus function (smooth version of ReLU)
def softplus(x : torch.Tensor)->torch.Tensor:
    return torch.log(1 + torch.exp(x))

# forward forward layer (learning layer)
class FF_Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.05)
        self.threshold = config.threshold
        self.num_epochs = config.num_epochs

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, pos_sample, neg_sample):
        loss_values = []
        gen_pos_values = []
        gen_neg_values = []
        for i in tqdm(range(self.num_epochs)):

            pos_generated = self.forward(pos_sample).pow(2).mean(1)
            neg_generated = self.forward(neg_sample).pow(2).mean(1)

            pos_loss = -pos_generated + self.threshold
            neg_loss =  neg_generated - self.threshold

            # Softplus Function (smooth)
            loss = softplus(torch.cat([pos_loss,neg_loss])).mean()

            self.opt.zero_grad()

            loss.backward()
            self.opt.step()
          
            if i % 100 == 0:  
                loss_values.append(loss.item())
                gen_pos_values.append(pos_loss.mean().item())  # take mean of all batch values
                gen_neg_values.append(neg_loss.mean().item())  # take mean of all batch values

                # plotting
                plt.subplot(2,1,1)
                plt.plot(loss_values, color='blue')
                plt.title("Loss during training")

                plt.subplot(2,1,2)
                plt.plot(gen_pos_values, color='green',label='positive', linestyle='solid')
                plt.plot(gen_neg_values, color='red',label='negative', linestyle='dashed')
                plt.title("generated sample during training")
                plt.legend()

                plt.tight_layout()
                clear_output(wait=True)  # clear previous output and display new plots in smooth manner
                plt.show()

                print(f'Loss at step {i}: {loss.item()}')
            
        return self.forward(pos_sample).detach(), self.forward(neg_sample).detach()

# Forward Forward Network
class FF_Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [FF_Layer(dims[d], dims[d + 1])]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = generate_sample_overlay(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, positive_sample, negative_sample):
        h_pos, h_neg = positive_sample, negative_sample
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)
            
        print('--' * 30)
        print('training done.')
        


# :TODO: 
# 1. adding spatial variant and 
# 2. network are 4 hidden layers, each containing 2000 ReLU units.

# def shift_image(img, dx, dy):
#     padded = nn.functional.pad(img, (2, 2, 2, 2), mode='replicate')
#     x_start = 2 + dx
#     y_start = 2 + dy
#     return padded[:,y_start:y_start + 28, x_start:x_start + 28].squeeze(0)  # shape: [28, 28]



# class MNISTSpatialVariants:
#     """
#     MNIST dataset with spatial variants.
#     This class extends the torchvision MNIST dataset to include spatial variants.
#     """

#     def __init__(self,train=True):
#         transform = Compose(
#             [
#                 ToTensor(),
#                 Normalize((0.1307,), (0.3081,)),
#                 # Lambda(lambda x: torch.flatten(x))
#             ]
#         )
        
#         self.base_data = MNIST(
#             root='./data', 
#             train=train, 
#             download=True,
#             transform=transform,
#         )
#         self.shifts = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)] # 5x5 grid of shifts (include original image)

#     def __len__(self):
#         return len(self.base_data) * len(self.shifts)
    
#     def __getitem__(self, idx):
#         img_idx = idx // len(self.shifts)
#         shift_idx = idx % len(self.shifts)
#         dx, dy = self.shifts[shift_idx]
        
#         img, label = self.base_data[img_idx]

#         shifted_img = shift_image(img, dx, dy)
#         return shifted_img, label

# def generate_sample_overlay(img : torch.Tensor , lbl : torch.Tensor , state : str ='positive') -> torch.Tensor:
#     if(state == 'positive'):
#         pos_img = img.clone()
        
#         if pos_img.ndim > 2 :
#             for i in range(pos_img.shape[0]):
#                 pos_img[i][:1, :10] = 0.0  # set first 10 pixels of first row to zero
                
#                 label = lbl[i] if isinstance(lbl, torch.Tensor) and lbl.ndim > 0 else lbl
                
#                 pos_img[i][:1, label] = pos_img[i].max()  # set the pixel at the label
#         else: # case of generating positive sample and check the model
#             pos_img[:1, :10] = 0.0
#             pos_img[:1, lbl] = pos_img.max()

#         return pos_img

#     elif(state == 'negative'):
#         neg_img = img.clone()
        
#         for i in range(neg_img.shape[0]):
#             neg_img[i][:1, :10] = 0.0  # set first 10 pixels of first row to zero
#             rnd = torch.randint(0,10, (1,))
#             neg_img[i][:1, lbl[rnd]] = neg_img[i].max()  # set the pixel at the label index to the max value

#         return neg_img