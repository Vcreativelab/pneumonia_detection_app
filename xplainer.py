# Imports
import torch
import torchvision
import pytorch_lightning as pl
import torchvision.transforms.functional as fn
import matplotlib.pyplot as plt
import streamlit as st

# In this notebook we'll see how to use Class Activation Map (CAM)
# First, we recap the resnet18 architecture
temp_model = torchvision.models.resnet18()
# We can convert the network to a generator using the children() function
# list(temp_model.children())[:-2]  # get all layers up to avgpool (except the last two)
# Using Sequential from pytorch, we convert the list of layers back to a Sequential Model
# we need the star operator (*) to unpack the layers into positional arguments
torch.nn.Sequential(*list(temp_model.children())[:-2])


# Now we are ready to go.
# We add another output to the forward function of our pneumonia model,
# to return the feature maps of the last convolutional layer
# We extract the feature map in the forward pass, followed by global average pooling and flattening.
# Finally, we use the fully connected layer to compute the final class prediction.

class Xplainer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18()
        # Change conv1 from 3 to 1 input channels
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Change out_feature of the last fully connected layer (called fc in resnet18) from 1000 to 1
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)

        # Extract the feature map
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, data):
        # Compute feature map
        feature_map = self.feature_map(data)
        # Use Adaptive Average Pooling as in the original model
        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(input=feature_map, output_size=(1, 1))
        print(f"Avg pool output shape: {avg_pool_output.shape}")
        # Flatten the output into a 512 element vector
        avg_pool_output_flattened = torch.flatten(avg_pool_output)
        print(f"Avg pool flattened output shape: {avg_pool_output_flattened.shape}")
        # Compute prediction
        pred = self.model.fc(avg_pool_output_flattened)
        return pred, feature_map


def cam(model, img):
    """
    Compute class activation map according to cam algorithm
    """
    with torch.no_grad():
        pred, features = model(img)
    b, c, h, w = features.shape

    # We reshape the 512x7x7 feature tensor into a 512x49 tensor in order to simplify the multiplication
    features = features.reshape((c, h * w))

    # Get only the weights, not the bias
    weight_params = list(model.model.fc.parameters())[0]

    # Remove gradient information from weight parameters to enable numpy conversion
    weight = weight_params[0].detach()
    print(f"Last fc layer weight shape: {weight.shape}")
    # Compute multiplication between weight and features with the formula from above.
    # We use matmul because it directly multiplies each filter with the weights
    # and then computes the sum. This yields a vector of 49 (7x7 elements)
    cam = torch.matmul(weight, features)
    print(f"Features shape:  {features.shape}")

    # The following loop performs the same operations in a less optimized way
    # cam = torch.zeros((7 * 7))
    # for i in range(len(cam)):
    #    cam[i] = torch.sum(weight*features[:,i])
    ##################################################################

    # Normalize and standardize the class activation map (Not always necessary, thus not shown in the lecture)
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)
    # Reshape the class activation map from 512x7x7 to 7x7 and move the tensor back to CPU
    cam_img = cam_img.reshape(h, w).cpu()

    return cam_img, torch.sigmoid(pred)

