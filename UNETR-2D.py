from logging import config
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np

class OrangeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(OrangeBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.BatchNorm2d(x)
        x = self.relu(x)
        return x

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GreenBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,padding=0,stride=2)

    def forward(self, x):
        return self.deconv(x)
    
class UNETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Linear Projection
        self.patch_embedding = nn.Linear(config["patch_height"] * config["patch_width"] * config["num_channels"] , config["hidden_dim"])
        
        # Positional Embedding
        self.positions = torch.arange(start = 0, end = config["num_patches"], step = 1)
        self.positions_embeddings = nn.Embedding(config["num_patches"], config["hidden_dim"])
        
    def forward(self, inputs):
        patch_embedding = self.patch_embedding(inputs)
        print(patch_embedding.shape)
        positions = self.positions
        positions_embeddings = self.positions_embeddings(positions)
        print(positions_embeddings.shape)
    
    
if __name__ == "__main__":
    
    configuration = {}
    # For exemple Image.shape = (256, 256, 3) and patch size = (16, 16, 3)
    image_shape = np.array([256, 256, 3])
    configuration["image_height"] = image_shape[0]
    configuration["image_width"] = image_shape[1]
    configuration["num_channels"] = image_shape[2]
    configuration["patch_height"] = 16
    configuration["patch_width"] = 16
    configuration["num_patches"] = (configuration["image_height"] * configuration["image_width"]) // (configuration["patch_height"] * configuration["patch_width"]) 
    configuration["num_layers"] = 12
    configuration["hidden_dim"] = 768
    configuration["mlp_dim"] = 3072
    configuration["dropout_rate"] = 0.1    
    
    
    # Input Tensor
    x = torch.randn((8, configuration["num_patches"], configuration["patch_height"] * configuration["patch_width"] * configuration["num_channels"]))
    print(x.shape)
    
    model = UNETR(configuration)
    model(x)
    