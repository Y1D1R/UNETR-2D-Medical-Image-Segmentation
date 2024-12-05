from logging import config
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np

class OrangeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(OrangeBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GreenBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,padding=0,stride=2)

    def forward(self, x):
        return self.deconv(x)

class BlueBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlueBlock, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,padding=0,stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        #print("X input blue block : ", x.shape)
        x = self.deconv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        #print("X output blue block : ", x.shape)
        return x
        
class UNETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Linear Projection
        self.patch_embedding = nn.Linear(config["patch_height"] * config["patch_width"] * config["num_channels"] , config["hidden_dim"])
        
        # Positional Embedding
        self.positions = torch.arange(start = 0, end = config["num_patches"], step = 1)
        self.positions_embeddings = nn.Embedding(config["num_patches"], config["hidden_dim"])
        
        # Transformer Encoder
        self.encoder_layers = []
        for i in range(config["num_layers"]):
            layer = nn.TransformerEncoderLayer(d_model=config["hidden_dim"],nhead=config["num_layers"], dim_feedforward = config["mlp_dim"], dropout = config["dropout_rate"], activation = nn.GELU(), batch_first = True)
            self.encoder_layers.append(layer)
            
    
    
    def forward(self, inputs):
        
        batch_size = inputs.shape[0]
        
        # Linear Projection and Positional Embedding
        patch_embedding = self.patch_embedding(inputs)
        print("inputs shape : ", inputs.shape)
        print("Patch Embedding Shape : ", patch_embedding.shape)
        positions = self.positions
        positions_embeddings = self.positions_embeddings(positions)
        print("Positions Embeddings Shape : ", positions_embeddings.shape)
        x = patch_embedding + positions_embeddings
        print("Concatenated Embeddings Shape : ", x.shape)
        
        # Transformer Encoder
        connection_map = [3, 6, 9, 12]
        feature_map = []
        j=1
        for layer in self.encoder_layers:
            x = layer(x)
            if j in connection_map:
                feature_map.append(x)
            j=j+1
            
        # CNN Decoder
        z0 = inputs.view(batch_size, self.config["num_channels"], self.config["image_width"], self.config["image_height"])
        print("Z0 after Reshaping : ", z0.shape)
        
        z3, z6, z9, z12 = feature_map
        hidden_shape = (batch_size, self.config["hidden_dim"], self.config["patch_width"], self.config["patch_height"])
        print("CNN Decoder Feature Maps Shape Before Reshaping : ", z3.shape, z6.shape, z9.shape, z12.shape)
        z3 = z3.view(hidden_shape)
        z6 = z6.view(hidden_shape)
        z9 = z9.view(hidden_shape)
        z12 = z12.view(hidden_shape)
        print("CNN Decoder Feature Maps Shape After Reshaping : ", z3.shape, z6.shape, z9.shape, z12.shape)
        
        
        # z9 and z12 Decoder
        x = BlueBlock(self.config["hidden_dim"], 512)(z9)
        x1 = GreenBlock(self.config["hidden_dim"], 512)(z12)
        x = torch.cat([x, x1], dim = 1)
        x = OrangeBlock(512 + 512, 512)(x)
        x = OrangeBlock(512, 512)(x)
        print("X Shape : ", x.shape)
    
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
    print("Input Tensor Shape : ", x.shape)
    
    model = UNETR(configuration)
    model(x)
    