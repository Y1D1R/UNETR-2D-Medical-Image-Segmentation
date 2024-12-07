from logging import config
import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import transforms # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore

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
    
class GreyBlock(nn.Module):
    #Classification Block
    def __init__(self, in_channels, out_channels):
        super(GreyBlock, self).__init__()    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
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
        
            
        # Convolutional Decoder
        z3, z6, z9, z12 = feature_map
        hidden_shape = (batch_size, self.config["hidden_dim"], self.config["patch_width"], self.config["patch_height"])
        print("CNN Decoder Feature Maps Shape Before Reshaping : ", z3.shape, z6.shape, z9.shape, z12.shape)
        z3 = z3.view(hidden_shape)
        z6 = z6.view(hidden_shape)
        z9 = z9.view(hidden_shape)
        z12 = z12.view(hidden_shape)
        print("CNN Decoder Feature Maps Shape After Reshaping : ", z3.shape, z6.shape, z9.shape, z12.shape)
        
        
        # z9 and z12 Decoder
        z9_d1 = BlueBlock(self.config["hidden_dim"], 512)(z9)
        z12_d1 = GreenBlock(self.config["hidden_dim"], 512)(z12)
        z9z12_d1 = torch.cat([z9_d1, z12_d1], dim = 1)
        z9z12_d2 = OrangeBlock(512 + 512, 512)(z9z12_d1)
        z9z12_d2 = OrangeBlock(512, 512)(z9z12_d2)
        print("Z9-Z12 DECODER OUTPUT SHAPE : ", z9z12_d2.shape)
        
        # z6 and z9-z12 Decoder
        z9z12_d3 = GreenBlock(512, 256)(z9z12_d2)
        z6_d1 = BlueBlock(self.config["hidden_dim"], 256)(z6)
        z6_d1 = BlueBlock(256, 256)(z6_d1)
        z6z9z12_d3 = torch.cat([z6_d1, z9z12_d3], dim = 1)
        z6z9z12_d4 = OrangeBlock(256 + 256, 256)(z6z9z12_d3)
        z6z9z12_d4 = OrangeBlock(256, 256)(z6z9z12_d4)
        print("Z6-Z9-Z12 DECODER OUTPUT SHAPE : ", z6z9z12_d4.shape)
        
        # z3 and z6-z9-z12 Decoder
        z6z9z12_d5 = GreenBlock(256, 128)(z6z9z12_d4)
        z3_d1 = BlueBlock(self.config["hidden_dim"], 128)(z3)
        z3_d1 = BlueBlock(128, 128)(z3_d1)
        z3_d1 = BlueBlock(128, 128)(z3_d1)
        z3z6z9z12_d5 = torch.cat([z3_d1, z6z9z12_d5], dim = 1)
        z3z6z9z12_d6 = OrangeBlock(128 + 128, 128)(z3z6z9z12_d5)
        z3z6z9z12_d6 = OrangeBlock(128, 128)(z3z6z9z12_d6)
        print("Z3-Z6-Z9-Z12 DECODER OUTPUT SHAPE : ", z3z6z9z12_d6.shape)
        
        # z0 and z3-z6-z9-z12 Decoder
        z0 = inputs.view(batch_size, self.config["num_channels"], self.config["image_width"], self.config["image_height"])
        print("Z0 after Reshaping : ", z0.shape)
        z3z6z9z12_d7 = GreenBlock(128, 64)(z3z6z9z12_d6)
        z0_d1 = OrangeBlock(z0.shape[1], 64)(z0)
        z0_d1 = OrangeBlock(64, 64)(z0_d1)
        z0z3z6z9z12_d7 = torch.cat([z0_d1, z3z6z9z12_d7], dim = 1)
        z0z3z6z9z12_d8 = OrangeBlock(64+64, 64)(z0z3z6z9z12_d7)
        z0z3z6z9z12_d8 = OrangeBlock(64, 64)(z0z3z6z9z12_d8)
        print("Z0-Z3-Z6-Z9-Z12 DECODER OUTPUT SHAPE : ", z0z3z6z9z12_d8.shape)
        
        
        # Output (the mask)
        output = GreyBlock(64, 1)(z0z3z6z9z12_d8)
        return output
        
    
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
    

    
    # Load and preprocess an image
    image_path = "MRI4gt.png"  
    preprocess = transforms.Compose([ transforms.Resize((configuration["image_height"], configuration["image_width"])),transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)  
    print("Image Tensor Shape:", image_tensor.shape)
    
    # Convert image to patches
    patches = image_tensor.unfold(2, configuration["patch_height"], configuration["patch_height"]).unfold(3, configuration["patch_width"], configuration["patch_width"])
    patches = patches.contiguous().view(1, -1, configuration["patch_height"] * configuration["patch_width"] * configuration["num_channels"])
    print("Patch Tensor Shape:", patches.shape)
    
    # Model
    model = UNETR(configuration)
     
    
    output = model(patches)
    print("Model Output Shape:", output.shape)
        
       
    output_image = output.squeeze(0).squeeze(0).detach().numpy()  
    print("Output Image Shape:", output_image.shape)
        
    # Display input image and result
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(output_image, cmap="gray")
    plt.title("Output Mask")
    plt.axis("off")

    plt.show()