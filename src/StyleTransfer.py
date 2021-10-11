import os
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torchvision.models as models
from torchvision.utils import save_image
import torchvision.transforms as transforms

class StyleTransfer:
    def __init__(self,img_size:int=356,optimizer_func:optim=optim.Adam) -> None:
        self.img_size = 356
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_func = optimizer_func
        self.loader = transforms.Compose(
            [transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[],std=[])
            ]
        )
        
        
    def load_image(self,path):
        img = Image.open(path)
        img = self.loader(img).unsqueeze(0)
        return img.to(self.device)
    def Transfer(self,original_img_path:str, style_img_path:str,
                steps:int=600, learn_rate:float = 0.01,
                content_weight:float=0.56, style_weight:float=1000,
                save_ever_n_steps:int=50):

        original_img = self.load_image(original_img_path)
        style_img = self.load_image(style_img_path)
        # generated_img = torch.randn(original_img.shape,device=self.device, requires_grad=True)
        generated_img = original_img.clone().requires_grad_(True)

        model = VGG(self.device)
        optimizer  = self.optimizer_func([generated_img],lr=learn_rate)

        for step in range(steps):
            generated_img_features = model(generated_img)
            original_img_features = model(original_img)
            style_img_features = model(style_img)

            style_loss = original_loss = 0

            for gen_feature, orig_feature, style_feature in zip(
                generated_img_features, original_img_features, style_img_features):

                bash_size, channel, h,w =  gen_feature.shape

                original_loss += torch.mean((gen_feature-orig_feature)**2) 

                #compute gram matrix
                G = gen_feature.view(channel,h*w).mm(
                    gen_feature.view(channel,h*w).t())
                
                A = style_feature.view(channel,h*w).mm(
                    style_feature.view(channel,h*w).t())

                style_loss += torch.mean((G-A)**2) 

            total_loss = content_weight * original_loss + style_weight * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % save_ever_n_steps == 0:
                print(f"{step}: {total_loss.item()}")
                save_image(generated_img,f"{os.path.splitext(original_img_path)[0]}_{step}.png")
        save_image(generated_img,f"{os.path.splitext(original_img_path)[0]}Last.png")

class VGG(nn.Module):
    def __init__(self,device):
        super(VGG, self).__init__()
        self.features_layers = [0,5,10,19,28]
        self.model = models.vgg19(pretrained=True).features[:29].to(device).eval()
    def forward(self, x):
        features = []
        for num, layer in enumerate(self.model):
            x=layer(x)

            if num in self.features_layers :
                features.append(x)
        return features