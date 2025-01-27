## Update: 8th Jan, 2022
## this file is supposed to give you a general idea on how to
## use the pre-trained model for colorizing B&W images. This
## file still needs development.

import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from models import MainModel
from utils import lab_to_rgb

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

# the following is a workaround from the Github issues of this repo 
# https://github.com/moein-shariatnia/Deep-Learning/issues/11#issuecomment-1879759328 
# to fix "AttributeError: 'function' object has no attribute 'named_parameters'" that is apparently due to the newer version of fastai
def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
    model = MainModel(net_G=net_G)
    #model = MainModel()
    # You first need to download the final_model_weights.pt file from my drive
    # using the command: wget https://drive.google.com/uc?id=1lR6DcS4m5InSbZ5y59zkH2mHt_4RQ2KV
    model.load_state_dict(
        torch.load(
            "final_model_weights.pt", ### TODO: the original weights file has differing dict entries and does not work!
            map_location=device
        )
    )
    path = "exp/mamas_ahnen/shrunk/Ururopa_Heinrich_und_Ururoma_Elisabeth_Pfahls.jpg"
    img = PIL.Image.open(path)
    #img = img.resize((256, 256))  ## this input is already shrunk
    # to make it between -1 and 1
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    plt.imshow(colorized)
