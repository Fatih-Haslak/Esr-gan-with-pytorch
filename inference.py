import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightning as L
import pytorch_lightning as pl
from model.generator import RRDBNet
import numpy as np
import cv2
import glob
import os.path as osp
checkpoint="/home/fatih/esrgan/best_model.ckpt" 
device= torch.device('cuda')
model = RRDBNet(3, 3, 64, 2, gc=32)
generator_params = model.state_dict()
checkpoint = torch.load(checkpoint)
for param_name in generator_params:
    if param_name in checkpoint["state_dict"]:
        model.load_state_dict({param_name: checkpoint["state_dict"][param_name]}, strict=False)

# disable randomness, dropout, etc...
model = model.to(device)
model.eval()

test_img_folder="/home/fatih/esrgan/test/*"

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    print(img_LR.dtype)
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
    print('results/{:s}_rlt.png'.format(base))
    print("kaydetti")