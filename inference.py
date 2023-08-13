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

def test(test_img_folder:str,model):
    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        
        # Görüntüyü oku ve boyutlandır
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)
        
        # Modeli kullanarak tahmin yap
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        # Çıktıyı yeniden boyutlandır ve kaydet
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
        print('results/{:s}_rlt.png'.format(base))
        print("Kaydedildi")

if __name__ == "__main__":
    #model checkpoint
    checkpoint = "/home/fatih/esrgan/last_model.ckpt"
    # Test görüntülerinin klasörü
    test_img_folder = "/home/fatih/esrgan/test/*"
    #gpu 
    device = torch.device('cuda')
    #load model
    model = RRDBNet(3, 3, 64, 23, gc=32)
    generator_params = model.state_dict()
    checkpoint = torch.load(checkpoint)
    for param_name in generator_params:
        if param_name in checkpoint["state_dict"]:
            model.load_state_dict({param_name: checkpoint["state_dict"][param_name]}, strict=False)

    # Diğer model ayarları
    model = model.to(device)
    model.eval()
    test(test_img_folder,model)


   
