import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightning as L
import torch.utils.data as data


class DIV2KDataset(Dataset):
    def __init__(self, data_dir, label_data_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir= label_data_dir
        self.image_filenames = [filename for filename in os.listdir(data_dir) if filename.endswith('.png')]
        self.label_image_filenames = [filename for filename in os.listdir(label_data_dir) if filename.endswith('.png')]

        self.transform = transform
        # Veri dönüşümleri (isteğe bağlı)


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')

        label_img_path = os.path.join(self.label_dir, self.label_image_filenames[idx])
        label_image = Image.open(label_img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label_image)
          
        return (image,label)

class DIV2KDataLoader(L.LightningDataModule):
    def __init__(self, high_res_data_dir, low_res_data_dir, test_data_dir, batch_size, transform=None):
        self.batch_size = batch_size

        self.data_transform = transforms.Compose([
        transforms.Resize((128, 128)),   # Gerekirse boyutu değiştirin
        transforms.ToTensor(),           # Tensor formatına çevirin
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizasyon
        ])

        self.data = DIV2KDataset(high_res_data_dir,low_res_data_dir, transform=self.data_transform)
        #train test split for validation
        train_set_size = int(len(self.data) * 0.8) #train set size
        valid_set_size = len(self.data) - train_set_size #valid set size
        seed = torch.Generator().manual_seed(42) #seed
        self.train_set, self.valid_set = data.random_split(self.data, [train_set_size, valid_set_size], generator=seed)

        #self.valdata= DIV2KVal(test_data_dir, transform=self.data_transform)
         
        # self.low_res_dataset = DIV2KDataset(low_res_data_dir, transform=self.data_transform2)
        # self.test_dataset = DIV2KDataset(test_data_dir, transform=self.data_transform)
    

    def train_dataloader(self, shuffle=True, num_workers=4):

        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers,pin_memory=True)

    def val_dataloader(self, shuffle=True, num_workers=4):
        
        return DataLoader(self.valid_set, batch_size=1, shuffle=None, num_workers=num_workers,pin_memory=True)
    
    # def test_dataloader(self,shuffle=True, num_workers=4):

    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)

