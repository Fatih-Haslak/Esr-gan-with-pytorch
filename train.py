import os
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data.dataloader import DIV2KDataLoader
from model.generator import RRDBNet
from model.discriminator import RelativisticDiscriminator
from utils.perceptual_loss import VGGPerceptualLoss
from utils.relativistic_loss import RelativisticDiscriminatorLoss


class GAN(L.LightningModule):
    def __init__(
        self,
        in_nc : int = 3,
        out_nc : int = 3,
        nf : int = 64,
        nb : int = 2, # increase by gpu perfonmance
        gc : int = 32,
        lr: float = 0.0001,
        b1: float = 0.9,
        b2: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = RRDBNet(in_nc,out_nc,nf,nb,gc)
        self.discriminator = RelativisticDiscriminator(in_nc,nf)
        self.perceptual_loss = VGGPerceptualLoss()
        self.relativistic_loss =  RelativisticDiscriminatorLoss()


    def forward(self, z):
        return self.generator(z) #torch.image generator forward

    def __perceptual_loss(self, y_hat, y): #predict and real 
        return self.perceptual_loss.forward(y_hat, y)

    def __relativistic_loss(self,y_hat,y):
        return self.relativistic_loss.forward(y_hat, y)

    def training_step(self, batch,batch_idx):
        #high-r image, #low-r image
        imgs_high, imgs_low = batch

        optimizer_g, optimizer_d = self.optimizers()
        
        #train generator
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(imgs_low) #generator-e low olan resmi ver

        #calculate loss
        perceptual__loss= self.perceptual_loss(self.generated_imgs,imgs_high) #perceptual loss
        real_scores, fake_scores= self.discriminator(imgs_high,self.generated_imgs) # discrimantorden sonuc al
        realitivistic__loss= self.__relativistic_loss(real_scores,fake_scores) # realitivistic loss
        g_loss=perceptual__loss+realitivistic__loss # toplam loss # bir tane daha gelcek ?
        
        self.log("g_loss", g_loss, prog_bar=True) #loggla
        
        #backward generator
        self.manual_backward(g_loss,retain_graph=True)# backward
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        #generator end

        #discrimatoru eğit
        self.toggle_optimizer(optimizer_d)
        real_scores, fake_scores=self.discriminator(imgs_high,self.generated_imgs)
        realitivistic__loss= self.__relativistic_loss(real_scores,fake_scores)
        self.log("d_loss", realitivistic__loss, prog_bar=True) #loggla
        self.manual_backward(realitivistic__loss,retain_graph=True)# backward
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        #disc bitti train bu skeılde

    def validation_step(self, batch, batch_idx):
        x, y = batch
        #x=self.validation_z.type_as(x)
        z = self.generator(y)
        real_scores, fake_scores = self.discriminator(x,z)
        realitivistic__loss= self.__relativistic_loss(real_scores,fake_scores)
        perceptual__loss= self.perceptual_loss(z,x)
        loss = realitivistic__loss + perceptual__loss
        self.log('val_loss', loss,on_epoch=True,prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []


if __name__ == "__main__":
    data_dir_high = '/home/fatih/esrgan/dataset/train_hr/DIV2K_train_HR' #high res image
    data_dir_low = '/home/fatih/esrgan/dataset/train_lr/DIV2K_train_LR_bicubic_X2/DIV2K_train_LR_bicubic/X2' #low res image
    test_data_dir = '/home/fatih/esrgan/dataset/test/DIV2K_valid_HR' #test data don't any train steps on this data
    batch_size= 2
    div2k = DIV2KDataLoader(data_dir_high,data_dir_low,test_data_dir,batch_size)
    model=GAN()
    #accelerator="gpu", devices=8, strategy="ddp", num_nodes=4
    trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=100,
    #strategy="ddp_find_unused_parameters_true",
    #num_nodes=1
    )
    trainer.fit(model, div2k.train_dataloader(), div2k.val_dataloader())
    trainer.save_checkpoint("best_model.ckpt")
 