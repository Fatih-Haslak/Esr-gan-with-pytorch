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
        nb : int = 1,
        gc : int = 32,
        lr: float = 0.0001,
        b1: float = 0.9,
        b2: float = 0.999,
        batch_size: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = RRDBNet(in_nc,out_nc,nf,nb,gc)
        self.discriminator = RelativisticDiscriminator(in_nc,nf)
        self.perceptual_loss = VGGPerceptualLoss()
        self.relativistic_loss =  RelativisticDiscriminatorLoss()


    def forward(self, z):
        return self.generator(z) #torch.image

    def __perceptual_loss(self, y_hat, y): #tahmin and real
        return self.perceptual_loss.forward(y_hat, y)
    def __relativistic_loss(self,y_hat,y):
        return self.relativistic_loss.forward(y_hat, y)

    def training_step(self, batch):
        imgs_high, imgs_low = batch
        optimizer_g, optimizer_d = self.optimizers()
        #train generator
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(imgs_low) #generator-e low olan resmi ver
        
        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, 0)

        perceptual__loss= self.perceptual_loss(self.generated_imgs,imgs_high) #perceptual loss
        real_scores, fake_scores= self.discriminator(imgs_high,self.generated_imgs) # discrimantorden sonuc al
        realitivistic__loss= self.__relativistic_loss(real_scores,fake_scores) # realitivistic loss
        g_loss=perceptual__loss+realitivistic__loss # toplam loss # bir tane daha gelcek
        self.log("g_loss", g_loss, prog_bar=True) #loggla
        self.manual_backward(g_loss,retain_graph=True)# backward
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        #generator bitti

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





    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.ResidualDenseBlock_5C[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)





if __name__ == "__main__":
    data_dir_high = '/home/fatih/esrgan/dataset/train/DIV2K_train_HR' 
    data_dir_low = '/home/fatih/esrgan/dataset/train_lr/DIV2K_train_LR_bicubic_X2/DIV2K_train_LR_bicubic/X2'
    test_data_dir = '/home/fatih/esrgan/dataset/test/DIV2K_valid_HR'
    batch_size=1
    div2k = DIV2KDataLoader(data_dir_high,data_dir_low,test_data_dir,batch_size)
    model=GAN()
    trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=3,
    )
    trainer.fit(model, div2k.train_dataloader(), div2k.val_dataloader())
    trainer.save_checkpoint("best_model.ckpt")