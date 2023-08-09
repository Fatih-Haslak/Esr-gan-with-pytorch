import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativisticDiscriminatorLoss(nn.Module):
    def __init__(self):
        super(RelativisticDiscriminatorLoss, self).__init__()

    def forward(self, real_scores, fake_scores):
        # Discriminator'un gerçek ve sahte çıkışlarının karşılaştırılması
        loss_real = F.binary_cross_entropy_with_logits(real_scores - fake_scores.mean(), torch.ones_like(real_scores))
        loss_fake = F.binary_cross_entropy_with_logits(fake_scores - real_scores.mean(), torch.zeros_like(fake_scores))
        
        # Toplam loss hesaplaması
        total_loss = (loss_real + loss_fake) * 0.5
        
        return total_loss

