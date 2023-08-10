import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativisticDiscriminator(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64):
        super(RelativisticDiscriminator, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels*4, hidden_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.final_layer = nn.Conv2d(hidden_channels*8, 1, kernel_size=4, stride=1, padding=0)

        self.linear_arch=nn.Sequential(
            nn.Linear(in_features=hidden_channels*8, out_features=512),  
            nn.ReLU(),  # Aktivasyon fonksiyonu
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=128),   
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=32,out_features=1)
        )

    
    def forward(self, real_input, fake_input):
        real_features = self.shared_layers(real_input) # Real input
        fake_features = self.shared_layers(fake_input) # GENERATOR OUTPUT
        #linear model
        
        real_scores = self.final_layer(real_features)
        fake_scores = self.final_layer(fake_features)
        
        return real_scores, fake_scores