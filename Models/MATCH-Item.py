import torch
import torch.nn as nn
import torch.nn.functional as F

class MATCH(nn.Module):
    def __init__(self, n_items, n_cat, n_base, out_len):
        super().__init__()
        
        self.item = conv_block(in_channels=n_items*n_cat, out_channels=n_items, dprob=0,
                                    groups=n_items, kernel_size=1, bias=False)
        
        self.long1 = conv_block(n_items, 16, dprob=0.4, kernel_size=5, padding=3)
        self.mask1 = conv_block(n_items, 8, dprob=0.4, kernel_size=5, padding=3)
        
        self.long2 = conv_block(24, 16, dprob=0.4, kernel_size=5, padding=3)
        self.mask2 = conv_block(8, 8, dprob=0.4, kernel_size=5, padding=3)
        
        self.long3 = conv_block(24, 16, dprob=0.4, kernel_size=5, padding=3)
        self.mask3 = conv_block(8, 8, dprob=0.4, kernel_size=5, padding=3)
        
        self.long4 = conv_block(24, 16, dprob=0.4, kernel_size=3)
        
        self.survival = nn.Sequential(
            nn.Linear(16 + n_base, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.Linear(16,out_len)
        )
            
    
    def forward(self, long, base, mask):
        
        # reshape - stack n_item and n_cat dimensions
        x = long.reshape(long.shape[0],long.shape[1]*long.shape[2],long.shape[3])
        
        x = self.item(x)
        
        x = self.long1(x)
        mask = self.mask1(mask)
        x = torch.cat((x,mask),dim=1)
        
        x = self.long2(x)
        mask = self.mask2(mask)
        x = torch.cat((x,mask),dim=1)
        
        x = self.long3(x)
        mask = self.mask3(mask)
        x = torch.cat((x,mask),dim=1)
        
        x = self.long4(x)
        
        x = F.adaptive_avg_pool1d(x,1).squeeze()
        if base is not None:
            x = torch.cat((x,base),dim=1)
        x = self.survival(x)
        return x
        
