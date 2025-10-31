import torch
import torch.nn as nn
import torch.nn.functional as F

class MATCH(nn.Module):
    def __init__(self, n_items, n_cat, n_base, out_len,
                 l1=16, l23=16, l4=16, lmask=8, llin=16,
                 pos_constraint = "relu"):
        super().__init__()
        
        self.item = conv_block(in_channels=n_items*n_cat, out_channels=n_items, dprob=0,
                                    groups=n_items, kernel_size=1, bias=False,
                                    pos_constraint=pos_constraint)
        
        self.long1 = conv_block(n_items, l1, dprob=0.4, kernel_size=5, padding=3)
        self.mask1 = conv_block(n_items, lmask, dprob=0.4, kernel_size=5, padding=3)
        
        self.long2 = conv_block(l1 + lmask, l23, dprob=0.4, kernel_size=5, padding=3)
        self.mask2 = conv_block(lmask, lmask, dprob=0.4, kernel_size=5, padding=3)
        
        self.long3 = conv_block(l23 + lmask, l23, dprob=0.4, kernel_size=5, padding=3)
        self.mask3 = conv_block(lmask, lmask, dprob=0.4, kernel_size=5, padding=3)
        
        self.long4 = conv_block(l23 + lmask, l4, dprob=0.4, kernel_size=3)
        
        self.survival = nn.Sequential(
            nn.Linear(l4 + n_base, llin),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(llin),
            nn.Linear(llin, llin),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(llin),
            nn.Linear(llin,out_len)
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
        
    
class relu_constraint(nn.Module):
    def forward(self, x):
        return F.relu(x)


class softplus_constraint(nn.Module):
    def forward(self, x):
        return F.softplus(x)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, dprob, pos_constraint="none", **kwargs):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, padding_mode='replicate', **kwargs),
            nn.ReLU(),
            nn.Dropout(dprob),
            nn.BatchNorm1d(out_channels)
            )
        
        if pos_constraint == "relu":
            nn.utils.parametrize.register_parametrization(self.convolution[0], "weight", relu_constraint())
        
        if pos_constraint == "softplus":
            nn.utils.parametrize.register_parametrization(self.convolution[0], "weight", softplus_constraint())
        
        if pos_constraint == "none":
            pass
        
    def forward(self, x):
            return self.convolution(x)

