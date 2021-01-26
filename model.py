import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiScaleBlock(nn.Module):
    def __init__(self,inc,ouc,branchs=6,stride=1,reduction=16):
        super(MultiScaleBlock,self).__init__()
        self.inconvs = nn.ModuleList([])
        for branch in range(branchs):
            self.inconvs.append(nn.Sequential(
                nn.Conv1d(inc, ouc, kernel_size=3 + branch * 2, padding=branch+1,stride=stride, bias=False),
                nn.BatchNorm1d(ouc,eps=0.001,momentum=0.01),
            ))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ouc*branchs,ouc*branchs//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ouc*branchs//reduction,ouc),
            nn.Sigmoid(),
        )
        self.ouconvs = nn.Sequential(
            nn.Conv1d(branchs*ouc,ouc,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm1d(ouc,eps=0.001,momentum=0.01),
        )
    def forward(self,x):
        for i, conv in enumerate(self.inconvs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        z = self.pool(feas).squeeze(-1)
        mask = self.fc(z).unsqueeze(-1)
        out = self.ouconvs(feas)
        out = mask*out
        return F.relu(out)

class ScaleAdaptiveNet(nn.Module):
    def __init__(self,num_classes=30):
        super(ScaleAdaptiveNet, self).__init__()
        self.feat = nn.Sequential(
            MultiScaleBlock(1,16,stride=2),  #500
            MultiScaleBlock(16,32,stride=2), #250
            MultiScaleBlock(32,64,stride=2), #125
            MultiScaleBlock(64,128,stride=2),#63
            MultiScaleBlock(128,192,stride=2),#32
            nn.Conv1d(192,32,1,1,bias=False),
            nn.BatchNorm1d(32,eps=0.001,momentum=0.01),
            nn.Dropout(0.5)
        )
        self.classify = nn.Linear(1024,num_classes)
    def forward(self,x):
        out = self.feat(x)
        out = out.flatten(start_dim=1)
        out = self.classify(out)
        return out
    def getgradCAM(self,x,target):
        self.eval()
        f = self.feat(x).detach()
        f.requires_grad_()
        out = f.flatten(start_dim=1)
        out = self.classify(out)
        onehot = torch.zeros(out.shape).to(x.device)
        for i in range(len(target)):
            onehot[i][target[i].long().item()]=1
        out = onehot*out
        out = out.sum()
        out.backward()
        mask = f.grad
        mask = mask.mean(-1,keepdim=True)
        cam = (f*mask).sum(dim=1)
        cam = F.relu(cam).unsqueeze(1)
        cam = F.interpolate(cam,scale_factor=x.shape[-1]/cam.shape[-1],mode='linear',align_corners=True,recompute_scale_factor=True)
        cam = (cam-cam.min())/(cam.max()-cam.min())
        return cam.squeeze(1)