import numpy as np
import torch
from tqdm import tqdm
import random
import os
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count

class EarlyStop:
    def __init__(self,patience=10,mode='max',delta=0.0001):
        self.patientce=patience
        self.counter=0
        self.mode = mode
        self.best_score=None
        self.early_stop=False
        self.delta = delta
        if self.mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
    def __call__(self,epoch_score,model,model_path):
        if self.mode=='min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score,model,model_path)
        elif score<self.best_score+self.delta:
            self.counter+=1
            if self.counter>=self.patientce:
                self.early_stop=True
        else:
            self.best_score=score
            self.save_checkpoint(epoch_score,model,model_path)
            self.counter=0

    def save_checkpoint(self,epoch_score,model,model_path):
        torch.save(model.state_dict(),model_path)
        self.val_score = epoch_score

class Engine:
    @staticmethod
    def train(fold,epoch,data_loader,lossF,model,optimizer,device,sheduler,accumulation_steps=1,fp16=True,sheduler_type='ReduceLROnPlateau',binary=False):
        losses = AverageMeter()
        accuracies = AverageMeter()
        model.train()
        scalar = torch.cuda.amp.GradScaler()
        bar = tqdm(data_loader)
        for batch,(data,target) in enumerate(bar):
            iter = len(bar)
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            if binary:
                target = target.unsqueeze(1)
            if batch % accumulation_steps == 0:
                optimizer.zero_grad()
            if fp16:
                with torch.cuda.amp.autocast():
                    out = model(data)

                    loss = lossF(out, target)
                scalar.scale(loss).backward()
                if batch % accumulation_steps == accumulation_steps - 1:
                    if sheduler_type=='CosineAnnealingLR':
                        sheduler.step(epoch + batch / iter - 1)
                    scalar.step(optimizer)
                    scalar.update()
            else:
                out = model(data)
                loss = lossF(out, target)
                loss.backward()
                if batch % accumulation_steps == accumulation_steps - 1:
                    if sheduler_type=='CosineAnnealingLR':
                        sheduler.step(epoch + batch / iter - 1)
                    optimizer.step()
            if binary:
                predictions = (torch.sigmoid(out)>0.5).int().view(-1).detach().cpu().numpy()
                target = target.int().view(-1)
            else:
                predictions = torch.argmax(out,dim=1).view(-1).detach().cpu().numpy()
            accuracy = (predictions == target.detach().cpu().numpy()).mean()
            losses.update(loss.item(),batch_size)
            accuracies.update(accuracy.item(),batch_size)
            bar.set_description(
                    "Fold%d, Epoch%d : Average train loss is %.6f, the accuracy rate of trainset is %.4f " % (fold,epoch,losses.avg,accuracies.avg))

        return accuracies.avg,losses.avg
    @staticmethod
    def evaluate(fold,epoch,data_loader,lossF,model,device,binary=False):
        accuracies = AverageMeter()
        losses = AverageMeter()
        model.eval()
        bar = tqdm(data_loader)
        with torch.no_grad():
            for i,(data,target) in enumerate(bar):
                batch_size = data.size(0)
                data,target = data.to(device),target.to(device)
                if binary:

                    target = target.unsqueeze(1)

                out = model(data)
                loss = lossF(out, target)
                if binary:
                    predictions = (torch.sigmoid(out) > 0.5).int().view(-1).detach().cpu().numpy()
                    target = target.int().view(-1)
                else:
                    predictions = torch.argmax(out, dim=1).view(-1).detach().cpu().numpy()
                accuracy = (predictions == target.detach().cpu().numpy()).mean()
                losses.update(loss.item(), batch_size)
                accuracies.update(accuracy.item(),batch_size)
                bar.set_description(
                    "Fold%d, Epoch%d,: Average validate loss is %.6f, the accuracy rate of valset is %.4f " % (fold,epoch,losses.avg, accuracies.avg))

        return accuracies.avg,losses.avg
    def test(data_loader,lossF,model,device,binary=False):
        accuracies = AverageMeter()
        losses = AverageMeter()
        model.eval()
        bar = tqdm(data_loader)
        probs = []
        predicticted = []
        true = []
        with torch.no_grad():
            for i,(data,target) in enumerate(bar):
                data,target = data.to(device),target.to(device)
                batch_size = data.size(0)
                if binary:
                    target = target.unsqueeze(1)
                out = model(data)
                loss = lossF(out, target)

                if binary:
                    prob = torch.sigmoid(out).view(-1)
                    predictions = (prob > 0.5).int().detach().cpu().numpy()
                    target = target.int().view(-1)
                else:
                    predictions = torch.argmax(out, dim=1).view(-1).detach().cpu().numpy()
                    prob = torch.softmax(out,dim=-1).detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                accuracy = (predictions == target).mean()

                losses.update(loss.item(), batch_size)
                accuracies.update(accuracy.item(),batch_size)
                bar.set_description(
                    "Average test loss is %.6f, the accuracy rate of testset is %.4f " % (losses.avg, accuracies.avg))
                predicticted += predictions.tolist()
                true += target.tolist()
                probs += prob.tolist()
        return np.array(probs),np.array(predicticted),np.array(true),accuracies.avg,losses.avg

