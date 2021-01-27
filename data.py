
from transform import *
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import StratifiedKFold
from config import ATCC_GROUPINGS
from torchvision import transforms
class SpectralDataset(Dataset):
    """
        Builds a dataset of spectral data. Use Str dataset,which can be selected as '2018clinical','2019clinical',
        'reference','finetune' or 'test'to determine which dataset we use.
        Use index to specify which samples we use in the selected dataset.
    """
    def __init__(self,X,y,index_list=None,transform=None,antibiotics=False,mssa_mrsa=False):
        self.X = X
        self.y = y
        self.transform = transform
        self.antibiotics = antibiotics
        self.mssa_mrsa = mssa_mrsa
        if index_list is None:
            self.index_list = np.arange(len(X))
        else:
            self.index_list = index_list

    def __getitem__(self, index):
        index = self.index_list[index]
        signal, target = self.X[index],self.y[index].astype(int)
        if self.transform:
            signal = self.transform(signal)
        if self.antibiotics:
            target = ATCC_GROUPINGS[int(target)]
        if self.mssa_mrsa:
            target = 0. if target in [14,15,18] else 1.
        return signal, target

    def __len__(self):
        return len(self.index_list)


train_transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomChoice([
        RandomBlur(),
        AddGaussianNoise(),
    ])],p=0.5),
    transforms.RandomApply([RandomDropout()], p=0.5),
    transforms.RandomApply([RandomScaleTransform()], p=0.5),
    ToFloatTensor()
])

valid_transform = ToFloatTensor()

def spectralloader(dataset,batch_size=16,num_workers=12,antibiotics = False,mssa_mrsa=False,num_folds = 5,seed=42):

    X = np.load('data/X_' + dataset + '.npy')
    y = np.load('data/y_' + dataset + '.npy')
    if mssa_mrsa:
        l = y.shape[0]//30
        X = X[14*l:19*l]
        y = y[14*l:19*l]
    if dataset!='test':
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        x = np.arange(len(y))
        fold ={}
        i=1
        for train,val in skf.split(x,y):
            trainset = SpectralDataset(X,y,train,train_transform,antibiotics=antibiotics,mssa_mrsa=mssa_mrsa)
            valset = SpectralDataset(X,y,val,valid_transform,antibiotics=antibiotics,mssa_mrsa=mssa_mrsa)
            trainloader  = DataLoader(trainset,batch_size=batch_size,num_workers=num_workers,shuffle=True,pin_memory=True)
            valloader = DataLoader(valset,batch_size=batch_size,num_workers=num_workers,shuffle=False,pin_memory=True)
            fold[i] = {"train":trainloader,"val":valloader}
            i+=1
            return fold
    else:
        testset = SpectralDataset(X, y, None, valid_transform, antibiotics=antibiotics,mssa_mrsa=mssa_mrsa)
        return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
