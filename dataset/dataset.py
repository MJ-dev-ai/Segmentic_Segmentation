from torch.uitls.data import Dataset
from torchvision.datasets import VOCSegmentation

class VOCSegmentationDataset(Dataset):
    def __init__(self, root='./data', year='2012', image_set='train', transform=None):
        self.dataset = VOCSegmentation(root=root,year=year,image_set=image_set,download=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        image, mask = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask