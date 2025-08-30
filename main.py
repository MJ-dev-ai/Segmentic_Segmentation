from config import flags
from train.train import train
from models.models import EfficientUnet
from dataset.dataset import VOCSegmentationDataset
from torch.utils.data import DataLoader
import torch
from torch import optim, nn
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mask_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
dataset = VOCSegmentationDataset(root='./data/VOC2012', image_set='train', transform=transform, mask_transform=mask_transform)
train_loader = DataLoader(dataset, batch_size=flags['batch_size'], shuffle=True)
model = EfficientUnet(num_classes=21, model_name='efficientnet-b0')
optimizer = optim.Adam(model.parameters(), lr=flags['learning_rate'])
loss_fn = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(model, train_loader, loss_fn, optimizer, device, flags)

model = EfficientUnet(num_classes=21, model_name='efficientnet-b0')
model.load_state_dict(torch.load(flags['save_path']))
model.eval()
dataset = VOCSegmentationDataset(root='./data/VOC2012', image_set='val', transform=transform, mask_transform=mask_transform)

import matplotlib.pyplot as plt
import random
l = len(dataset)
rand_idx = random.randint(0, l-1)
img, mask = dataset[rand_idx]
img = img.unsqueeze(0)
with torch.no_grad():
    output = model(img)
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.title('Input Image')
    plt.imshow(img.squeeze(0).permute(1,2,0).cpu().numpy())
    plt.subplot(1,3,2)
    plt.title('Ground Truth Mask')
    plt.imshow(mask.squeeze(0).cpu().numpy())
    plt.subplot(1,3,3)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask)
    plt.show()
