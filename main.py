from config import flags
from train.train import train
from models.models import EfficientUnet
from dataset.dataset import VOCSegmentationDataset
from torch.utils.data import DataLoader
from utils.utils import overlay_mask_on_image
import numpy as np
import torch
from torch import optim, nn
from torchvision import transforms

def mask_to_tensor(mask):
    # PIL Image -> numpy -> torch.LongTensor
    return torch.from_numpy(np.array(mask)).long()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((224,224)),
    mask_to_tensor
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = VOCSegmentationDataset(root='./data/VOC2012', image_set='train', transform=transform, mask_transform=mask_transform)
train_loader = DataLoader(dataset, batch_size=flags['batch_size'], shuffle=True)
model = EfficientUnet(num_classes=21, model_name='efficientnet-b0').to(device)
model.load_state_dict(torch.load(flags['save_path']))
optimizer = optim.Adam(model.parameters(), lr=flags['learning_rate'])
weights = torch.ones(21)
weights[0] = 0.1

loss_fn = nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=255)

#train(model, train_loader, loss_fn, optimizer, device, flags)

model.eval()
dataset = VOCSegmentationDataset(root='./data/VOC2012', image_set='val', transform=transform, mask_transform=mask_transform)

import matplotlib.pyplot as plt
import random
l = len(dataset)
rand_idx = random.randint(0, l-1)
img, mask = dataset[rand_idx]
img, mask = img.to(device), mask.to(device)
img = img.unsqueeze(0)
with torch.no_grad():
    output = model(img)
    pred_mask = torch.argmax(output, dim=1).squeeze(0)
    img = img.squeeze(0)
    pred = overlay_mask_on_image(img, pred_mask, alpha=0.5)
    truth = overlay_mask_on_image(img, mask, alpha=0.5)
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.title('Input Image')
    plt.imshow(img.permute(1,2,0).cpu().numpy())
    plt.subplot(1,3,2)
    plt.title('Ground Truth Mask')
    plt.imshow(truth)
    plt.subplot(1,3,3)
    plt.title('Predicted Mask')
    plt.imshow(pred)
    plt.show()
