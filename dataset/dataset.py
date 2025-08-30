from torch.utils.data import Dataset
import os
from PIL import Image

class VOCSegmentationDataset(Dataset):
    def __init__(self, root='./data/VOC2012', image_set='train', transform=None, mask_transform=None):
        image_list_fname = image_set + '.txt'
        # 해당 데이터셋의 이미지 파일 이름을 담은 텍스트 파일 경로
        image_list_path = os.path.join(root, 'ImageSets', 'Segmentation', image_list_fname)
        # 이미지 파일 이름을 읽어 리스트로 저장
        if os.path.exists(image_list_path):
            with open(image_list_path, 'r') as file:
                self.image_ids = [line.strip() for line in file.readlines()]
        else:
            raise FileNotFoundError(f"{image_list_path} does not exist.")
        # 실제 이미지와 마스크가 있는 경로
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.mask_dir = os.path.join(root, 'SegmentationClass')
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # idx에 해당하는 이미지와 마스크의 파일 경로 설정
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_id + '.png')
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask