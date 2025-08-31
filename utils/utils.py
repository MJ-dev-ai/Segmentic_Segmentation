import numpy as np
import torch
from PIL import Image
import cv2

def overlay_mask_on_image(image, mask, alpha=0.5, colormap=None):
    """
    이미지 위에 마스크를 alpha blending으로 투영합니다. (OpenCV 사용)
    image: PIL.Image, torch.Tensor (C,H,W 또는 H,W,C), 또는 np.ndarray (H,W,3)
    mask:  (H,W) torch.Tensor 또는 np.ndarray (클래스 인덱스)
    alpha: float, 마스크 투명도 (0~1)
    colormap: (선택) 클래스별 RGB 색상 리스트, 예: [(0,0,0), (255,0,0), ...]
    """
    # 이미지 numpy 변환
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in [1,3]:
            image = image.permute(1,2,0).cpu().numpy()
        else:
            image = image.cpu().numpy()
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise ValueError("Unsupported image type")
    # mask numpy 변환
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)
    # colormap 지정 (VOC 기본 21 클래스)
    if colormap is None:
        colormap = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
            (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
            (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
            (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
            (0, 64, 128)
        ]
    # 마스크를 컬러로 변환 (H,W,3)
    color_mask = np.zeros(image.shape, dtype=np.uint8)
    for class_idx, color in enumerate(colormap):
        color_mask[mask == class_idx] = color
    # OpenCV addWeighted로 blending
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return Image.fromarray(overlay)