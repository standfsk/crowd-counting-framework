import os
from typing import List
from typing import Optional, Callable, Union, Tuple, Iterator

import cv2
import numpy as np
import torch
from core.transforms import get_transforms
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Normalize

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']


def generate_density_map(
    label: Tensor,
    height: int,
    width: int,
    sigma: Optional[float] = None,
) -> Tensor:
    """
    Generate the density map based on the dot annotations provided by the label.
    """
    density_map = torch.zeros((1, height, width), dtype=torch.float32)

    if len(label) > 0:
        assert len(label.shape) == 2 and label.shape[1] == 2, f"label should be a Nx2 tensor, got {label.shape}."
        label_ = label.long()
        label_[:, 0] = label_[:, 0].clamp(min=0, max=width - 1)
        label_[:, 1] = label_[:, 1].clamp(min=0, max=height - 1)
        density_map[0, label_[:, 1], label_[:, 0]] = 1.0

    if sigma is not None:
        assert sigma > 0, f"sigma should be positive if not None, got {sigma}."
        density_map = torch.from_numpy(gaussian_filter(density_map, sigma=sigma))

    return density_map


def collate_fn(
    batch: List[Tensor]
) -> Tuple[Tensor, List[Tensor], Tensor, List[str], List[np.ndarray]]:
    batch = list(zip(*batch))
    images = batch[0]
    assert len(images[0].shape) == 4, f"images should be a 4D tensor, got {images[0].shape}."
    images = torch.cat(images, 0)
    points = batch[1]  # list of lists of tensors, flatten it
    points = [p for points_ in points for p in points_]
    densities = torch.cat(batch[2], 0)
    data_paths = batch[3]  # list of lists of strings, flatten it
    data_paths = [path for path_ in data_paths for path in path_]
    original_images = batch[4]
    original_images = [img for img_ in original_images for img in img_]
    return images, points, densities, data_paths, original_images

class DatasetWithLabels(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        input_size: int,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        num_crops: int = 1,
    ) -> None:
        self.image_paths = self.get_image_paths(dataset_path)
        self.split = split

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transforms = transforms

        self.sigma = sigma
        self.num_crops = num_crops
        self.input_size = input_size

    def get_image_paths(self, dataset_path: str) -> List[str]:
        with open(dataset_path, "r") as f:
            image_paths = f.read().splitlines()
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, List[Tensor], Tensor, List[str], List[np.ndarray]]:
        image_path = self.image_paths[idx]
        label_path = image_path.replace(".jpg", ".npy")

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # image = cv2.imread(image_path)
        original_image = image.copy()
        image = self.to_tensor(image)

        with open(label_path, "rb") as f:
            label = np.load(f)

        label = torch.from_numpy(label).float()
        if self.transforms is not None:
            images_labels = [self.transforms(image.clone(), label.clone()) for _ in range(self.num_crops)]
            images, labels = zip(*images_labels)
        else:
            images = [image.clone() for _ in range(self.num_crops)]
            labels = [label.clone() for _ in range(self.num_crops)]

        images = [self.normalize(img) for img in images]
        density_maps = torch.stack(
            [generate_density_map(label, image.shape[-2], image.shape[-1], sigma=self.sigma) for image, label in
             zip(images, labels)], 0)

        data_paths = [image_path] * len(images)
        original_images = [original_image] * len(images)
        images = torch.stack(images, 0)
        return images, labels, density_maps, data_paths, original_images


class DatasetWithoutLabels(Dataset):
    def __init__(
        self,
        dataset_path: str,
        input_size: int
    ) -> None:
        self.image_paths, self.video_paths = self.get_media_paths(dataset_path)
        self.data_paths = self.image_paths + self.video_paths
        ni, nv = len(self.image_paths), len(self.video_paths)

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.input_size = input_size
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(self.video_paths):
            self.new_video(self.video_paths[0])
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found.'

    def get_media_paths(self, dataset_path: str) -> Tuple[List[str], List[str]]:
        image_paths, video_paths = [], []
        for root, _, files in os.walk(dataset_path):  # Walk through the folder
            for file in files:
                file_path = os.path.join(root, file)
                if file.split(".")[-1].lower() in img_formats:
                    image_paths.append(file_path)
                elif file.split(".")[-1].lower() in vid_formats:
                    video_paths.append(file_path)
        return image_paths, video_paths

    def __len__(self) -> int:
        return self.nf

    def __iter__(self) -> Iterator:
        self.count = 0
        return self

    def __next__(self) -> Tuple[Tensor, np.ndarray, str]:
        if self.count == self.nf:
            raise StopIteration
        data_path = self.data_paths[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret, original_image = self.cap.read()
            if not ret:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    data_path = self.data_paths[self.count]
                    self.new_video(data_path)
                    ret, original_image = self.cap.read()

            self.frame += 1

        else:
            # Read image
            self.count += 1
            original_image = cv2.imread(data_path)  # BGR
            assert original_image is not None, 'Image Not Found ' + data_path

        # image = cv2.resize(original_image, (self.input_size, self.input_size))
        image = self.to_tensor(original_image)
        image = self.normalize(image)

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        return image, original_image, data_path

    def new_video(self, video_path: str) -> None:
        self.frame = 0
        self.cap = cv2.VideoCapture(video_path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_dataloader(
    config: object,
    split: str = "train",
    ddp: bool = False,
) -> Union[Tuple[DataLoader, Union[DistributedSampler, None]], DataLoader]:
    if split == "train":  # train, strong augmentation
        transforms = get_transforms(config)
    else:
        transforms = None

    dataset = DatasetWithLabels(
        dataset_path=os.path.join("./datasets", f"{split}.txt"),
        split=split,
        input_size=config.input_size,
        transforms=transforms,
        sigma=None,
        num_crops=config.num_crops if split == "train" else 1
    )
    if ddp and split == "train":  # data_loader for training in DDP
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return data_loader, sampler

    elif split == "train":  # data_loader for training
        data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return data_loader, None

    else:  # data_loader for evaluation
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Use batch size 1 for evaluation
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return data_loader

