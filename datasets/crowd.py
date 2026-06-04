import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
# from turbojpeg import TurboJPEG, TJPF_RGB
# jpeg_decoder = TurboJPEG()

import numpy as np
from typing import Optional, Callable, Union, Tuple

from .utils import get_id, generate_density_map

curr_dir = os.path.dirname(os.path.abspath(__file__))

available_datasets = [
    "shanghaitech_a", "sha",
    "shanghaitech_b", "shb",
    "shanghaitech", "sh",
    "ucf_qnrf", "qnrf", "ucf-qnrf",
    "nwpu", "nwpu_crowd", "nwpu-crowd",
]

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)


def standardize_dataset_name(dataset: str) -> str:
    assert dataset.lower() in available_datasets, f"Dataset {dataset} is not available."
    if dataset.lower() in ["shanghaitech_a", "sha"]:
        return "sha"
    elif dataset.lower() in ["shanghaitech_b", "shb"]:
        return "shb"
    elif dataset.lower() in ["shanghaitech", "sh"]:
        return "sh"
    elif dataset.lower() in ["ucf_qnrf", "qnrf", "ucf-qnrf"]:
        return "qnrf"
    else:
        assert dataset.lower() in ["nwpu", "nwpu_crowd", "nwpu-crowd"], f"Dataset {dataset} is not available."
        return "nwpu"


class Crowd(Dataset):
    def __init__(
        self,
        dataset: str,
        split: str,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        return_filename: bool = False,
        num_crops: int = 1,
    ) -> None:
        """
        Dataset for crowd counting.
        """
        assert dataset.lower() in available_datasets, f"Dataset {dataset} is not available."
        assert dataset.lower() not in ["shanghaitech", "sh"], "For the combined ShanghaiTech dataset, use ShanghaiTech class."
        assert split in ["train", "val", "test"], f"Split {split} is not available."
        assert num_crops > 0, f"num_crops should be positive, got {num_crops}."

        self.dataset = standardize_dataset_name(dataset)
        self.split = split

        self.__find_root__()
        self.__make_dataset__()
        self.__check_sanity__()

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=mean, std=std)
        self.transforms = transforms

        self.sigma = sigma
        self.return_filename = return_filename
        self.num_crops = num_crops

    def __find_root__(self) -> None:
        self.root = os.path.join(curr_dir, "..", "data", self.dataset)

    def __make_dataset__(self) -> None:
        image_names = glob(os.path.join(self.root, self.split, "images", "*.jpg"))

        label_names = glob(os.path.join(self.root, self.split, "labels", "*.npy"))
        image_names = [os.path.basename(image_name) for image_name in image_names]
        label_names = [os.path.basename(label_name) for label_name in label_names]
        image_names.sort(key=get_id)
        label_names.sort(key=get_id)
        image_ids = tuple([get_id(image_name) for image_name in image_names])
        label_ids = tuple([get_id(label_name) for label_name in label_names])
        assert image_ids == label_ids, "image_ids and label_ids do not match."
        self.image_names = tuple(image_names)
        self.label_names = tuple(label_names)

    def __check_sanity__(self) -> None:
        if self.dataset == "sha":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 300, f"ShanghaiTech_A train split should have 300 images, but found {len(self.image_names)}."
            else:
                assert self.split == "val", f"Split {self.split} is not available for dataset {self.dataset}."
                assert len(self.image_names) == len(self.label_names) == 182, f"ShanghaiTech_A val split should have 182 images, but found {len(self.image_names)}."
        elif self.dataset == "shb":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 399, f"ShanghaiTech_B train split should have 399 images, but found {len(self.image_names)}."
            else:
                assert self.split == "val", f"Split {self.split} is not available for dataset {self.dataset}."
                assert len(self.image_names) == len(self.label_names) == 316, f"ShanghaiTech_B val split should have 316 images, but found {len(self.image_names)}."
        elif self.dataset == "nwpu":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 3109, f"NWPU train split should have 3109 images, but found {len(self.image_names)}."
            else:
                assert self.split == "val", f"Split {self.split} is not available for dataset {self.dataset}."
                assert len(self.image_names) == len(self.label_names) == 500, f"NWPU val split should have 500 images, but found {len(self.image_names)}."
        elif self.dataset == "qnrf":
            if self.split == "train":
                assert len(self.image_names) == len(self.label_names) == 1201, f"UCF_QNRF train split should have 1201 images, but found {len(self.image_names)}."
            else:
                assert self.split == "val", f"Split {self.split} is not available for dataset {self.dataset}."
                assert len(self.image_names) == len(self.label_names) == 334, f"UCF_QNRF val split should have 334 images, but found {len(self.image_names)}."

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, str]]:
        image_name = self.image_names[idx]
        label_name = self.label_names[idx]

        image_path = os.path.join(self.root, self.split, "images", image_name)
        label_path = os.path.join(self.root, self.split, "labels", label_name)

        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")
            # image = jpeg_decoder.decode(f.read(), pixel_format=TJPF_RGB)
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
        density_maps = torch.stack([generate_density_map(label, image.shape[-2], image.shape[-1], sigma=self.sigma) for image, label in zip(images, labels)], 0)
        image_names = [image_name] * len(images)
        images = torch.stack(images, 0)

        if self.return_filename:
            return images, labels, density_maps, image_names
        else:
            return images, labels, density_maps


class InMemoryCrowd(Dataset):
    def __init__(
        self,
        dataset: str,
        split: str,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        return_filename: bool = False,
        num_crops: int = 1,
    ) -> None:
        """
        Dataset for crowd counting, with images and labels loaded into memory.
        """
        crowd = Crowd(
            dataset=dataset,
            split=split,
            transforms=None,
            sigma=sigma,
            return_filename=True,
            num_crops=1,
        )
        print(f"Loading {len(crowd)} samples from {dataset} {split} split into memory...")
        self.images, self.labels, self.image_names = [], [], []
        self.unnormalize = Compose([
            Normalize(mean=(0., 0., 0.), std=(1./std[0], 1./std[1], 1./std[2]), inplace=True),
            Normalize(mean=(-mean[0], -mean[1], -mean[2]), std=(1., 1., 1.), inplace=True)
        ])

        for i in tqdm(range(len(crowd)), desc="Loading images and labels into memory"):
            image, label, _, image_name = crowd[i]
            self.images.append(self.unnormalize(image[0]))  # recover original image
            self.labels.append(label[0])
            self.image_names.append(image_name[0])
        
        assert len(self.images) == len(self.labels) == len(self.image_names), "Mismatch in number of images, labels, and image names."

        self.transforms = transforms
        self.sigma = sigma
        self.num_crops = num_crops
        self.return_filename = return_filename
        self.normalize = Normalize(mean=mean, std=std, inplace=False)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, str]]:
        image, label, image_name = self.images[idx].clone(), self.labels[idx].clone(), self.image_names[idx]

        if self.transforms is not None:
            images_labels = [self.transforms(image.clone(), label.clone()) for _ in range(self.num_crops)]
            images, labels = zip(*images_labels)
        else:
            images = [image.clone() for _ in range(self.num_crops)]
            labels = [label.clone() for _ in range(self.num_crops)]

        images = [self.normalize(img) for img in images]
        density_maps = torch.stack([generate_density_map(label, image.shape[-2], image.shape[-1], sigma=self.sigma) for image, label in zip(images, labels)], 0)
        image_names = [image_name] * len(images)
        images = torch.stack(images, 0)

        if self.return_filename:
            return images, labels, density_maps, image_names
        else:
            return images, labels, density_maps
    

class NWPUTest(Dataset):
    def __init__(
        self,
        transforms: Optional[Callable] = None,
        return_filename: bool = False,
    ) -> None:
        """
        The test set of NWPU-Crowd dataset. The test set is not labeled, so only images are returned.
        """
        self.root = os.path.join(curr_dir, "..", "data", "nwpu")
        image_names = glob(os.path.join(self.root, "test", "images", "*.jpg"))

        image_names = [os.path.basename(image_name) for image_name in image_names]
        assert len(image_names) == 1500, f"NWPU test split should have 1500 images, but found {len(image_names)}."
        image_names.sort(key=get_id)
        self.image_names = tuple(image_names)

        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=mean, std=std)
        self.transforms = transforms
        self.return_filename = return_filename

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, str]]:
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root, "test", "images", image_name)

        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")
            # image = jpeg_decoder.decode(f.read(), pixel_format=TJPF_RGB)
            image = self.to_tensor(image)
        
        label = torch.tensor([], dtype=torch.float)  # dummy label
        image, _ = self.transforms(image, label) if self.transforms is not None else (image, label)
        image = self.normalize(image)

        if self.return_filename:
            return image, image_name
        else:
            return image


class ShanghaiTech(Dataset):
    def __init__(
        self,
        split: str,
        transforms: Optional[Callable] = None,
        sigma: Optional[float] = None,
        return_filename: bool = False,
        num_crops: int = 1,
    ) -> None:
        super().__init__()
        self.sha = Crowd(
            dataset="sha",
            split=split,
            transforms=transforms,
            sigma=sigma,
            return_filename=return_filename,
            num_crops=num_crops,
        )
        self.shb = Crowd(
            dataset="shb",
            split=split,
            transforms=transforms,
            sigma=sigma,
            return_filename=return_filename,
            num_crops=num_crops,
        )
        self.dataset = "sh"
        self.split = split
        self.transforms = transforms
        self.sigma = sigma
        self.return_filename = return_filename
        self.num_crops = num_crops

    def __len__(self) -> int:
        return len(self.sha) + len(self.shb)
    
    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, str]]:
        if idx < len(self.sha):
            return self.sha[idx]
        else:
            return self.shb[idx - len(self.sha)]
