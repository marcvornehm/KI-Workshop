from enum import Enum
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image


class SimpleLabel(Enum):
    pass


class BrainTumorLabel(SimpleLabel):
    NOTUMOR = 0
    TUMOR = 1


class CardiacViewLabel(SimpleLabel):
    LAX = 0
    SAX = 1


class CatsVsDogsLabel(SimpleLabel):
    CAT = 0
    DOG = 1


class SimpleDataset(torch.utils.data.Dataset):
    LabelNegative: SimpleLabel
    LabelPositive: SimpleLabel

    image_size: Tuple[int, int] = (256, 256)

    def __init__(self, normalize: bool = True, color: bool = False, crop_or_pad: str = 'pad'):
        self.data = []
        self.labels = []
        self.normalize = normalize
        self.color = color
        self.crop_or_pad = crop_or_pad

    def append_directory(self, path: Path, label: SimpleLabel, n_files: int | None = None):
        appended = 0
        for file in path.glob('*'):
            if n_files is not None and appended >= n_files:
                break
            img = Image.open(file)

            w, h = img.size
            min_size = min(w, h)
            max_size = max(w, h)
            if self.crop_or_pad == 'crop':
                target_size = (
                    int(w / min_size * self.image_size[0]),
                    int(h / min_size * self.image_size[1]),
                )
            elif self.crop_or_pad == 'pad':
                target_size = (
                    int(w / max_size * self.image_size[0]),
                    int(h / max_size * self.image_size[1]),
                )
            else:
                raise ValueError(f'Invalid crop_or_pad: {self.crop_or_pad}')
            img = img.resize(target_size)
            img = img.crop((
                (target_size[0] - self.image_size[0]) // 2,
                (target_size[1] - self.image_size[1]) // 2,
                (target_size[0] + self.image_size[0]) // 2,
                (target_size[1] + self.image_size[1]) // 2,
            ))

            if self.color:
                img = img.convert('RGB')
                img = np.asarray(img, dtype=np.float32)
                img = np.moveaxis(img, 2, 0)
            else:
                img = img.convert('L')
                img = np.asarray(img, dtype=np.float32)
                img = np.expand_dims(img, axis=0)

            self.data.append(img)
            self.labels.append(label)
            appended += 1

    def show_examples(self):
        _, axs = plt.subplots(3, 5, figsize=(15, 9))
        for i in range(15):
            idx = np.random.randint(len(self))
            img = self.data[idx]
            label = self.labels[idx]
            img -= np.min(img)
            img /= np.max(img)
            row = i // 5
            col = i % 5
            if self.color:
                axs[row, col].imshow(np.moveaxis(img, 0, 2))
            else:
                axs[row, col].imshow(img[0], cmap='gray')
            axs[row, col].set_title(f'Label: {label.name} ({label.value})')
            axs[row, col].axis('off')
        plt.tight_layout()
        plt.show()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        if self.normalize:
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / std
        label = self.labels[index].value
        return img, label


class BrainTumorDataset(SimpleDataset):
    LabelNegative = BrainTumorLabel.NOTUMOR
    LabelPositive = BrainTumorLabel.TUMOR

    def __init__(
            self,
            notumor: int | None = None,
            tumor: int | None = None,
            normalize: bool = True,
            color: bool = False,
            split: str = 'train',
        ):
        super().__init__(normalize=normalize, color=color, crop_or_pad='pad')
        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split: {split}')
        path_notumor = Path(__file__).parent.parent / 'data' / 'braintumor' / split / 'notumor'
        path_tumor = Path(__file__).parent.parent / 'data' / 'braintumor' / split / 'tumor'
        self.append_directory(path_notumor, self.LabelNegative, notumor)
        self.append_directory(path_tumor, self.LabelPositive, tumor)


class CardiacViewDataset(SimpleDataset):
    LabelNegative = CardiacViewLabel.LAX
    LabelPositive = CardiacViewLabel.SAX

    def __init__(
            self,
            long_axes: int | None = None,
            short_axes: int | None = None,
            normalize: bool = True,
            color: bool = False,
            split: str = 'train',
        ):
        super().__init__(normalize=normalize, color=color, crop_or_pad='pad')
        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split: {split}')
        path_lax = Path(__file__).parent.parent / 'data' / 'cardiacview' / split / 'lax'
        path_sax = Path(__file__).parent.parent / 'data' / 'cardiacview' / split / 'sax'
        self.append_directory(path_lax, self.LabelNegative, long_axes)
        self.append_directory(path_sax, self.LabelPositive, short_axes)


class CatsVsDogsDataset(SimpleDataset):
    LabelNegative = CatsVsDogsLabel.CAT
    LabelPositive = CatsVsDogsLabel.DOG

    def __init__(
            self,
            cats: int | None = None,
            dogs: int | None = None,
            normalize: bool = True,
            color: bool = False,
            split: str = 'train',
        ):
        super().__init__(normalize=normalize, color=color, crop_or_pad='crop')
        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split: {split}')
        path_cats = Path(__file__).parent.parent / 'data' / 'catsvsdogs' / split / 'cats'
        path_dogs = Path(__file__).parent.parent / 'data' / 'catsvsdogs' / split / 'dogs'
        self.append_directory(path_cats, self.LabelNegative, cats)
        self.append_directory(path_dogs, self.LabelPositive, dogs)


if __name__ == '__main__':
    dataset = BrainTumorDataset()
    print(f'Dataset length: {len(dataset)}')
    dataset.show_examples()
