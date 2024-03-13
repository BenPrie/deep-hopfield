# Imports as always.
import os
import re
from copy import deepcopy

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


# Pre-define a couple of transform functions to and from tensors and images.
tensor_to_image = transforms.ToPILImage()
image_to_tensor = transforms.ToTensor()


# A custom dataset object for handling the ISIC2018 data.
class ISICDataset(Dataset):
    def __init__(
            self, root_path,
            input_path,
            target_path,
            transform=False,
            damage=False,
            damage_degree=-1,
            image_width=-1,
            image_height=-1
    ):
        super(ISICDataset, self).__init__()

        self.root_path = root_path
        self.input_path = input_path
        self.target_path = target_path
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        self.damage = damage
        self.damage_degree = damage_degree

        # Get a list of all the image IDs.
        pattern = re.compile(r'\d+')
        self.indices = []
        for filename in os.listdir(os.path.join(self.root_path, self.input_path)):
            matches = re.findall(pattern, filename)
            if len(matches) != 0:
                self.indices.append(int(matches[0]))

    def __len__(self):
        # List the inputs and target masks.
        inputs = os.listdir(os.path.join(self.root_path, self.input_path))
        targets = os.listdir(os.path.join(self.root_path, self.target_path))

        # Count each.
        n_inputs = len(inputs)
        n_targets = len(targets)

        # If these aren't the same, there's some kind of mismatch.
        assert n_inputs == n_targets, 'Mismatched number of inputs and targets.'

        # Because these folders contain additional files, this number is too large.
        # The number of indices counted in the __init__ gives the length of the dataset.
        return len(self.indices)

    def __getitem__(self, idx):
        # Translate the given idx into the corresponding image index.
        isic_id = self.indices[idx]

        input_path = os.path.join(self.root_path, self.input_path, f'ISIC_{isic_id:07d}.jpg')
        target_path = os.path.join(self.root_path, self.target_path, f'ISIC_{isic_id:07d}_segmentation.png')

        input_image = Image.open(input_path)
        target_mask = Image.open(target_path)

        # Apply transforms.
        if self.transform:
            input_image = self.transform_image(input_image)
            target_mask = self.transform_image(target_mask)

        # Apply damage (to input only).
        if self.damage:
            input_image = self.damage_image(input_image)

        return input_image, target_mask

    def transform_image(self, image, interpolation_mode=transforms.InterpolationMode.BILINEAR):
        # Resize the image.
        if not (self.image_width < 1 or self.image_height < 1):
            image = transforms.Resize((self.image_height, self.image_width), interpolation=interpolation_mode)(image)

        # Convert to tensor.
        image = image_to_tensor(image)

        return image

    def damage_image(self, image, damage_method='random-corrupt'):
        # Damage by random corruption -- zeroing random pixels.
        if damage_method == 'random-corrupt':
            masked_image = deepcopy(image)
            _, height, _ = image.shape

            # Randomly zero damage_degree% of pixels to damage. We consider all channels together -- damaging pixels.
            zero_indices = np.random.choice(height * height, int(height * height * self.damage_degree), replace=False)
            for idx in zero_indices:
                x, y = np.divmod(idx, height)
                masked_image[:, x, y] = 0

            return masked_image

        else:
            return image