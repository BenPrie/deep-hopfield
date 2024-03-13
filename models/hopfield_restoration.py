# Imports as always.
import numpy as np
import torch
from torch import nn
from hflayers import HopfieldLayer


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_device('cuda')

def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

def get_sinusoidal_encoding(n_tokens, token_length):
    def get_position_angle_vector(i):
        return [i / np.power(10000, 2 * (j // 2) / token_length) for j in range(token_length)]

    table = np.array([get_position_angle_vector(i) for i in range(n_tokens)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])

    return torch.cuda.FloatTensor(table).unsqueeze(0)


class Embedding(nn.Module):
    def __init__(self, image_size, patch_size, channels, embed_dim):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        self.embed_dim = int(embed_dim)

        # Trainable linear projection for mapping dimension of patches.
        self.W_E = nn.Parameter(torch.randn(self.patch_size * self.patch_size * self.channels, self.embed_dim))

        # Fixed sinusoidal positional embedding.
        self.n_patches = self.image_size ** 2 // self.patch_size ** 2
        self.PE = get_sinusoidal_encoding(n_tokens=self.n_patches, token_length=self.embed_dim)

    def forward(self, x):
        # Patching.
        patches = x.unfold(1, self.channels, self.channels).unfold(2, self.patch_size, self.patch_size).unfold(3,
                                                                                                               self.patch_size,
                                                                                                               self.patch_size)
        patches = patches.contiguous().view(patches.size(0), -1,
                                            self.channels * self.patch_size * self.patch_size).float()

        # Patch embeddings.
        patch_embeddings = torch.matmul(patches, self.W_E)

        # Position embeddings.
        embeddings = patch_embeddings + self.PE

        # Transpose so that each column represents a patch embedding.
        embeddings = torch.transpose(embeddings, 1, 2)

        return embeddings


class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, output_dim):
        super().__init__()

        # Linear projection to take us from the association space to the full output space.
        self.projection = nn.Linear(embed_dim, patch_size * patch_size * output_dim)

        # And fold into an image shape.
        self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # We transposed at the end of the encoder, so we must transpose at the start of the decoder.
        x = torch.transpose(x, 1, 2)

        # Project.
        x = self.projection(x)

        # Fold into shape.
        x = x.permute(0, 2, 1)
        x = self.fold(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)

        weighted = torch.transpose(x, 1, 2)

        return weighted


def build_hopfield_restoration_block(image_size, patch_size, channels, embed_dim, stored_patterns):
    return nn.Sequential(
        Embedding(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim
        ),
        HopfieldLayer(
            input_size=(image_size // patch_size) ** 2,
            quantity=stored_patterns,

            lookup_targets_as_trainable=True,
            stored_pattern_as_static=True,
            state_pattern_as_static=True,
            pattern_projection_as_static=True
        ),
        SelfAttention(
            input_dim=embed_dim
        ),
        OutputProjection(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            output_dim=channels
        )
    )