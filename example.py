import torch

from src.models.models import NoiseModule


if __name__ == "__main__":
    
    input = torch.randn((128, 9))
    nm = NoiseModule(shape=(9,))
    out = nm(input)
    print(f"{out.shape=}")
    pass