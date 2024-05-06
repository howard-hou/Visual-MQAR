from typing import Tuple, List

from datasets import load_dataset
import os
from visual_mqar.dataset import VisualMQARDataset
from visual_mqar.config import Config
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import patheffects as path_effects

os.environ["http_proxy"] = "http://127.0.0.1:8889"
os.environ["https_proxy"] = "http://127.0.0.1:8889"


def fetch_huggingface_dataset():
    dataset = load_dataset('nomodeset/idl_image-1k-hf')['data']
    config = Config()
    dataset = VisualMQARDataset(
        dataset=dataset,
        config=config
    )
    return dataset


def main():
    dataset = fetch_huggingface_dataset()
    print("dataset size:", len(dataset))
    for example in dataset:
        for key, value in example.items():
            print(f"{key}: {value.shape}")
        break

    num_items = 0

    for example in dataset.dataset:
        image = example["image"]
        image = np.asanyarray(image)
        print(f"{image.shape = }")
        size: List[float] = [i / 100 for i in image.shape]
        size: Tuple[float, float] = (size[1], size[0])
        fig = plt.figure(figsize=size, dpi=100)
        fig_image = fig.figimage(image)
        fig_image.set_path_effects([path_effects.Normal()])
        plt.show()
        if num_items > 10:
            break
        else:
            num_items += 1


if __name__ == '__main__':
    print("main")
    main()
