from datasets import load_dataset
import os
from visual_mqar.dataset import VisualMQARDataset
from visual_mqar.config import Config

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


if __name__ == '__main__':
    print("main")
    main()
