from datasets import load_dataset
import os

os.environ["http_proxy"] = "http://127.0.0.1:8889"
os.environ["https_proxy"] = "http://127.0.0.1:8889"


if __name__ == '__main__':
    dataset = load_dataset('nomodeset/idl_image-1k-hf')['data']
    from visual_mqar.dataset import VisualMQARDataset
    dataset = VisualMQARDataset(dataset)
    for example in dataset:
        print(example)
        break

