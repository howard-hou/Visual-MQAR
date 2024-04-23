import torch
from torch import optim
from visual_mqar.model import VisualMQAR
from datasets import load_dataset
from visual_mqar.dataset import VisualMQARDataset
from visual_mqar.config import Config
import schedulefree
import os

os.environ["http_proxy"] = "http://127.0.0.1:8889"
os.environ["https_proxy"] = "http://127.0.0.1:8889"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    config = Config()
    model = VisualMQAR(config=config)
    model = model.train().to(device)
    return model


def fetch_huggingface_dataset():
    dataset = load_dataset('nomodeset/idl_image-1k-hf')['data']
    config = Config()
    dataset = VisualMQARDataset(
        dataset=dataset,
        config=config
    )
    return dataset


def train_loop(train_dataset, model, optimizer, epoch, n_epochs, scheduler=None):
    loss_value = 0.
    ii = 0
    for sample in train_dataset:
        for key, value in sample.items():
            print(key, value.shape)
            sample[key] = value.to(device)
        optimizer.zero_grad()
        loss = model.training_step(sample)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_value += loss.detach().cpu().numpy().mean()
        ii += 1
    loss_value /= ii
    print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {loss_value:.4f}')


def train_with_adam():
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=4e-4, betas=(0.7, 0.96), eps=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    train_dataset = fetch_huggingface_dataset()
    n_epochs = 2000
    for epoch in range(n_epochs):
        train_loop(
            train_dataset=train_dataset,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            scheduler=scheduler,
            n_epochs=n_epochs
        )


def train_with_schedulefree():
    model = build_model()
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=4e-4, betas=(0.7, 0.96), eps=1e-9)
    train_dataset = fetch_huggingface_dataset()
    n_epochs = 2000
    for epoch in range(n_epochs):
        train_loop(
            train_dataset=train_dataset,
            model=model,
            epoch=epoch,
            optimizer=optimizer,
            n_epochs=n_epochs
        )


def main():
    # train_with_adam()
    train_with_schedulefree()


if __name__ == '__main__':
    main()
