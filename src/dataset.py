from torch.utils.data import Dataset


class VisualMQARDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, config):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.image_processor(sample["image"])['pixel_values'][0]
        inputs = self.tokenizer(sample["text"], return_tensors="pt", padding="max_length",
                                max_length=self.config.max_length, truncation=True)
        return {"images": image, "input_ids": inputs["input_ids"].squeeze(), "labels": inputs["labels"].squeeze()}
