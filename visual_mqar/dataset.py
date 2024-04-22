from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPImageProcessor


class VisualMQARDataset(Dataset):
    def __init__(self, dataset, tokenizer=None, image_processor=None, config=None):
        self.dataset = dataset
        self.tokenizer = tokenizer if config is None \
            else AutoTokenizer.from_pretrained(config.text_decoder)
        self.image_processor = image_processor if config is None \
            else CLIPImageProcessor.from_pretrained(config.vision_encoder)
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        if self.image_processor is not None:
            image = self.image_processor()['pixel_values'][0]
        inputs = '\n'.join([v['value'] for v in sample['conversations']])
        if self.tokenizer is not None:
            inputs = self.tokenizer(inputs,
                                    return_tensors="pt", padding="max_length",
                                    max_length=self.config.max_length, truncation=True)
        return dict(
            images=image,
            **
            dict(
                input_ids=inputs["input_ids"].squeeze(),
                labels=inputs["labels"].squeeze()
            )
            if inputs is dict
            else dict(
                input_ids=inputs,
            )
        )


