import transformers
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPImageProcessor
import torch


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = torch.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible minus values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids < 0, pad_token_id)

    return shifted_input_ids


class VisualMQARDataset(Dataset):
    def __init__(self, dataset, tokenizer=None, image_processor=None, config=None):
        self.dataset = dataset
        self.tokenizer = tokenizer if config is None \
            else AutoTokenizer.from_pretrained(config.text_decoder)
        self.image_processor = image_processor if config is None \
            else CLIPImageProcessor.from_pretrained(config.vision_encoder)
        self.config = config
        self.iter_index = 0

        self.tokenizer.pad_token_id = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        image = self.image_processor(image)['pixel_values'][0]
        image = torch.from_numpy(image).unsqueeze(0)
        inputs = '\n'.join([v['value'] for v in sample['conversations']])
        inputs = self.tokenizer(inputs,
                                return_tensors="pt", padding="max_length",
                                max_length=self.config.max_length, truncation=True)
        return dict(
            images=image,
            input_ids=inputs["input_ids"],
            labels=shift_tokens_right(
                input_ids=inputs["input_ids"],
                pad_token_id=self.tokenizer.eos_token_id,
                decoder_start_token_id=self.tokenizer.bos_token_id
            )
        )

    def __next__(self):
        data = self[self.iter_index]
        self.iter_index += 1
        return data

    def __iter__(self):
        self.iter_index = 0
        return self
