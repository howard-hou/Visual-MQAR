import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import CLIPVisionModel, AutoModel


class VisualMQAR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.vision_encoder)
        self.text_decoder = AutoModel.from_pretrained(config.text_decoder)
        self.vision_projector =  nn.Linear(config.vision_encoder_hidden_size, config.text_decoder_hidden_size, bias=False)
        self.lm_head = nn.Linear(config.text_decoder_hidden_size, config.vocab_size, bias=False)
        self.vision_encoder.requires_grad_(False)
        
    def encode_images(self, images):
        image_features = self.vision_encoder(images).last_hidden_state[:, 1:]
        image_features = self.vision_projector(image_features)
        return image_features
    
    def get_input_embeddings(self, input_ids):
        return self.text_decoder.get_input_embeddings()(input_ids)
    
    def preparing_embedding(self, samples):
        image_features  = self.encode_images(samples["images"])
        # prepare text embedding
        text_embeds = self.get_input_embeddings(samples["input_ids"])
        print(image_features.shape, text_embeds.shape)
        input_embeds = torch.cat([image_features, text_embeds], dim=1)
        return input_embeds, samples["labels"]

    def forward(self, samples):
        input_embeds, targets = self.preparing_embedding(samples)
        last_hidden_state = self.text_decoder(inputs_embeds = input_embeds).last_hidden_state
        logits = self.lm_head(last_hidden_state)
        return logits, targets
    
    def training_step(self, batch, batch_idx):
        logits, targets = self(batch)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
        return loss

if __name__ == "__main__":
    from config import Config
    config = Config()
    model = VisualMQAR(config)
    # generate random data
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 50256, (2, 32))
    labels = torch.randint(0, 50256, (2, 32))
    samples = {"images": images, "input_ids": input_ids, "labels": labels}
    logits, targets = model(samples)