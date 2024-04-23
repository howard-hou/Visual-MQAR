from dataclasses import dataclass


@dataclass
class Config:
    vision_encoder: str = "openai/clip-vit-base-patch32"
    text_decoder: str = "openai-community/gpt2"
    vision_encoder_hidden_size: int = 768
    text_decoder_hidden_size: int = 768
    vocab_size: int = 50256
    max_length: int = 975
