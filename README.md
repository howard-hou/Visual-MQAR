# Visual-MQAR
Understand and test multi-modal language model architectures on Visual-MQAR task.

## Introduction

This repository contains the code for the Visual-MQAR task. The task is a benchmark for evaluating the performance of multi-modal language models on visual inputs.

The main task is such: given a text prompt and a set of images, the model should generate a natural language response that is consistent with the text prompt and the images. The model should be able to extract relevant information from both the text and the images and generate a coherent and fluent response.

## Dataset and Preprocessing

You can build up the dataset by following the steps below:

```bash
cd tests
python test_dataset_generation.py
```

This will generate the dataset in the `tests/data` directory.

It is based on the openwikitext dataset, which is a large-scale multilingual dataset for text generation. We use the English version of the dataset for this task.


## Training

To train a multi-modal language model on the Visual-MQAR task, follow the steps below:

1. Clone the repository:

```bash
git clone https://github.com/xforcevesa/Visual-MQAR.git
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Test the training script:

```bash
python tests/test_model_train.py
```


