"""
Model used
"""

import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModel

model_checkpoint = "intfloat/e5-mistral-7b-instruct"

def get_model():
    """
    Get the model used throughout the project.
    Note that model is automatically is on CUDA due to BitsAndBytesConfig
    :return:
    """


    # Utilizing quantization due to limited GPU memory
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    # downloading and using the model
    model = AutoModel.from_pretrained(
        model_checkpoint,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # model tokens max length
    model.max_seq_length = 2 * 4096

    return model

def get_tokenizer():
    """
    Getting tokenizer for the model with the specified model_checkpoint
    :return:
    """
    return AutoTokenizer.from_pretrained(model_checkpoint)

def get_device(name=None):
    """
    Getting device
    :return:
    """
    if name is None:
        name = 'cuda'

    # getting device to use for calculations
    if torch.cuda.is_available() and name=='cuda':
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device