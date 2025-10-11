"""
Use the model to embed the tokens and save it to the final dataset
"""

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from datasets import load_from_disk
from html_cleaner import clean_dataset_path
from model import get_model, get_tokenizer, get_device
import torch

# where to save the database
embeddings_dataset_path = "stlawu-webpages-with-embeddings"

def cls_pooling(model_output):
    """
    Use cls pooling (embeddings from the [CLS] token) to get the token embedding for the whole text
    :param model_output:
    :return:
    """
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text: str, model, tokenizer, device):
    """
    Get the embeddings from the text
    :param device:
    :param tokenizer:
    :param model:
    :param text:
    :return:
    """
    encoded_input = tokenizer(
        text, padding='max_length', truncation=True, return_tensors="pt", max_length=model.max_seq_length
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # DEBUG:
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # disabling gradient calculations
    with torch.no_grad():
        model_output = model(**encoded_input)
    return {'embeddings': cls_pooling(model_output)[0]}

def batched_embeddings(rows, model, tokenizer, device):
    """
    Same as ``get_embeddings`` function, but used in batched mode for ``dataset.map()``
    :param device:
    :param tokenizer:
    :param model:
    :param rows:
    :return:
    """
    embeds = []
    for i in range(len(rows['html_doc'])):
        embeds.append(get_embeddings(rows['html_doc'][i], model, tokenizer, device)['embeddings'])
    return {'embeddings': embeds}




if __name__ == '__main__':



    print(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)

    # downloading and using the model
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_device()

    # loading model to device
    # NO NEED TO DO IT WITH bitsandbytes MODELS
    # model.to(device)



    clean_dataset = load_from_disk(clean_dataset_path)


    # creating token embeddings
    embeddings = clean_dataset.map(
        batched_embeddings,
        batched=True,
        batch_size=64,
        fn_kwargs={
            'model': model,
            'tokenizer': tokenizer,
            'device': device
        }
    )


    # Old version
    # embeddings = clean_dataset.map(
    #     lambda x: get_embeddings(x['html_doc']),
    #     fn_kwargs={
    #         'model': model,
    #         'tokenizer': tokenizer,
    #         'device': device
    #     }
    # )

    # saving to the disk
    embeddings.save_to_disk(embeddings_dataset_path)




