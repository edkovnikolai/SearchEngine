"""
Use the model to embed the tokens and save it to the final dataset
"""

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from datasets import load_from_disk, Dataset
from html_cleaner import clean_dataset_path
import torch
from tqdm.auto import tqdm

# where to save the database
embeddings_dataset_path = "stlawu-webpages-with-embeddings"

def cls_pooling(model_output):
    """
    Use cls pooling (embeddings from the [CLS] token) to get the token embedding for the whole text
    :param model_output:
    :return:
    """
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text: str):
    """
    Get the embeddings from the text
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
    return {'embeddings': cls_pooling(model_output)}

def batched_embeddings(rows):
    """
    Same as ``get_embeddings`` function, but used in batched mode for ``dataset.map()``
    :param rows:
    :return:
    """
    embeds = []
    for i in range(len(rows['html_doc'])):
        embeds.append(get_embeddings(rows['html_doc'][i])['embeddings'])
    return {'embeddings': embeds}




if __name__ == '__main__':

    # getting device to use for calculations
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)

    # Utilizing quantization due to limited GPU memory
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    # downloading and using the model
    model_ckpt = "intfloat/e5-mistral-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(
        model_ckpt,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # model tokens max length
    model.max_seq_length = 2 * 4096

    # loading model to device
    # NO NEED TO DO IT WITH bitsandbytes MODELS
    # model.to(device)



    clean_dataset = load_from_disk(clean_dataset_path)

    # FIXME remove this debug thing
    # clean_dataset = clean_dataset.select(range(64))

    # creating token embeddings
    embeddings = clean_dataset.map(
        batched_embeddings,
        batched=True,
        batch_size=64,
    )


    # Old version
    # embeddings = clean_dataset.map(
    #     lambda x: get_embeddings(x['html_doc'])
    # )

    # saving to the disk
    embeddings.save_to_disk(embeddings_dataset_path)

    # THE FOLLOWING CODE CRASHES. FOR NOW JUST DO EMBEDDINGS

    # adding faiss index
    # embeddings.add_faiss_index(column='embeddings')

    # saving to the disk
    # embeddings.save_to_disk(embeddings_dataset_path)



