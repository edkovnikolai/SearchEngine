"""
Use the model to embed the tokens and save it to the final dataset
"""

from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk, Dataset
from html_cleaner import clean_dataset_path

# where to save the database
embeddings_dataset_path = "stlawu-webpages-embeddings"

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
    # TODO: Later do it for the GPU
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

def batched_embeddings(rows):
    """
    Same as ``get_embeddings`` function, but used in batched mode for ``dataset.map()``
    :param rows:
    :return:
    """
    embeds = []
    for i in range(len(rows)):
        embeds.append(get_embeddings(rows['html_doc'][i]))
    return {'embeddings': embeds}




if __name__ == '__main__':

    # downloading and using the model
    model_ckpt = "intfloat/e5-mistral-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    # model tokens max length
    model.max_seq_length = 4096

    clean_dataset = load_from_disk(clean_dataset_path)

    # creating token embeddings
    # TODO Do it on GPU
    # embeddings = clean_dataset.map(
    #     batched_embeddings,
    #     batched=True,
    #     batch_size=10
    # )

    embeddings = clean_dataset.map(
        lambda x: get_embeddings(x['html_doc']),
        num_proc=10
    )

    # adding faiss index
    embeddings.add_faiss_index(column='embeddings')

    # saving to the disk
    embeddings.save_to_disk(embeddings_dataset_path)



