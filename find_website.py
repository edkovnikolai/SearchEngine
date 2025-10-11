"""
Find the needed website.
"""
from torch import device

from create_embeddings import embeddings_dataset_path, get_embeddings
from datasets import load_from_disk
from model import get_model, get_tokenizer, get_device
import numpy as np

if __name__ == "__main__":
    # loading dataset from disk
    embeddings_dataset = load_from_disk(embeddings_dataset_path)
    embeddings_dataset.add_faiss_index(column="embeddings")

    model = get_model()
    tokenizer = get_tokenizer()
    device = get_device()

    while True:
        try:
            text = input()
        except EOFError:
            break
        search_embeddings = np.array([get_embeddings(text, model, tokenizer, device)['embeddings'].cpu().detach().numpy()])

        print(search_embeddings)
        print(search_embeddings.shape)

        print(embeddings_dataset[0]['embeddings'])

        # get the nearest samples
        num_of_samples = 5
        scores, samples = embeddings_dataset.get_nearest_examples(
            "embeddings",
            search_embeddings,
            k=num_of_samples
        )

        print(f"Request: {text}")
        for i in range(num_of_samples):
            print(f"Score: {scores[i]:.3f}, url: {samples['url'][i]}")




