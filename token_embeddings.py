"""
Embed tokens in the dataset
"""
# path to raw dataset
raw_dataset_path = './stlawu-webpages.jsonl'

from datasets import load_dataset
import pandas as pd

raw_dataset = load_dataset('json', data_files=raw_dataset_path)
print(raw_dataset)

# df = pd.read_json(raw_dataset_path, lines=True)

# print(df)
