from datasets import load_dataset
dataset =  load_dataset("imagefolder", data_dir="./data/fish-label")
dataset.push_to_hub("hublot/fish-label", private=True)