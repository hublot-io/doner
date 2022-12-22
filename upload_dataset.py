from datasets import load_dataset
dataset =  load_dataset("imagefolder", data_dir="./data/OCR")
dataset.push_to_hub("hublot/fish-label", private=True)