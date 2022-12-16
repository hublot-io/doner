from trdg.generators import GeneratorFromStrings
from datasets import load_dataset
from PIL import Image
import json 
import argparse
import numpy as np
import random

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--split", help="train | validation | test")
argParser.add_argument("-n", "--num", help="num of items")
argParser.add_argument("-t", "--type", help="Type of gt_parse: text_sequence | ner")
args = argParser.parse_args()

dataset = load_dataset("conll2003")
chunk_size = 20
dataset_split = args.split
iterator = dataset[dataset_split].__iter__()
img_height = 32
img_width = 2000
BILOU_MAP = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

def get_text_sequence(conll_item):
    return {"text_sequence": " ".join(conll_item['tokens'])}
    

def get_ner_sequence(conll_item):
    ziped = zip(conll_item['tokens'], conll_item['ner_tags'])
    res = []
    for (key,val) in ziped:
        m = {}
        m[key] = val
        res.append(m)
    return {"ner":res}

for i, c in enumerate(iterator):
    image_name = "{}/{}_{}.jpg".format(dataset_split,dataset_split, i)

    build_ner = []

    data = {
        "ground_truth": json.dumps({
            "gt_parse": get_ner_sequence(c) if args.type == 'ner' else get_text_sequence(c)
        }),
        "file_name": "{}_{}.jpg".format(dataset_split, i)
    }

    text = " ".join(c['tokens'])
    num_chunks = len(c['tokens']) if len(c['tokens']) <= 6 else random.randrange(1, 4)
    chunk = np.array_split(c['tokens'],num_chunks)
    chunk = [list(c) for c in chunk]
    strs = [" ".join(t) for t in chunk]
    generator = GeneratorFromStrings(
        strs,
        random_blur=True,
        random_skew=True,
        distorsion_type=3,
        background_type=1,
        text_color="#000000,#888888",
        count=len(strs)
    )
    imgs = [val[0] for val in generator]
    max_w = max([img.width for img in imgs])
    img_dest = Image.new('RGB', (max_w, num_chunks*img_height),(255, 255, 255))
    for ii, img in enumerate(imgs):
        img_dest.paste(img, (0, img_height * ii))
    
    with open('data/OCR/{}/metadata.jsonl'.format(dataset_split), 'a') as fw:
        fp = "data/OCR/{}".format(image_name)
        img_dest.save(fp)
        fw.write("{}\n".format(json.dumps(data)))
    if i >= int(args.num) :
        break 
