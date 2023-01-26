from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from torch.utils.mobile_optimizer import optimize_for_mobile
import json
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

processor = DonutProcessor.from_pretrained("hublot/doner", use_auth_token=True, torch_dtype=torch.float16, torchscript=True)
model = VisionEncoderDecoderModel.from_pretrained("hublot/doner", use_auth_token=True, torch_dtype=torch.float16, torchscript=True)
print("loaded models")

model.to(device)
model = model.float()
model.eval()

# model = torch.quantization.convert()
im = Image.open("test.jpg")
print("opened test image")
task_prompt = "<s_ner>"
pixel_values = processor(im, return_tensors="pt").pixel_values
tokenized  = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")
def get_swap_dict(d):
    return {v: k for k, v in d.items()}
vocab = processor.tokenizer.get_vocab()
swaped = get_swap_dict(vocab)
f = open("vocab.txt", "a")
json_object = json.dumps(swaped, indent=4)

f.write(json_object)
decoder_input_ids = tokenized.input_ids
print("got intputs")
pixel_values = pixel_values.to(device)
decoder_input_ids = decoder_input_ids.to(device)
outputs = model.generate(
        pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

print("Quantize model")
model = torch.quantization.convert(model)
print("tracing model")
traced_model = torch.jit.trace(model, [pixel_values, decoder_input_ids])

print("Scripting processor")
traced_proc = torch.jit.trace(processor, im)
traced_script_module_optimized = optimize_for_mobile(traced_model)
print("scripted")
print(traced_script_module_optimized)

print(dir(traced_script_module_optimized))
traced_script_module_optimized._save_for_lite_interpreter("mobile.ptl")