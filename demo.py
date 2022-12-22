import gradio as gr
from donut import DonutModel, JSONParseEvaluator, load_json, save_json
import torchvision.transforms as transforms
transform = transforms.ToTensor()
model = DonutModel.from_pretrained('result/train_ner/test_experiment_dict',torchscript=True)
model.to('cuda:0')
model.half()

def greet(image):
    output = model.inference(image=image, prompt=f"<s_ner>", return_json=True)["predictions"][0]
    return output

demo = gr.Interface(fn=greet, inputs=gr.Image(type="pil"), outputs="json")

demo.launch(share=True)   