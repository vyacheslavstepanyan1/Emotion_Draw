import os
current_dir = os.getcwd()
auth_token_dir = os.path.join(current_dir, "Emotion_Draw/api/auth_token.txt")

with open(auth_token_dir,'r') as f:
  auth_token = f.read()
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64 
from ..bert_part.inference.inference import Inference
app = FastAPI()

model_checkpoint = os.path.join(current_dir, "Emotion_Draw/bert_part/models_trained/multiclass_experiment3_albert-base-v2_best_checkpoint.pth")

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device = 'cuda'
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token= auth_token)
pipe.to(device)

@app.get("/")
def generate(prompt: str): 

    sentences = []
    inference = Inference(model_checkpoint, prompt)
    for i in inference[0]:
      confidence = (i["score"]/1) * 100
      sentence = f"We are {confidence:.2f}% confident you are feeling {i['label']}."
      sentences.append(sentence)

    emotion1 = inference[0][0]['label']
    emotion2 = inference[0][1]['label']
    emotion3 = inference[0][2]['label']

    prompt1 = f"""ENERGY ART STYLE representation of the feeling of {emotion1}. colors that best match {emotion1}. 3-4 colors used. Pastel tones. Waterpaint. Transition between colors very smooth."""
    prompt2 = f"""ENERGY ART STYLE representation of the feeling of {emotion2}. colors that best match {emotion2}. 3-4 colors used. Pastel tones. Waterpaint. Transition between colors very smooth."""
    prompt3 = f"""ENERGY ART STYLE representation of the feeling of {emotion2}. colors that best match {emotion3}. 3-4 colors used. Pastel tones. Waterpaint. Transition between colors very smooth."""

    with autocast(device): 
        image1 = pipe(prompt1, guidance_scale=8.5).images[0]
    
    with autocast(device): 
        image2 = pipe(prompt2, guidance_scale=8.5).images[0]
    
    with autocast(device): 
        image3 = pipe(prompt3, guidance_scale=8.5).images[0]

    image_dir = os.path.join(current_dir, "Emotion_Draw/api")

    image1.save(f"{image_dir}/image1.png")
    image2.save(f"{image_dir}/image2.png")
    image3.save(f"{image_dir}/image3.png")

    buffer1 = BytesIO()
    buffer2 = BytesIO()
    buffer3 = BytesIO()

    image1.save(buffer1, format="PNG")
    image2.save(buffer2, format="PNG")
    image3.save(buffer3, format="PNG")

    imgstr1 = base64.b64encode(buffer1.getvalue())
    imgstr2 = base64.b64encode(buffer2.getvalue())
    imgstr3 = base64.b64encode(buffer3.getvalue())

    response1 = Response(content=imgstr1, media_type="image/png")
    response2 = Response(content=imgstr2, media_type="image/png")
    response3 = Response(content=imgstr3, media_type="image/png")

    return sentences[0], sentences[1], sentences[2], response1, response2, response3