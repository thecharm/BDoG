import torch
from PIL import Image
from abc import abstractproperty
import os.path as osp
import random
import os, sys
from ..smp import *
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


class InstructBLIP:

    INSTALL_REQ = True

    def __init__(self, name):
        
        model = InstructBlipForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, device_map='cuda').eval()
        processor = InstructBlipProcessor.from_pretrained(name)
        self.processors = processor
        self.model = model

    def generate(self, prompt, image_path, dataset=None, max_length=100):
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processors(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
        try:
            outputs = self.model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=5,
                        max_length=max_length,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1)
            generated_text = self.processors.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        except Exception as e:
            generated_text = random.choice(['A','B','C','D','E'])
            print("Prompt_Error: ", e)
            

        return generated_text
