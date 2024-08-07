from .vlm import *
from .api import GPT4V, GeminiProVision
from functools import partial

models = {
    'instructblip_13b': partial(InstructBLIP, name='/code/BaseModel/VLM/instructblip-vicuna-13b'),
}

api_models = {
    'GPT4V': partial(GPT4V, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    'GeminiProVision': partial(GeminiProVision, temperature=0, retry=10),
}

supported_VLM = {}
for model_set in [models, api_models]:
    supported_VLM.update(model_set)
