from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import numpy as np
import gc
import torch
from comfy_extras.nodes_mask import MaskComposite
from folder_paths import models_dir, folder_names_and_paths, add_model_folder_path, get_folder_paths, get_filename_list, get_full_path
import os
import cv2

kosmos2_dir = "kosmos2"
huggingface_name = "microsoft/"
kosmos2_model_path = f"{models_dir}/{kosmos2_dir}"

try:
    if kosmos2_model_path not in get_folder_paths(kosmos2_dir):
        raise KeyError
except KeyError:
    add_model_folder_path(kosmos2_dir, kosmos2_model_path)

class KosmosLoader2:
    MODEL_NAMES = ["microsoft/kosmos-2-patch14-224"]
    DEVICES = ["cpu", "gpu"] if torch.cuda.is_available() else  ["cpu"]

    def __init__(self):
        self.model = None
        self.processor = None
        self.modelname = ""
        self.device = ""
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (s.MODEL_NAMES, {"default": s.MODEL_NAMES[0]},),
                "device": (s.DEVICES, {"default": s.DEVICES[0]},),
            }   
        }
    
    RETURN_TYPES = ("CUSTOM","CUSTOM",)
    RETURN_NAMES = ("model","processor",)
    FUNCTION = "load_kosmos_model"
    CATEGORY = "Story Nodes/Kosmos2 Sampler Simple2"
    
    def load_kosmos_model(self, model:str, device:str):
    
        dev = "cuda" if device.lower() == "gpu" else "cpu"
        model = model.replace('microsoft/', '')
        model_paths = get_folder_paths(kosmos2_dir)

        def model_in_path() -> str | None:
            for p in model_paths:
                result = f"{p}/{model}"
                if os.path.isdir(result):
                    return result
            return None
        model_path = model_in_path()

        if not model_path:
            model_path = f"{huggingface_name}{model}"

        if (self.model == None) or (self.processor == None) or (self.modelname != model) or (device != self.device):
            del self.model
            del self.processor
            gc.collect()
            if (device == "cpu") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"kosmos2: loading model {model_path}, please stand by....")
            self.model = AutoModelForVision2Seq.from_pretrained(model_path).to(dev)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.modelname = model
            self.device = device
    
        return (self.model,self.processor,)




class Kosmos2SamplerSimple2:
    def __init__(self):
        self.prefix = "<grounding> "
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("CUSTOM", {"default": ""}),
                "processor" : ("CUSTOM", {"default": ""}),
                "prompt": ("STRING",{"forceInput": True} ),
                "strip_prompt": ("BOOLEAN", {"default": True},),
                
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING","IMAGE",)
    RETURN_NAMES = ("description", "keyword","image")
    FUNCTION = "generate_text"
    CATEGORY = "Story Nodes/Kosmos2 Sampler Simple2"
    
    def generate_text(self, image:torch.Tensor, prompt:str,strip_prompt:bool,model,processor):
        descriptions = ""
        entity_str = ""
        entity_str2 = ""
        width = round(image.shape[2])
        height = round(image.shape[1])
        mask = torch.full((1, height, width), 0., dtype=torch.float32, device="cpu")
        image_copy = image.numpy()
        
        for im in image:
            i = 255. * im.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            prompt_full = self.prefix + prompt

            inputs = processor(text=prompt_full, images=img, return_tensors="pt").to("cuda")
            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if strip_prompt == True:
                generated_text = generated_text.replace(prompt_full, '').strip()

            description, entities = processor.post_process_generation(generated_text)
            descriptions += description + '\n'

            elist = []
            for entity_name, (start, end), bboxx in entities:
                bbx = bboxx[0]
                x = int(bbx[0] * width)
                y = int(bbx[1] * height)
                w = int(bbx[2] * width)
                h = int(bbx[3] * height)
                print(f"kosmos-2 entity '{entity_name}' at {x}, {y}, {w}, {h}")
                m = torch.full((1, h, w), 1., dtype=torch.float32, device="cpu")
                mask = MaskComposite.combine(self, mask, m, x, y, "or")[0]
                elist.append([entity_name,x,y,w,h])
            entity_str += ",".join([f"{entity_name},{x},{y},{w},{h}" for entity_name, x, y, w, h in elist])
            parts = entity_str.split(",")
            entity_str2 += ",".join([f"{entity_name}" for entity_name, x, y, w, h in elist])
            parts = entity_str.split(",")
            
            white_image = np.zeros((height, width, 3), dtype=np.float32)
            
            coordinates = []
            image_np = np.copy(image_copy) 
            image_np2 = image.numpy()
            image_np2 = np.squeeze(image_np2)
            entities_dict = {}
            colors = [(255, 0, 0), (255, 255, 0), (0, 0, 255), (0, 150, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)]
            i = 0
            while i < len(parts):
                entity_name = parts[i]
                x1 = int(parts[i+1])
                y1 = int(parts[i+2])
                x2 = int(parts[i+3])
                y2 = int(parts[i+4])

                entities_dict[entity_name] = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
                i += 5
                
            for i, (cls, coord) in zip(range(len(entities_dict)), entities_dict.items()):
                cv2.rectangle(image_np2, (int(coord["x1"]),int(coord["y1"])), (int(coord["x2"]),int(coord["y2"])), colors[i], 2)
                cv2.putText(image_np2, cls, (int(coord["x1"]),int(coord["y1"])-7), cv2.FONT_HERSHEY_COMPLEX_SMALL ,0.7, colors[i], 1, cv2.LINE_AA) 

            torch_tensor2 = torch.from_numpy(image_np2)
            torch_tensor2 = torch_tensor2.unsqueeze(0)
        return (descriptions, entity_str2 ,torch_tensor2,)         
        



NODE_CLASS_MAPPINGS = {

    "KosmosLoader2": KosmosLoader2,
    "Kosmos2SamplerSimple2" : Kosmos2SamplerSimple2

}

NODE_DISPLAY_NAME_MAPPINGS = {

    "KosmosLoader2": "Kosmos2 Loader2",
    "Kosmos2SamplerSimple2" : "Kosmos2 Sampler Simple2"

}
