from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class StoryLoader:

    MODEL_NAMES = ["oztrkoguz/phi3_short_story_merged_bfloat16"]
    def __init__(self):
        self.model = None
        self.tokenizer = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (s.MODEL_NAMES, {"default": s.MODEL_NAMES[0]},),
            }   
        }
    
    RETURN_TYPES = ("CUSTOM","CUSTOM")
    RETURN_NAMES = ("model","tokenizer",)
    FUNCTION = "load_story_model"
    CATEGORY = "Story Nodes/Story Sampler Simple"
    
    def load_story_model(self, model:str):
        tokenizer_model = "unsloth/Phi-3-mini-4k-instruct"
        lora_model = "oztrkoguz/phi3_short_story_merged_bfloat16"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.model = AutoModelForCausalLM.from_pretrained(lora_model).to("cuda")
    
        return (self.model,self.tokenizer)




class StorySamplerSimple:

    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CUSTOM", {"default": ""}),
                "tokenizer" : ("CUSTOM", {"default": ""}),
                "prompt": ("STRING",{"forceInput": True} ),
                
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "generate_text"
    CATEGORY = "Story Nodes/Story Sampler Simple"
    

    def generate_text(self, prompt, model,tokenizer):
        
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        create a short story from this keywords

        ### Input:
        {}

        ### Response:
        {}"""

        # Use the merged model for inference
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                prompt,
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=150
            )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response_marker = "Response:"
        response_index = generated_text.find(response_marker)
        if response_index != -1:
            response_text = generated_text[response_index + len(response_marker):].strip()
            return (response_text,)
        else:
            return "No answer found."
   

NODE_CLASS_MAPPINGS = {

    "StoryLoader": StoryLoader,
    "StorySamplerSimple" : StorySamplerSimple

}

NODE_DISPLAY_NAME_MAPPINGS = {

    "StoryLoader": "Story Loader",
    "StorySamplerSimple" : "Story Sampler Simple"

}
