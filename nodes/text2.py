class Write2:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
                "input_text": ("STRING",{"multiline": True,"default": "",},),
            },
        }
    RETURN_TYPES = ("STRING",)

    FUNCTION = "simple_text"

    CATEGORY = "Story Nodes/Write2"

    def simple_text(self, input_text):

        return (input_text, )

NODE_CLASS_MAPPINGS = {
    "Write2": Write2
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Write2": "Simple Text2"
}