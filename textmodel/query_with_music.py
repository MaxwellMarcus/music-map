import numpy as np
from PIL import Image
import torch
import os

def query_with_music( model, processor, query: str, music_data: np.ndarray, conversation_history: list[dict]=[], max_new_tokens: int=16 ) -> list[dict]:
    set_lora_scaling( model, 8 )

    blank_image = Image.new( "RGB", (856, 856) )
    chat = [
        # { "role": "system", "content": "You are a helpful assistant that can answer questions about the music data." },
    ]
    if conversation_history:
        chat += conversation_history
    chat += [
                { "role": "user",
                    "content": [
                    { "type": "image", "image": blank_image },
                        { "type": "text", "text": query }
                    ] 
                },
            ]
    prompt = processor.apply_chat_template( chat, add_generation_prompt=True, tokenize=True,
                          return_dict=True, return_tensors="pt"
                    ).to( model.device )
    prompt[ "pixel_values" ] = torch.from_numpy( music_data  ).to( torch.float32 ).to( model.device )
    
    output = model.generate(**prompt, max_new_tokens=max_new_tokens)
    text_output = processor.decode( output[0], skip_special_tokens=True )
    return text_output

def query( model, processor, query: str="", conversation_history: list[dict]=[], max_new_tokens: int=16 ) -> list[dict]:
    set_lora_scaling( model, 0 )

    chat = [
        # { "role": "system", "content": "You are a helpful assistant that can answer questions about the music data." },
    ]
    if conversation_history:
        chat += conversation_history
    if query:
        chat += [
                    { "role": "user",
                        "content": [
                            { "type": "text", "text": query }
                        ] 
                    },
                ]
    prompt = processor.apply_chat_template( chat, add_generation_prompt=True, tokenize=True,
                          return_dict=True, return_tensors="pt"
                    ).to( model.device )
    
    output = model.generate(**prompt, max_new_tokens=max_new_tokens)
    text_output = processor.decode( output[0], skip_special_tokens=True )
    return text_output

def set_lora_scaling(model, scale):
    """
    Dynamically sets the scaling factor for all LoRA layers in a PEFT model.

    Args:
        model (PeftModel): The PEFT model with LoRA adapters.
        scale (float): The desired scaling factor. A value of 1.0 represents
                       the original trained scaling.
    """
    for module in model.modules():
        if "lora.layer.Linear" in str(type(module)):
            # print( module.scaling )
            module.scaling = { "default": scale }