import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration, Gemma3Model, QuantoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import laion_clap
import numpy as np
import huggingface_hub
import os

# --- 1. Define the Embedding Projector ---
class EmbeddingProjector(nn.Module):
    def __init__(self, input_embedding_dim, llm_embedding_dim, num_virtual_tokens=10, hidden_dim=512):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.llm_embedding_dim = llm_embedding_dim

        # Project the input_embedding_dim into (num_virtual_tokens * llm_embedding_dim)
        # Reshape later to (num_virtual_tokens, llm_embedding_dim)
        self.projector = nn.Linear(input_embedding_dim, llm_embedding_dim * num_virtual_tokens )
            # nn.GELU(), # Or ReLU, GELU etc.
            # nn.Linear(hidden_dim, llm_embedding_dim * num_virtual_tokens)

    def forward(self, embedding):
        # embedding shape: (batch_size, input_embedding_dim)
        projected_flat = self.projector(embedding)
        # Reshape to (batch_size, num_virtual_tokens, llm_embedding_dim)
        projected_virtual_tokens = projected_flat.view(
            # embedding.size(0), self.num_virtual_tokens, self.llm_embedding_dim
            -1, self.llm_embedding_dim
        )
        return projected_virtual_tokens

class MusicGemma( Gemma3Model ):
  def __init__(self, *args, **kwargs):
    super().__init__( *args, **kwargs )
    self.music_model = None
    self.vision_tower = None
    self.multi_modal_projector = None
    self.music_model = None
    self.use_music_model = False

    input_embedding_dim = 512 # Example: BERT base embedding dimension
    num_virtual_tokens = 256 # How many virtual tokens to create from the embedding
    llm_embedding_dim = 2560

    self.embedding_projector = EmbeddingProjector(input_embedding_dim, llm_embedding_dim, num_virtual_tokens)


  def use_music_model( self, use_music_model ):
    self.use_music_model = use_music_model
    if self.use_music_model and self.music_model is None:
      self.load_music_model()

  def load_music_model( self ):
      self.music_model = laion_clap.CLAP_Module( enable_fusion=False, amodel="HTSAT-base")
      if not os.path.exists( "./textmodel/music_audioset_epoch_15_esc_90.14.pt" ):
        print( "Downloading music model weights..." )
        huggingface_hub.hf_hub_download(
            repo_id="lukewys/laion_clap", 
            filename="music_audioset_epoch_15_esc_90.14.pt",
            local_dir="./textmodel"
        )
      self.music_model.load_ckpt( "./textmodel/music_audioset_epoch_15_esc_90.14.pt" )
      self.music_model.to( self.device )
      self.music_model.to( self.dtype )
      self.music_model.eval() # This might be wrong

  def get_image_features( self, pixel_values ):
    input = pixel_values
    if self.use_music_model:
      input = self.music_model.get_audio_embedding_from_data( x=pixel_values, use_tensor=True )
    
    output = self.embedding_projector( input )
    return output.to( self.dtype )

class MusicGemmaForConditionalGeneration( Gemma3ForConditionalGeneration ):
  def __init__(self, *args, **kwargs):
    super().__init__( *args, **kwargs )
    self.model = MusicGemma( *args, **kwargs )

  def use_music_model( self, use_music_model ):
    self.model.use_music_model( use_music_model )
                            

def get_model_and_such( device_map="mps" ):
    quantization_config = QuantoConfig( weights_dtype="int4" )

    model = MusicGemmaForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        device_map=device_map,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
    "google/gemma-3-4b-it",
    padding_side="right"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-4b-it",
        padding_side="right"
    )
    input_dir = "./textmodel/text_pretrained_3"
      
    model = PeftModel.from_pretrained(model, input_dir, is_trainable=False)
    model.model.model.load_music_model()
    return model, processor, tokenizer
  