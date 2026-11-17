"""Load native Qwen3-4B plus the full layer-29 state, with no hooks or edits."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(device='cuda'):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from safetensors.torch import load_file
 tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(ROOT/'model',torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,trust_remote_code=False).to(device).eval()
 state=load_file(str(ROOT/'molts/layer-29/checkpoint.safetensors'),device=device)
 config=json.loads((ROOT/'molts/layer-29/config.json').read_text())
 return model,tokenizer,state,config
