"""LoRA on MASt3R attention; differentiable, checkpointed training without C++ BA gradients."""
import math,types
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class LoRALinear(nn.Module):
 def __init__(self,base,rank=4,alpha=4):
  super().__init__();self.base=base;self.scale=alpha/rank
  self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device));self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device))
  nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
 def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.lora_A),self.lora_B)*self.scale

def install_lora(model,rank=4,alpha=4):
 for p in model.parameters():p.requires_grad_(False)
 names=[]
 for name,module in list(model.named_modules()):
  if isinstance(module,nn.Linear) and name.startswith(('enc_blocks.','dec_blocks.','dec_blocks2.')) and any(k in name for k in ('.attn.','.cross_attn.')):
   parent_path,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent_path),leaf,LoRALinear(module,rank,alpha));names.append(name)
 return names
def lora_state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items() if '.lora_' in k}
def load_lora(model,state):
 result=model.load_state_dict(state,strict=False);assert not result.unexpected_keys,result.unexpected_keys
 missing=[k for k in result.missing_keys if '.lora_' in k];assert not missing,missing
def merge_lora(model):
 for name,module in list(model.named_modules()):
  if isinstance(module,LoRALinear):
   with torch.no_grad():module.base.weight.add_((module.lora_B@module.lora_A).to(module.base.weight.dtype)*module.scale)
   parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,module.base)

class SafeRoPE(nn.Module):
 """Same split-half rotation as upstream fallback, avoiding in-place autograd mutations."""
 def __init__(self,base=100.,f0=1.):super().__init__();self.base=base;self.f0=f0
 def forward(self,x,positions):
  d=x.shape[-1]//2;inv=self.f0/self.base**(torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d)
  parts=[]
  for value,pos in zip(x.chunk(2,dim=-1),(positions[:,:,0],positions[:,:,1])):
   angles=pos.float()[:,:,None]*inv;angles=torch.cat((angles,angles),dim=-1)[:,None].to(x.dtype)
   a,b=value.chunk(2,dim=-1);parts.append(value*angles.cos()+torch.cat((-b,a),dim=-1)*angles.sin())
  return torch.cat(parts,dim=-1)
def self_attention(self,x,xpos):
 b,n,c=x.shape;q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
 if self.rope is not None:q=self.rope(q,xpos);k=self.rope(k,xpos)
 y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
 return self.proj_drop(self.proj(y.transpose(1,2).reshape(b,n,c)))
def cross_attention(self,query,key,value,qpos,kpos):
 b,n,c=query.shape
 def split(x):return x.reshape(b,-1,self.num_heads,c//self.num_heads).transpose(1,2)
 q,k,v=split(self.projq(query)),split(self.projk(key)),split(self.projv(value))
 if self.rope is not None:q=self.rope(q,qpos);k=self.rope(k,kpos)
 y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
 return self.proj_drop(self.proj(y.transpose(1,2).reshape(b,n,c)))
def prepare_training(model):
 for module in list(model.modules()):
  if hasattr(module,'rope') and module.rope is not None:
   old=module.rope;module.rope=SafeRoPE(old.base,old.F0)
  if module.__class__.__name__=='Attention':module.forward=types.MethodType(self_attention,module)
  if module.__class__.__name__=='CrossAttention':module.forward=types.MethodType(cross_attention,module)
 for group in (model.enc_blocks,model.dec_blocks,model.dec_blocks2):
  for block in group:
   original=block.forward
   def wrapped(*args,_original=original):
    return checkpoint(_original,*args,use_reentrant=False) if torch.is_grad_enabled() else _original(*args)
   block.forward=wrapped
 return model

def make_view(rgb,index=0):
 h,w=rgb.shape[-2:]
 return {'img':rgb*2-1,'true_shape':torch.tensor([[h,w]],dtype=torch.int32),'idx':[index],'instance':[str(index)]}
