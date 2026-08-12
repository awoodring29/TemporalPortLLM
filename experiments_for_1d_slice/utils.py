import torch# type:ignore
import torch.nn as nn# type:ignore
import time
import copy
import math

# def r_v(dL,theta,v,device): # no longer needed
#     w=torch.zeros(dL.size(),requires_grad=True,device=device)
#     g=torch.autograd.grad(dL,theta,grad_outputs=w,create_graph=True,allow_unused=True)
#     # print(f"g:{g}")
#     r=torch.autograd.grad(to_vector(g),w,grad_outputs=v,create_graph=False,allow_unused=True)

#     return r[0]

def to_vector(g):
    g=list(g)
    out=[]
    while len(g)>0:
        out.append(g.pop(0).flatten())
        out=[torch.concat(out)]
    return out[0]

def vg_lora(net,inputs,device,v,print_memory=False,num_rand_to_sim=0,rand_norm=1): # just gradient vector products
    if print_memory:
        print(f"Memory reserved before net.zero_grad(): {torch.cuda.memory_reserved()/(1024**3)} GB.")
    net.zero_grad()
    if print_memory:
        print(f"Memory reserved after net.zero_grad(): {torch.cuda.memory_reserved()/(1024**3)} GB.")
    net_out=net(**inputs)
    if print_memory:
        print(f"Memory reserved after forward pass: {torch.cuda.memory_reserved()/(1024**3)} GB.")
    L=net_out.loss
    parameters=[param for name,param in net.named_parameters() if "lora_" in name]
    if print_memory:
        print(f"Memory reserved after getting params: {torch.cuda.memory_reserved()/(1024**3)} GB.")
    for name,param in net.named_parameters():
        if "lora_" in name and param.requires_grad==False:
            print(f"param {name} does not require grad")

    y=to_vector(torch.autograd.grad(L,parameters)).to(dtype=torch.float16,device=device)
    v=to_vector(v).to(dtype=torch.float16,device=device)
    vg=torch.dot(v,y)
    grad_norm=torch.norm(y)
    loss=L.detach()
    if num_rand_to_sim==0:
        return vg, loss, grad_norm
    rand_sim=[]
    for i in range(num_rand_to_sim):
        r=torch.randn(v.shape,device=device,dtype=torch.float16)
        r=r/torch.norm(r)
        r=r*rand_norm
        rand_sim.append(float(torch.dot(r,y)))
    return vg, sum(rand_sim)/len(rand_sim)