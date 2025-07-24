from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import random
import torch
import torch.fft
import torchvision.transforms as T


def generate_gaussian_noise(shape, mean=0, std_dev=0.1, seed=0):

    if seed is not None:
        torch.manual_seed(seed)
        
    noise = torch.randn(shape) * std_dev + mean
    
    return noise    

class Dict_enhaced(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.alpha = nn.Parameter(torch.ones(512) * 1.0)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=args.nhead, norm_first=True, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)        
        
    def forward(self, query, dict_feats, val):

        txt_rich = self.transformer_decoder(query, dict_feats)
 
        return txt_rich

class DGCMIA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        
        self.register_buffer('img_memory', torch.rand(self.args.ndata, 512).half())
        self.register_buffer('txt_memory', torch.rand(self.args.ndata, 512).half())
                                   
        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    
    def cross_former1(self, x):

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :]
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    

    def build_transforms(img_size=(384, 128), aug=False, is_train=True):
        height, width = img_size

        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        if not is_train:
            transform = T.Compose([
                T.Resize((height, width)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            return transform

        # transform for training
        if aug:
            transform = T.Compose([
                T.Resize((height, width)),
                T.RandomHorizontalFlip(0.5),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                T.RandomErasing(scale=(0.02, 0.4), value=mean),
            ])
        else:
            transform = T.Compose([
                T.Resize((height, width)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        return transform

    def forward(self, batch, epoch, train_loader):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :]
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)]
            
        images_aug = batch['images_aug']
        i_feats_aug = self.encode_image(images_aug)  
    
        mlm_tokens = batch['mlm_ids']
        mlm_tokens_feats = self.encode_text(mlm_tokens)            
                       
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        
        index = batch['index']        


        if "memory_soft" in self.current_task:    
            if epoch==1:
                pass
            else:
                sampled_indices = torch.zeros(i_feats.shape[0]*3, dtype=torch.long, device=i_feats.device)-1
                sampled_nums = torch.zeros(i_feats.shape[0], dtype=torch.long, device=i_feats.device)-1
                                                
                i_feats_norm = i_feats / i_feats.norm(dim=-1, keepdim=True)
                img_memory_norm = self.img_memory / self.img_memory.norm(dim=-1, keepdim=True)
                sim_matrix_i = i_feats_norm @ img_memory_norm.t()
                
                # t_feats_norm = t_feats / t_feats.norm(dim=-1, keepdim=True)
                # txt_memory_norm = self.txt_memory / self.txt_memory.norm(dim=-1, keepdim=True)
                # sim_matrix_t = t_feats_norm @ txt_memory_norm.t()
                                                
                softmax = torch.nn.Softmax(dim=0)
                sample_weight = []  
                for n in range(sim_matrix_i.shape[0]):
                    current_similarities = sim_matrix_i[n]
                    sorted_indices = torch.argsort(current_similarities, descending=True)
                    candidate_indices = sorted_indices[1:4]
                    
                    candidate_indices = [i.item() for i in candidate_indices if current_similarities[i]>self.args.id_threshod]
                
                    if len(candidate_indices)==0:
                        sample_weight.append(-1)
                    if len(candidate_indices)>0:
                        # current_similarities[candidate_indices]
                        new_element = torch.tensor([1.0], device='cuda', dtype=current_similarities[candidate_indices].dtype)
                        new_tensor = torch.cat((new_element, current_similarities[candidate_indices]), dim=0)
                        weight = 1-softmax(self.args.soft_scale * new_tensor)
                        sample_weight.append(weight[1:])
                                                                
                    len_pad = len(candidate_indices)
                    start_index = (sampled_indices == -1).nonzero(as_tuple=False)[0].item()
                    sampled_indices[start_index:start_index+len_pad] = torch.tensor(candidate_indices)
                    sampled_nums[n]=len_pad
                                      
                if  (sampled_indices == -1).any().item():
                    end_index = (sampled_indices == -1).nonzero(as_tuple=False)[0].item()
                    sampled_indices = sampled_indices[:end_index]
                    
                    pos_sample_img_all = self.img_memory[sampled_indices]
                    pos_sample_txt_all = self.txt_memory[sampled_indices]                    
                    
                    result_list= [i for i in range(i_feats.shape[0])]
                    for num in range(i_feats.shape[0]):
                        result_list.extend([num] * sampled_nums[num])
                        
                else:
                    pos_sample_img_all = self.img_memory[sampled_indices]
                    pos_sample_txt_all = self.txt_memory[sampled_indices]
                    result_list= [i for i in range(i_feats.shape[0])]
                    for num in range(i_feats.shape[0]):
                        result_list.extend([num] * 3)

                new_batch_img = torch.cat([i_feats, pos_sample_img_all])
                new_batch_txt = torch.cat([t_feats, pos_sample_txt_all])
                
                new_batch_size = new_batch_img.shape[0]
                pid = torch.tensor(result_list).reshape((-1, 1)).cuda() # make sure pid size is [batch_size, 1]
                pid_dist = pid - pid.t()
                labels = (pid_dist == 0).half()
                # labels = (labels-torch.eye(new_batch_size).cuda()) * self.args.mem_soft + torch.eye(new_batch_size).cuda()  
                                
                with torch.no_grad():
                    l = labels[:i_feats.shape[0],:]
                    for idex, i_label in enumerate(l):
                        if sampled_nums[idex] > 0:
                            nozero_idex = torch.nonzero(i_label).flatten()[1:]
                            i_label[nozero_idex]=sample_weight[idex] 
                        else:
                            continue
                    labels[:i_feats.shape[0],:]=l
                    
                ret.update({'mem_loss': objectives.compute_sdm_label(new_batch_img.float(), new_batch_txt.float(), pid=labels)})   
                
            if epoch == 1:    
                self.img_memory[index] = i_feats.detach()
                self.txt_memory[index] = t_feats.detach() 
            else:
                self.img_memory[index] = self.args.alpha_mom * i_feats.detach() + (1 - self.args.alpha_mom) * self.img_memory[index]
                self.txt_memory[index] = self.args.alpha_mom * t_feats.detach() + (1 - self.args.alpha_mom) * self.txt_memory[index]      

                            
        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats.float(), t_feats.float(), logit_scale)})
            
        if 'itc1' in self.current_task:
            ret.update({'sdm_loss':objectives.get_similarity_soft_base(i_feats.float(), t_feats.float(), 
                                                                       self.args.thred1, self.args.thred2)
                        })            
        if 'cons' in self.current_task:           
            ret.update({'cons_loss':objectives.get_similarity_soft_base(i_feats_aug.float(), mlm_tokens_feats.float(),
                                                                                                self.args.thred1, self.args.thred2)
                        })           
        

        return ret


def build_model(args, num_classes=11003):
    model = DGCMIA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
