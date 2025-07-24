import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer2(args, model):
    params = []
    keys = []
    model.requires_grad_(True)
    args.lr1 = args.lr
    print(f'lr_init:{args.lr}')
    
    for key, value in model.named_parameters():    
        
        if "base_model.transformer" in key :
            value.requires_grad_(True)
            lr = args.lr1 * args.lr_factor
            weight_decay = args.weight_decay  
            keys += [key]           
            params += [{"names":[key],"params": [value], "lr": lr, "weight_decay": weight_decay}]  
            continue

        if key in['base_model.text_projection','base_model.positional_embedding',
                                                     'base_model.token_embedding.weight',
                                                     'base_model.ln_final.weight','base_model.ln_final.bias']:            
            value.requires_grad_(True)
            lr = args.lr1  
            weight_decay = args.weight_decay  
            keys += [key]           
            params += [{"names":[key],"params": [value], "lr": lr, "weight_decay": weight_decay}]  
            continue
        
            # value.requires_grad_(False)
            # continue

        if "base_model.visual" in key: 
            value.requires_grad_(True)
            lr = args.lr1 
            weight_decay = args.weight_decay  
            keys += [key]           
            params += [{"names":[key],"params": [value], "lr": lr, "weight_decay": weight_decay}] 
            continue  
        else:   
            lr = args.lr1
            weight_decay = args.weight_decay
            params += [{"names":[key], "params":[value], "lr": lr, "weight_decay": weight_decay}]
    

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr1, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr1,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr1,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer



def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        if "classifier" in key or "mlm_head" in key:
            lr = args.lr * args.lr_factor
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
