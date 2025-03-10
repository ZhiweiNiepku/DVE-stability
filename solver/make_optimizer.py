import torch

def make_optimizer(model, lr, optimizer_name="Adam"):
    if optimizer_name == "Adam":
        print("using Adam")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        print("using AdamW")
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=0.01)
    return optimizer