from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, make_scheduler
from processor import do_train
from utils.set_seed import set_seed
import torch

seed = 1024
set_seed(seed)

lr = 6e-6
epochs = 20
batch_size = 1
device = "cuda:0"
load_checkpoint = False
# --> data
training_loader, testing_loader = make_dataloader("data/S2648.csv", "data/s669.csv", batch_size)
# --> model
model = make_model("ESM_ensamble_cls_env_attention")
optimizer_name = "Adam"
# --> optimizer and scheduler
optimizer =  make_optimizer(model, lr, optimizer_name=optimizer_name)
scheduler =  make_scheduler(optimizer, lr, training_loader, epochs, scheduler_name="CosineLR")
# --> load
if load_checkpoint:
    checkpoint_path = "[path of your checkpoint]"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
model.to(device)
# --> train
do_train(model, 
         optimizer, 
         scheduler, 
         training_loader, 
         testing_loader, 
         device, 
         epochs)