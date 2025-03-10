import torch
from datasets import make_dataloader
from model import make_model
from processor import do_test
from utils.set_seed import set_seed

batch_size = 8
device = "cuda:0"

set_seed(1024)
testing_loader = make_dataloader(test_dataset_path = "data/S669.csv", batch_size=batch_size, is_training=False)
model = make_model("ESM_ensamble_cls_env_attention")
checkpoint = torch.load("[path of your checkpoint]", map_location="cpu")
model.load_state_dict(checkpoint)

do_test(
    model,
    testing_loader,
    device
)