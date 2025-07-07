from .models import ESM_ensamble_cls_env_attention

def make_model(model_name):
    model_class = globals()[model_name]
    model = model_class()
    return model