from tqdm import tqdm
import torch
from torch.cuda import amp
import numpy as np
from scipy import stats
from scipy.stats import pearsonr

def do_test(model, testing_loader, device):

    model.to(device)
    labels, predictions = do_valid(model, testing_loader, device)
    print(f'RMSE {np.sqrt(np.mean((labels-predictions)**2))} MAE {np.mean(np.abs(labels - predictions))} Correlation {stats.spearmanr(labels, predictions)}, Pearsonr {pearsonr(labels, predictions)[0]}')


def do_valid(model, testing_loader, device):
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels, eval_scores = [], [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            input_ids1, input_ids2, pos, labels, classes = batch            
            input_ids1 = input_ids1.to(device)
            input_ids2 = input_ids2.to(device)
            labels = labels.to(device).reshape(-1, 1)
            logits = model(input_ids1, input_ids2, pos)
            loss = torch.nn.functional.mse_loss(logits, labels)
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
                     
            eval_labels.extend(labels.cpu().detach())
            eval_preds.extend(logits.cpu().detach())
            
    labels = [id.item() for id in eval_labels]
    predictions = [id.item() for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")

    return np.array(labels), np.array(predictions)