from scipy import stats
from scipy.stats import pearsonr
import os
import torch
import numpy as np
from torch.cuda import amp
from tqdm import tqdm

def do_train(model, optimizer, scheduler, training_loader, testing_loader, device, epochs):
    
    scaler = amp.GradScaler()
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        nb_tr_examples, nb_tr_steps = 0, 0
        for idx, batch in enumerate(training_loader):
            optimizer.zero_grad()
            input_ids1, input_ids2, pos, labels, classes = batch
        
            input_ids1 = input_ids1.to(device)
            input_ids2 = input_ids2.to(device)
            labels = labels.to(device).reshape(-1, 1)

            with amp.autocast(enabled=True):
                logitis_output = model(input_ids1, input_ids2, pos, is_training = True)
                if len(logitis_output) == 3:
                    logits_1, logits_2, logits_mean = logitis_output[0], logitis_output[1], logitis_output[2]
                    loss = torch.nn.functional.mse_loss(logits_1, labels) + torch.nn.functional.mse_loss(logits_2, labels) + torch.nn.functional.mse_loss(logits_mean, labels)
                else:
                    logits_pred = logitis_output[0]
                    loss = torch.nn.functional.mse_loss(logits_pred, labels)
            tr_loss += loss.item()
            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
        scheduler.step(epoch)
        
        epoch_loss = tr_loss / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")

        labels, predictions = do_valid(model, testing_loader, device)
        print(f'RMSE {np.sqrt(np.mean((labels-predictions)**2))} MAE {np.mean(np.abs(labels - predictions))} \
            Correlation {stats.spearmanr(labels, predictions)}, Pearsonr {pearsonr(labels, predictions)[0]}')
        
    print("saving model...")
    torch.save(model.state_dict(), "checkpoint/training.pt")


def do_valid(model, testing_loader, device):
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels, eval_scores = [], [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            input_ids1, input_ids2, pos, labels, classes = batch  
            # input_ids1, input_ids2, pos, labels = batch          
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
    predictions = [id.item() for id in eval_preds]w
    
    eval_loss = eval_loss / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")

    return np.array(labels), np.array(predictions)