import numpy as np
import torch
from src.utils.metrics import rmse, nasa_score


def evaluate(model, test_tensor, y_test, max_rul):
    model.eval()
    with torch.no_grad():
        test_preds, _, _, _ = model(test_tensor)
        preds = test_preds.cpu().numpy()

    preds = np.clip(preds, 0, max_rul)
    cur_rmse = rmse(y_test, preds)
    cur_score = nasa_score(y_test, preds)
    return cur_rmse, cur_score, preds
