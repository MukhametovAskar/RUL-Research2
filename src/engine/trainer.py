import torch


def train_one_epoch(encoder, ema_encoder, loader, optimizer, criterion, device):
    encoder.train()
    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        pred_rul, log_weights, _, _ = encoder(batch_x)
        loss, loss_mse, loss_kl = criterion(pred_rul, log_weights, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        ema_encoder.update_parameters(encoder)

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_kl += loss_kl.item()

    n = len(loader)
    return total_loss / n, total_mse / n, total_kl / n
