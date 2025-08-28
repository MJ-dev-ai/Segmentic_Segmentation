import torch
from tqdm.auto import tqdm

def train(model, dataloader, loss_fn, optimizer, device, flags):
    model.train()
    total_loss = []
    for epoch in range(flags.epochs):
        epoch_loss = 0.0
        bar = tqdm(dataloader)
        for images, masks in bar:
            bar.set_description("Train")
            images, masks = images.to(device), masks.to(device)
            model = model.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item()
            bar.set_postfix(loss=loss.cpu().item())
        
        epoch_loss /= len(dataloader)
        total_loss.append(epoch_loss)
        print(f"Epoch {epoch+1}/{flags.epochs}, Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), flags.save_path)
    return total_loss