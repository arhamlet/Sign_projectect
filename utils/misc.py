import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   checking accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:           #for i, (images, labels) in enumerate(loader)
            x = x.to(device=device)
            y = y.to(device=device)
            print(y)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        accuracy = float(num_correct)/float(num_samples)*100
        print(f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}")

        return accuracy


def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
    print("Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    model.eval()
    print('Loading Checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint["epoch"]
    best_acc = checkpoint["acc"]
    print(f"=> loaded checkpoint at epoch {epoch})", checkpoint["epoch"])