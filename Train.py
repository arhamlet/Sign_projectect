import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F

classes = ('0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9')

learning_rate = 0.001
resume=None
weight_decay = 0
momentum = 0.9
epochs = 50
batch_size = 32
log_interval = 240
number_workers = 0
create_validationset = True
seed = 1
save_model = True
init_padding=2
validation_size=0.2
random_seed=1
in_channels = 3
load_model = True
best_acc = 0
best_loss = 1000









#   checking accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:           #for i, (images, labels) in enumerate(loader)
            x = x.to(device=device)
            y = y.to(device=device)

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

def load_checkpoint(checkpoint):
    model.eval()
    print('Loading Checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint["epoch"]
    best_acc = checkpoint["acc"]
    print(f"=> loaded checkpoint at epoch {epoch})", checkpoint["epoch"])

class Signdataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)   #number of images

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


#   Code execution starts here
print("Loading Dataset")
data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Grayscale(num_output_channels=3),
            #transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.7107, 0.6669, 0.6378), (0.2402, 0.2660, 0.2668))     #transforms.Normalize(mean,std)
])


dataset = Signdataset(                       #generic_data
    csv_file="signlabels.csv",
    root_dir="images",
    transform=data_transforms
)

train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [4000, 240, 700])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size,shuffle= True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#   Tensorboard writer
writer = SummaryWriter(log_dir='graphs')
#step = 0

def matplotlib_imshow(img):
    #img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow((npimg * 255).astype(np.uint8))   #fixed error .astype('uint8')


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(6, 6))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])

        matplotlib_imshow(images[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

print("Initializing Model")
#   model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.googlenet(pretrained = True, aux_logits=False)
model.fc = nn.Linear(1024,10)
#print(model)
model.to(device)


# Visualize model in TensorBoard
#images1, _ = next(iter(train_loader))
#writer.add_graph(model, images1.to(device))
#writer.close()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience= 5, verbose = True)



#loading model
if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))

# time
start_time = time.time()

print("Start training")
#   Training the network
for epoch in range(epochs):
    epoch_start_time = time.time()
    losses = []
    total_batch_images = 0
    batch_correct_pred = 0
    step = 0
    #save model
    # if batch_accuracy>best_acc:
    #     best_acc = batch_accuracy
    #     checkpoint = {'state_dict': model.state_dict(),'acc' : batch_accuracy, 'epoch' : epoch,  'optimizer': optimizer.state_dict()}
    #     save_checkpoint(checkpoint)

    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Get data to cuda if possible
        
        images = images.to(device=device)
        labels = labels.to(device=device)

        # forward
        scores = model(images)
        loss = criterion(scores, labels)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        # visualizing Dataset images
        # img_grid = torchvision.utils.make_grid(images)
        # writer.add_image('Xray_images', img_grid, global_step = step)

        # calculation running accuracy
        model.eval()
        _, predictions = scores.max(1)
        num_correct = (predictions == labels).sum()
        batch_correct_pred += float(num_correct)
        total_batch_images += predictions.size(0)

        writer.add_figure('predictions vs. actuals',
                           plot_classes_preds(model, images, labels),
                           global_step=step)      #epoch * len(train_loader) + batch_idx
        step += 1

    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)
        
    print(batch_correct_pred)
    epoch_elapsed = (time.time() - epoch_start_time) / 60
    print(f'Epoch {epoch} completed in : {epoch_elapsed:.2f} min')

    batch_loss = sum(losses)/len(losses)
    batch_accuracy = (batch_correct_pred/total_batch_images)*100

    print(f"Cost at epoch {epoch} is {batch_loss}")
    print(f"Training accuracy at {epoch} is: {batch_accuracy:.2f}")
    # batch_accuracy = check_accuracy(train_loader, model)

    if batch_accuracy>=best_acc and batch_loss<best_loss:
        best_acc = batch_accuracy
        best_loss = batch_loss
        checkpoint = {'state_dict': model.state_dict(),'acc' : batch_accuracy, 'epoch' : epoch,  'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    writer.add_scalar('Training loss', batch_loss, global_step=step)
    writer.add_scalar('Training accuracy', batch_accuracy, global_step=step)
    step += 1


elapsed = (time.time() - start_time)/60
print(f'Training completed in: {elapsed:.2f} min')

# Accuracy Check
# exp
#print(model)
# exp
print("Checking accuracy on training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on valid Set")
check_accuracy(valid_loader, model)

print("Checking accuracy on test Set")
check_accuracy(test_loader, model)
