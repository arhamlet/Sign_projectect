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
from utils.dataloading import *
from utils.visuals import *
from utils.misc import *

classes = ('0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9')

learning_rate = 0.001
resume=None
weight_decay = 0
momentum = 0.9
epochs = 2
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
# best_acc = 0
# best_loss = 1000


class Training(object):
    def data_loading(self):
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
        return train_loader, valid_loader, test_loader

    def train(self):
        train_loader, valid_loader, test_loader = self.data_loading()
        #   Tensorboard writer
        # step = 0
        print("Initializing Model")
        #   model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.googlenet(pretrained=True, aux_logits=False)
        model.fc = nn.Linear(1024, 10)
        model.to(device)
        #writer = SummaryWriter(log_dir='graphs')

        # Visualize model in TensorBoard
        #images1, _ = next(iter(train_loader))
        #writer.add_graph(model, images1.to(device))
        #writer.close()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        #loading model
        if load_model:
            load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

        # time
        start_time = time.time()

        print("Start training")
        #   Training the network
        for epoch in range(epochs):
            epoch_start_time = time.time()
            losses = []
            best_acc = 0
            best_loss = 1000
            total_batch_images = 0
            batch_correct_pred = 0
            writer = SummaryWriter(log_dir='graphs')
            step = 0
            vis = visual()
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
                                   vis.plot_classes_preds(model,images,labels),
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

        print("Checking accuracy on training Set")
        check_accuracy(train_loader, model)

        print("Checking accuracy on valid Set")
        check_accuracy(valid_loader, model)

        print("Checking accuracy on test Set")
        check_accuracy(test_loader, model)
        
    def main(self):
        if __name__ == "__main__":
            self.data_loading()
            self.train()

t = Training()
t.main()