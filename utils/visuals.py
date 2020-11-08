import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F

classes = ('0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9')



class visual(object):
    def matplotlib_imshow(self, img):
        #img = img / 2 + 0.5  # unnormalize
        npimg = img.cpu().numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        plt.imshow((npimg * 255).astype(np.uint8))   #fixed error .astype('uint8')


    def images_to_probs( self, net, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = net(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


    def plot_classes_preds( self, net, images, labels):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self.images_to_probs(net, images)
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(6, 6))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])

            self.matplotlib_imshow(images[idx])
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"))
        return fig