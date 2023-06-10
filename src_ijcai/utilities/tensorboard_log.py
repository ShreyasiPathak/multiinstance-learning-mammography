# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

def matplotlib_imshow(img, one_channel):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_image(writer, images):
    print(images.shape)
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    print(img_grid.shape)
    # show images
    #matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('breast cancer data', img_grid)

def add_graph(writer, model, images, device):
    images=images.to(device)
    writer.add_graph(model, images)
    
# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def embedding_data(writer, classes, trainset):
    # select random images and their target indices
    images, labels = select_n_random(trainset.data, trainset.targets)
    
    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
    
# helper functions
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def loss_plot(writer, file_name, train_loss, val_loss, epoch):
    writer.add_scalars(file_name+'_loss', {'train_loss':train_loss, 'val_loss': val_loss}, epoch)#running_loss / 1000, epoch * len(batches) + batch_num)

def acc_plot(writer, file_name, train_acc, val_acc, epoch):
    writer.add_scalars(file_name+'_acc', {'train_acc':train_acc, 'val_acc': val_acc}, epoch)#running_loss / 1000, epoch * len(batches) + batch_num)

def grad_norm_plot(writer, name, param, iterno):
    writer.add_histogram(name, param.grad, iterno)

def tensorboard_log(epochs, file_name, dataloader_train, model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:2" if use_cuda else "cpu")
    
    writer = SummaryWriter('./runs/'+file_name)
    # get some random training images
    '''dataiter = iter(dataloader_train)
    idx, images, labels = dataiter.next()
    images1=images[0,:,:,:,:]'''
    
    '''for n_iter in range(epochs):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    '''
    
    #plot_image(writer, images1)
    #add_graph(writer, model, images, device)
    
    #writer.close()
    return writer

def weight_histograms_conv2d(writer, step, weights, layer_number):
  weights_shape = weights.shape
  num_kernels = weights_shape[0]
  for k in range(num_kernels):
    flattened_weights = weights[k].flatten()
    tag = f"layer_{layer_number}/kernel_{k}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number):
  flattened_weights = weights.flatten()
  tag = f"layer_{layer_number}"
  writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model):
  print("Visualizing model weights...")
  # Iterate over all model layers
  for layer_number in range(len(model.layers)):
    # Get layer
    layer = model.layers[layer_number]
    # Compute weight histograms for appropriate layer
    if isinstance(layer, nn.Conv2d):
      weights = layer.weight
      weight_histograms_conv2d(writer, step, weights, layer_number)
    elif isinstance(layer, nn.Linear):
      weights = layer.weight
      weight_histograms_linear(writer, step, weights, layer_number)

def histogram(writer, model, epoch):
    writer.add_histogram('four_view_resnet.cc.conv1.weight', model.four_view_resnet.cc.conv1.weight, epoch)
    writer.add_histogram('four_view_resnet.cc.layer1[0].conv1.weight', model.four_view_resnet.cc.layer1[0].conv1.weight, epoch)
    writer.add_histogram('four_view_resnet.cc.layer2[0].conv1.weight', model.four_view_resnet.cc.layer2[0].conv1.weight, epoch)
    writer.add_histogram('four_view_resnet.cc.layer3[0].conv1.weight', model.four_view_resnet.cc.layer3[0].conv1.weight, epoch)
    writer.add_histogram('four_view_resnet.cc.layer4[0].conv1.weight', model.four_view_resnet.cc.layer4[0].conv1.weight, epoch)
    writer.add_histogram('four_view_resnet.mlo.conv1.weight', model.four_view_resnet.mlo.conv1.weight, epoch)
    writer.add_histogram('four_view_resnet.mlo.layer1[0].conv1.weight', model.four_view_resnet.mlo.layer1[0].conv1.weight, epoch)
    writer.add_histogram('four_view_resnet.mlo.layer2[0].conv1.weight', model.four_view_resnet.mlo.layer2[0].conv1.weight, epoch)
    writer.add_histogram('four_view_resnet.mlo.layer3[0].conv1.weight', model.four_view_resnet.mlo.layer3[0].conv1.weight, epoch)
    writer.add_histogram('four_view_resnet.mlo.layer4[0].conv1.weight', model.four_view_resnet.mlo.layer4[0].conv1.weight, epoch)
    writer.add_histogram('model_attention_left.attention[0].weight', model.model_attention_left.attention[0].weight, epoch)
    writer.add_histogram('model_attention_right.attention[0].weight', model.model_attention_right.attention[0].weight, epoch)
    writer.add_histogram('model_attention_both.attention[0].weight', model.model_attention_both.attention[0].weight, epoch)
    writer.add_histogram('combined_linear_feature.linear[0].weight', model.combined_linear_feature.linear[0].weight, epoch)
    writer.add_histogram('classifier[0].weight', model.classifier[0].weight, epoch)
