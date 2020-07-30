# torch stuff
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from facenet_pytorch import MTCNN

# standard stuff
import numpy as np
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import os
import glob
import json
import cv2
from PIL import Image
from copy import deepcopy
from scipy import interp

# visualisation
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

# sklearn stufff
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score
)

# modules
from src.preprocessing_util import FaceDetection, get_face_labels_from_file_labels
from src.EfficientNet import EfficientNet, EfficientLSTM
from src.MesoNet import MesoNet4, Meso4LSTM


class VideoDataset(Dataset):

    """
    class holding the temporal version of the datasets
    """

    def __init__(self, labels_per_face_window, root_dir, joint_training=False,
                 second_root_dir=None, seq_len=5, is_train_loader=True):

        self.labels = labels_per_face_window
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.joint_training = joint_training
        self.second_root_dir = second_root_dir
        self.is_train_loader = is_train_loader

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        grabs ith file from data and returns a tensor of M cropped face images and label
        Args:
            idx: position of the datapoint

        Returns:
            tuple: tensor of M cropped face images and boolean label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if training from both datasets, randomly choose from either of the two root dirs
        if self.joint_training:
            if np.random.sample() > 0.5:
                root_dir = self.second_root_dir
            else:
                root_dir = self.root_dir
        else:
            root_dir = self.root_dir

        # load image and label
        im_seq, label = self.load_im(idx, root_dir)

        return im_seq, label

    def load_im(self, idx, root_dir):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # check value of random flip
        if np.random.sample() > 0.5 and self.is_train_loader:
            flip = True
        else:
            flip = False

        # get face image name
        file = self.labels.iloc[idx, 0]

        # get label associated with face image
        label = self.labels.iloc[idx, 1]

        # get image tensor of cropped facce
        im = torch.load(root_dir + file + '.pt')

        if self.seq_len != 5:
            im = im[:, :self.seq_len, :, :, :]

        # do random horizontal flip
        if flip:
            im = im.flip(4)

        im = torch.as_tensor(im, dtype=torch.float32)

        return im[0], label


class ImageDataset(Dataset):
    """
    class holding the non-temporal version of the datasets
    """

    def __init__(self, labels, root_dir, joint_training=False, second_root_dir=None, is_train_loader=True):

        self.labels = labels
        self.root_dir = root_dir
        self.joint_training = joint_training
        self.second_root_dir = second_root_dir
        self.is_train_loader = is_train_loader

        if self.is_train_loader:
            self.transform = transforms.Compose([

                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=224, scale=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([

                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])

        if self.joint_training:
            assert self.second_root_dir, 'Joint training is selected but no second dir has been specified'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # if training from both datasets, randomly choose from either of the two root dirs
        if self.joint_training:
            if np.random.sample() > 0.5:
                root_dir = self.second_root_dir
            else:
                root_dir = self.root_dir
        else:
            root_dir = self.root_dir

        # load image and label
        im, label = self.load_im(idx, root_dir)

        return im, label

    def load_im(self, idx, root_dir):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get face image name
        file = self.labels.iloc[idx, 0]

        # get label associated with face image
        label = self.labels.iloc[idx, 1]

        # get image tensor of cropped face
        im = Image.open(root_dir + file + '.jpg')
        im = self.transform(im)

        return im, label


def create_train_test_sets(labels_per_file, labels_per_face, root_dir, root_dir_aug, train_size=0.8, temporal=False):
    """
    this function creates a train, val, test split given a per file label csv
    it partitions and moves the image files into three folders: train, val, test, and saves a label csv file
    to each folder as well
    Args:
        labels_per_file: path to csv label file
        labels_per_face: path to csv label file per face
        root_dir: path to all images
        root_dir_aug: path to all augmented images
        train_size: float, proportion of images used for train set
        temporal: whether dataset is for temporal model or not

    Returns: nothing

    """

    # read in full list of files/labels and shuffle
    labels_per_file = pd.read_csv(labels_per_file, index_col=0).sample(frac=1).reset_index(drop=True)
    labels_per_face = pd.read_csv(labels_per_face, index_col=0).sample(frac=1).reset_index(drop=True)

    # use 0.x for training
    train, val_test = train_test_split(labels_per_file, test_size=1 - train_size)

    # use 1 - 0.x/2 for validation and 1 - 0.x/2 for testing
    val, test = train_test_split(val_test, test_size=0.5)

    # local function that takes one split (pandas label df) and redistributes
    # face image files to the corresponding folder
    def process_split(split, labels_per_face, name, root_dir, root_dir_aug):
        # derive face level labels
        split_per_face = get_face_labels_from_file_labels(split, labels_per_face)

        print("\nPer face labelling derived for split:")
        print(split_per_face)

        print("\nPer window labelling for split:")
        print(split)

        # store labels in split folders (both raw and aug)
        # per file (balanced)
        split.to_csv(root_dir + name[:-1] + 'labels_per_file.csv')
        split.to_csv(root_dir_aug + name[:-1] + 'labels_per_file.csv')

        # per face
        if temporal:
            filename = 'labels_per_face_window.csv'
        else:
            filename = 'labels_per_face.csv'
        split_per_face.to_csv(root_dir + name[:-1] + filename)
        split_per_face.to_csv(root_dir_aug + name[:-1] + filename)

    # for each split, locate the files and move to new directory
    for split, name in zip([train, val, test], ['train/', 'val/', 'test/']):
        process_split(split, labels_per_face, name, root_dir, root_dir_aug)
        print(f"Finished split {name}!")


def init_train_test_sets(root_dir, verbose=True, type_model='image', joint_training=False, second_root_dir=None, seq_len=5):

    # local function that reads in the appropriate label csvs and returns pandas df with labels for each split
    def read_in_train_test_labels(root_dir, type_model):

        # check which type of label is appropriate for the selected model type
        if type_model == 'video' or type_model == 'pairwise':
            type_file = 'face_window'
        else:
            type_file = 'face'

        # read in all three label files
        train = pd.read_csv(
            root_dir + 'trainlabels_per_' + type_file + '.csv', index_col=0).sample(frac=1).reset_index(drop=True)
        val = pd.read_csv(
            root_dir + 'vallabels_per_' + type_file + '.csv', index_col=0).sample(frac=1).reset_index(drop=True)
        test = pd.read_csv(
            root_dir + 'testlabels_per_' + type_file + '.csv', index_col=0).sample(frac=1).reset_index(drop=True)

        return train, val, test

    # local function that initializes the pytorch dataset instances
    def create_datasets(train, val, test, root_dir, type_model):

        # depending on the type of model, initialize the correct dataloader
        if type_model == 'video':
            trainset = VideoDataset(train, root_dir, joint_training, second_root_dir, seq_len)
            valset = VideoDataset(val, root_dir, joint_training, second_root_dir, seq_len, is_train_loader=False)
            testset = VideoDataset(test, root_dir, joint_training=False, second_root_dir=second_root_dir, seq_len=seq_len, is_train_loader=False)
        elif type_model == 'image':
            trainset = ImageDataset(train, root_dir, joint_training, second_root_dir)
            valset = ImageDataset(val, root_dir, joint_training, second_root_dir, is_train_loader=False)
            testset = ImageDataset(test, root_dir, joint_training=False, second_root_dir=second_root_dir, is_train_loader=False)

        else:
            assert False, "Invalid type of Dataset specified."

        return trainset, valset, testset

    """MAIN STEPS"""
    # read in the correct labels
    train, val, test = read_in_train_test_labels(root_dir, type_model)

    # create the corresponding datasets for the specified type of model
    trainset, valset, testset = create_datasets(train, val, test, root_dir, type_model)

    # print some stuff about the process
    if verbose:

        if joint_training:
            print("Training on joint dataset raw + aug")
        if type_model == 'video':
            print("Sucessfully created temporal versions of training, validation and test sets.")
        elif type_model == 'image':
            print("Sucessfully created non-temporal versions of training, validation and test sets.")
        else:
            print("Sucessfully created pairwise difference versions of training, validation and test sets.")

        print(f"Length of training data: {len(trainset)}")
        print(f"Length of validation data: {len(valset)}")
        print(f"Length of testing data: {len(testset)}")

    return trainset, valset, testset


def save_checkpoint(dir, model, optimizer, epoch):

    """
    function saves model at given training point
    Args:
        dir: path to save checkpoint
        model: model to save
        optimizer: current optimizer step
        epoch: current epoch

    Returns: nothing, but saves checkpoint
    """

    if not os.path.exists(dir):
        os.makedirs(dir)

    filepath = dir + 'checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)


def run_training(model,
                 batch_size,
                 num_epochs,
                 train_dataset,
                 title,
                 device,
                 validation_dataset=None,
                 lr=1e-4,
                 weight_decay=0.001,
                 show_plots=True,
                 num_workers=0,
                 log_dir=None,
                 freeze=False,
                 clip_gradients=False,
                 reduce_lr_on_plateau=False,
                 pickup_epoch=0):

    """
    function performs the main training loop

    Args:
        model: model to train
        batch_size: batch size
        num_epochs:  total number of epochs
        train_dataset: train dataset object
        title: title for plots
        device: cuda device
        validation_dataset: validation dataset object
        lr: initial learning rate
        weight_decay: weight decay
        show_plots: bool, whether or not to show plots during training
        num_workers: number of workers for dataloader
        log_dir: path to dir to save checkpoints
        freeze: whether freezing layers (defined in model class) should be enforced by optimizer
        clip_gradients: whether to use gradient clipping (for lstm models)
        reduce_lr_on_plateau: whether to reduce LR on validation plateau

    Returns: lists of training and validation loss/accuracy for each iter (batch)
    """

    # set model to training mode
    model.train()

    # init loss, dataloader, optimizer
    criterion = torch.nn.BCELoss().to(device)

    # init loader for training instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True)

    # if freeze is true, this freezes the predefined layers of the model
    if freeze:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # init learning rate scheduler
    if reduce_lr_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      mode='min',
                                      patience=2,
                                      verbose=True)
    if show_plots:
        # Init of the metric plots that show the training progress.
        model.metric_plots['fig'] = plt.figure(figsize=(9, 8), num=title)
        model.metric_plots['plt1'] = model.metric_plots['fig'].add_subplot(2, 1, 1)
        model.metric_plots['plt1'].set_ylim([0, 1])
        model.metric_plots['plt1_legend'] = None
        model.metric_plots['plt2'] = model.metric_plots['fig'].add_subplot(2, 1, 2)
        model.metric_plots['plt2_legend'] = None
        model.metric_plots['fig'].suptitle(title, y=0.93)

    # number of iterations per epoch
    num_train_iters_per_epoch = len(train_dataset) // batch_size

    # overall num of iterations
    num_train_iters = num_train_iters_per_epoch * (num_epochs-pickup_epoch)

    # In case we passed a validation dataset we also need to prepare its usage.
    if validation_dataset:
        # Define a validation data loader similar as for training above.
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=num_workers, pin_memory=True)

        # Calculate the iterations per epoch of validation.
        num_val_iters_per_epoch = len(validation_dataset) // batch_size

        # Then we space the evaluation iterations evenly across a training epoch, so that they finish together.
        val_iters_2_train_iters = np.linspace(start=0, stop=num_train_iters_per_epoch,
                                              num=num_val_iters_per_epoch,
                                              endpoint=False,
                                              dtype=int)

    # In case there is no validation dataset passed, we just define no validation iters and a dummy val loader.
    else:
        val_iters_2_train_iters = []
        val_loader = None

    # Theses lists hold the loss and accuracy metrics over training iters.
    iter_val_loss_list = list()
    iter_val_acc_list = list()
    iter_train_loss_list = list()
    iter_train_acc_list = list()
    iter_val_precision_list = list()
    iter_val_recall_list = list()
    iter_val_f1_list = list()

    # We initialize a progressbar over total number of iterations for training.
    pbar = tqdm(total=num_train_iters, position=0, unit="it", leave=True)

    # main training loop
    for epoch in range(pickup_epoch, num_epochs):

        # if we passed validation set
        if validation_dataset:
            val_loader_iter = iter(val_loader)

        # for all batches in train loader
        for iter_id, batch in enumerate(train_loader):

            im_batch, label_batch = batch
            im_batch = torch.as_tensor(im_batch, dtype=torch.float32).to(device)

            label_batch = torch.as_tensor(label_batch, dtype=torch.float32).to(device)

            # check batch size
            assert im_batch.shape[0] == batch_size, "Batch dimension incorrect."

            # feed through model
            fake_prob = model.forward(im_batch)

            # calculate loss based on labels
            fake_prob = F.sigmoid(fake_prob)

            loss = criterion(fake_prob, label_batch.view(-1, 1))

            # Reset the gradients in the network.
            optimizer.zero_grad()

            # Backprob the loss through the network.
            loss.backward()

            # clip gradients for lstms (experimental, not yet tuned)
            if clip_gradients:
                if model.__class__.__name__ == "Meso4LSTM" or model.__class__.__name__ == "EfficientLSTM":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Make a step of optimization.
            optimizer.step()

            # calculate accuracy of batch
            iter_train_acc = calculate_accuracy(label_batch, fake_prob)

            # Calculate the global iteration counter used for plotting.
            global_iter = epoch * num_train_iters_per_epoch + iter_id

            # save training loss and accuracy
            iter_train_loss_list.append([global_iter, loss.item()])
            iter_train_acc_list.append([global_iter, iter_train_acc])

            # free to save memory
            del fake_prob, loss
            torch.cuda.empty_cache()

            # if iteration is also validation iteration
            if validation_dataset and iter_id in val_iters_2_train_iters:
                # eval mode
                model.eval()

                # get batch validation data
                im_batch, label_batch = next(val_loader_iter, (None, None))

                assert im_batch is not None, "Validation passed empty batch"

                im_batch = torch.as_tensor(im_batch, dtype=torch.float32).to(device)
                label_batch = torch.as_tensor(label_batch, dtype=torch.float32).to(device)

                # no gradient computations
                with torch.no_grad():
                    assert im_batch.shape[0] == batch_size, "Validation batch size incorrect"

                    fake_prob = model.forward(im_batch)
                    fake_prob = F.sigmoid(fake_prob)

                    val_loss = criterion(fake_prob, label_batch.view(-1, 1))

                    iter_val_acc = calculate_accuracy(label_batch, fake_prob)
                    iter_val_precision, iter_val_recall, iter_val_f1 = calculate_f1(label_batch, fake_prob)

                    iter_val_loss_list.append([global_iter, val_loss.item()])
                    iter_val_acc_list.append([global_iter, iter_val_acc])

                    iter_val_precision_list.append([global_iter, iter_val_precision])
                    iter_val_recall_list.append([global_iter, iter_val_recall])
                    iter_val_f1_list.append([global_iter, iter_val_f1])

                    # for LR plateau reduce: calculate mean val loss 2 times per epoch, starting at epoch 10
                    if reduce_lr_on_plateau and iter_id % int(num_train_iters_per_epoch/2) == 0 and epoch >= 9:
                        mean_val_loss_last_quarter = np.asanyarray(iter_val_loss_list[-int(num_val_iters_per_epoch/2):])[:, 1].mean()
                        scheduler.step(mean_val_loss_last_quarter)

                    # free cache to save memory
                    del fake_prob, val_loss

                # set back to train mode
                model.train()

            # plotting scripts here to come
            # Every 200 iterations we redraw the training metric plots.
            if show_plots:
                if iter_id % 200 == 0:
                    update_plots(
                        model=model,
                        iter_train_loss_list=iter_train_loss_list,
                        iter_train_acc_list=iter_train_acc_list,
                        iter_val_loss_list=iter_val_loss_list,
                        iter_val_acc_list=iter_val_acc_list,
                    )
            pbar.update()

        # save checkpoint
        if log_dir:
            save_checkpoint(log_dir, model, optimizer, epoch)

        # Calculate the mean training loss and acc on the epoch for print statements.
        mean_train_acc_epoch = np.asanyarray(iter_train_acc_list[-num_train_iters_per_epoch:])[:, 1].mean()
        mean_train_loss_epoch = np.asanyarray(iter_train_loss_list[-num_train_iters_per_epoch:])[:, 1].mean()

        # In case of validation.
        if validation_dataset:
            # Calculate the mean validation loss and acc on the epoch for print statements.
            mean_val_acc_epoch = np.asanyarray(iter_val_acc_list[-num_val_iters_per_epoch:])[:, 1].mean()
            mean_val_loss_epoch = np.asanyarray(iter_val_loss_list[-num_val_iters_per_epoch:])[:, 1].mean()
            mean_val_f1_epoch = np.asanyarray(iter_val_f1_list[-num_val_iters_per_epoch:])[:, 1].mean()
            mean_val_precision_epoch = np.asanyarray(iter_val_precision_list[-num_val_iters_per_epoch:])[:, 1].mean()
            mean_val_recall_epoch = np.asanyarray(iter_val_recall_list[-num_val_iters_per_epoch:])[:, 1].mean()

            # print epoch stats including validation
            print(
                f"[Epoch: {epoch + 1:02d}/{num_epochs}] "
                f"[Mean acc train/val: {mean_train_acc_epoch:.3f}/{mean_val_acc_epoch:.3f}] "
                f"[Mean loss train/val: {mean_train_loss_epoch:.3f}/{mean_val_loss_epoch:.3f}] "
                f"[F1/Precision/Recall: {mean_val_f1_epoch: .3f}/{mean_val_precision_epoch: .3f}/{mean_val_recall_epoch: .3f}]"
            )
        else:
            # print epoch stats without validation
            print(
                f"[Epoch: {epoch + 1:02d}/{num_epochs}] "
                f"[Mean acc train: {mean_train_acc_epoch:.3f}] "
                f"[Mean loss train: {mean_train_loss_epoch:.3f}]"
            )

    pbar.close()

    return iter_val_loss_list, iter_val_acc_list, iter_train_loss_list, iter_train_acc_list


def evaluate(model, test_set, batch_size, title, device, save_to=None):

    """
    function performs evaluation loop over designated testset, reports accuracy
    also plots ROC curve and calculated AUC
    """

    # eval mode
    model.eval()

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    num_test_iters = len(test_set) // batch_size

    pbar = tqdm(total=num_test_iters, position=0, unit="it", leave=True)

    iter_test_acc_list = []
    full_pred = []
    full_label = []

    for im_batch, label_batch in test_loader:

        im_batch = torch.as_tensor(im_batch, dtype=torch.float32).to(device)
        label_batch = torch.as_tensor(label_batch, dtype=torch.float32).to(device)

        # no gradient computations
        with torch.no_grad():
            assert im_batch.shape[0] == batch_size, "Test batch size incorrect"

            fake_prob = model.forward(im_batch)
            fake_prob = F.sigmoid(fake_prob)

            iter_test_acc = calculate_accuracy(label_batch, fake_prob)

            full_pred.extend(torch.flatten(fake_prob).tolist())
            full_label.extend(torch.flatten(label_batch).tolist())

        iter_test_acc_list.append(iter_test_acc)
        pbar.update()

    acc = np.mean(iter_test_acc_list)
    print(f"[Test acc: {np.round(acc,3)}]")

    full_pred = np.asarray(full_pred)
    full_label = np.asarray(full_label)

    fpr, tpr, thresholds = roc_curve(full_label, full_pred)
    auc = roc_auc_score(full_label, full_pred)

    title += f" - [Test acc: {np.round(acc,3)}]"
    plot_roc_curve(fpr, tpr, auc, title, save_to)

    pbar.close()

    return acc


def plot_roc_curve(fpr, tpr, auc, title, save_to=None):
    """
    function plots, and shows the roc curve, AUC will be displayed in legend
    """

    lw = 1
    fig, axs = plt.subplots(1, 2, figsize=(14,6), num=title)
    for i, ax in enumerate(axs):
        ax.plot(fpr, tpr, color='darkorange',  lw=lw, label=f"ROC curve (AUC = {auc:0.2f})")
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        if i == 0:
            ax.axis([-0.02, 1.02, -0.02, 1.02])
        elif i == 1:
            ax.axis([0., 0.15, 0.85, 1.])
        else:
            assert False, 'Wrong amount of axis'
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")

    if save_to is not None:
        fig.savefig(save_to + '.svg', format='svg')
    else:
        plt.show()


def predict_files(model, path, device, temporal=False, title='ROC',
                  agg='mean', n_frames=10, plot_ims=False,
                  threshold=None, explore_thresholds=False, verbose=False, log_metrics=False):

    """
    used to predict clips from disk directly
    """

    from tqdm import tqdm_notebook as tqdm

    model.eval()

    # Load face detector
    face_detector = MTCNN(image_size=224, margin=10, keep_all=False, device=device, post_process=False).eval()
    face_detection = FaceDetection(face_detector, device, n_frames=n_frames)

    fake_files = glob.glob(os.path.join(path + 'fake/', '*.mp4'))
    real_files = glob.glob(os.path.join(path + 'real/', '*.mp4'))

    total_len = len(fake_files) + len(real_files)

    # progress bar
    pbar = tqdm(total=total_len, position=0, unit="it", leave=True)

    predictions = []
    labels = []

    # first do all clips from fake folder
    label = 'fake'
    predictions, labels = predict_clip_folder(model, fake_files, label, face_detection,
                                              predictions, labels, agg,  pbar, temporal, device, plot_ims=plot_ims,
                                              verbose=verbose)

    # next do all clips from real folder
    label = 'real'
    predictions, labels = predict_clip_folder(model, real_files, label, face_detection,
                                              predictions, labels, agg, pbar, temporal, device, plot_ims=plot_ims,
                                              verbose=verbose)

    # list to np arrays
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    # calc metrics and roc
    f1, precision, recall = calculate_f1(deepcopy(labels), deepcopy(predictions), from_numpy=True)
    fpr, tpr, thresholds = roc_curve(deepcopy(labels), deepcopy(predictions))
    auc = roc_auc_score(deepcopy(labels), deepcopy(predictions))

    if verbose:
        print(f"Length labels: {len(labels)}")
        print(f"Length predictions: {len(predictions)}")
        print(f"Optimal threshold = {thresholds[np.argmax(tpr - fpr)]}")

    if explore_thresholds:
        for th in [0.05, 0.1, 0.3, 0.5]:
            acc = None
            predictions_th = deepcopy(predictions)
            labels_th = deepcopy(labels)
            acc_th, predictions_th, labels_th = calculate_accuracy_np(labels_th, predictions_th, th)
            C = confusion_matrix(labels_th, predictions_th)

            if verbose:
                print(f"Confusion matrix for threshold {th}")
                print(C)
                print(f"[Acc: {np.round(acc_th, 3)}]")

    else:
        acc, predictions, labels = calculate_accuracy_np(labels, predictions, threshold)
        C = confusion_matrix(labels, predictions)

        if verbose:
            print(f"Confusion matrix for threshold {threshold}")
            print(C)
            print(
                f"[Acc: {np.round(acc, 3)}]\n"
                f"[Precision: {np.round(precision, 3)}]\n"
                f"[Recall: {np.round(recall, 3)}]\n"
                f"[F1: {np.round(f1, 3)}]"
            )

            plot_roc_curve(fpr, tpr, auc, title)

    pbar.close()

    if log_metrics:
        return acc, f1, precision, recall, auc, fpr, tpr
    else:
        return


def predict_clip_folder(model, files, label, face_detection,
                        predictions, labels, agg, pbar, temporal, device, plot_ims=False,
                        verbose=False):
    """
    needs to be revisited
    """

    with torch.no_grad():
        for file in files:
            filename = file.split('\\')[-1]

            single_label = 1 if label == 'fake' else 0

            faces = face_detection(file, temporal)

            if faces:
                faces = [f for f in faces if f is not None]
                if len(faces) == 0:
                    continue
            else:
                continue

            if verbose:
                #print(f"[Extracted {len(faces)} faces for file {filename}]")
                print(f"Label: {label}")

            # normalize
            faces = [face.type(torch.int32) / 255.0 for face in faces]
            faces = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(face) for face in
                     faces]

            # for temporal data, create face window tensors of length 5
            if temporal:
                face_windows = []
                for i in np.arange(0, len(faces), 5):
                    try:
                        face_windows.append(torch.stack([faces[i],
                                                         faces[i + 1],
                                                         faces[i + 2],
                                                         faces[i + 3],
                                                         faces[i + 4]]))

                    except IndexError:
                        print(f"Incomplete window indices at file {filename}")
                        pass

                if verbose:
                    print(f"[# extracted face windows: {len(face_windows)}]")

                faces = face_windows

            if not faces:
                pbar.update()
                continue

            faces = [face.unsqueeze(0) for face in faces]

            faces = torch.cat(faces)
            pred, fake_probs = predict_single_file(model, faces, agg, device, verbose)

            # check if correct prediction for plotting color
            correct = True if round(pred) == single_label else False

            if plot_ims:
                plot_sample_images(faces, title=fake_probs, temporal=temporal, correct=correct)

            predictions.append(pred)
            labels.append(single_label)

            pbar.update()

        return predictions, labels


def predict_single_file(model, faces, agg, device, verbose=False):

    """
    needs to be revisited
    """

    # formatting faces
    faces = torch.as_tensor(faces, dtype=torch.float32).to(device)

    # no gradient computations
    with torch.no_grad():
        # get per face prediction probabilities
        fake_prob = model.forward(faces)
        fake_prob = F.sigmoid(fake_prob)

        # aggregate prediction over all faces associated with file
        if agg == 'mean':
            fake_prob_file = torch.mean(torch.flatten(fake_prob))

        elif agg == 'max':
            fake_prob_file = torch.max(torch.flatten(fake_prob))

        else:
            assert False, "Invalid aggregation over probabilities specified"

        if verbose:
            print(f"Fake prob: {fake_prob_file}")

    return fake_prob_file.item(), fake_prob


def update_plots(model, iter_train_loss_list, iter_train_acc_list, iter_val_loss_list, iter_val_acc_list):
    """
    This function is used to update the online training metric plots.
    Args:
        iter_train_loss_list: A list containing tuples of global iter and train loss on the iter.
        iter_train_acc_list: A list containing tuples of global iter and train accuracy on the iter.
        iter_val_loss_list: A list containing tuples of global iter and validation loss on the iter.
        iter_val_acc_list: A list containing tuples of global iter and validation accuracy on the iter.
    Returns: None. but redraws plot in notebook.
    """

    '''
    The loss metric plot.
    '''
    # Convert list to numpy array for easier indexing.
    iter_train_loss_list = np.asanyarray(iter_train_loss_list)

    # Plot the train loss.
    model.metric_plots['plt1'].plot(smooth(iter_train_loss_list[:, 0], smoothing_steps=10),
                                    smooth(iter_train_loss_list[:, 1], smoothing_steps=10),
                                   color="royalblue", label="train")

    # In case there is a validation list given.
    if len(iter_val_loss_list) > 0:
        # Convert list to numpy array for easier indexing.
        iter_val_loss_list = np.asanyarray(iter_val_loss_list)
        # Plot the validation loss.
        model.metric_plots['plt1'].plot(smooth(iter_val_loss_list[:, 0], smoothing_steps=20),
                                        smooth(iter_val_loss_list[:, 1], smoothing_steps=20),
                                       color="orange", label="validation")

    # Set scale and labels.
    #model.metric_plots['plt1'].set_yscale("log")
    model.metric_plots['plt1'].set_ylabel("Loss")
    model.metric_plots['plt1'].set_xlabel("Iters")

    # In case the legend is not yet initialized do it.
    if not model.metric_plots['plt1_legend']:
        model.metric_plots['plt1_legend'] = model.metric_plots['plt1'].legend()

    '''
    The accuracy metric plot.
    '''

    # Convert list to numpy array for easier indexing.
    iter_train_acc_list = np.asanyarray(iter_train_acc_list)
    # Plot the train accuracy.
    model.metric_plots['plt2'].plot(iter_train_acc_list[:, 0],
                                    iter_train_acc_list[:, 1] * 100,
                                    color="royalblue", label="train")

    # In case there is a validation list given.
    if len(iter_val_acc_list) > 0:
        # Convert list to numpy array for easier indexing.
        iter_val_acc_list = np.asanyarray(iter_val_acc_list)
        # Plot the validation accuracy.
        model.metric_plots['plt2'].plot(iter_val_acc_list[:, 0],
                                        iter_val_acc_list[:, 1] * 100,
                                        color="orange", label="validation")

    # Set labels.
    model.metric_plots['plt2'].set_ylabel("Accuracy [%]")
    model.metric_plots['plt2'].set_xlabel("Iters")

    # In case the legend is not yet initialized do it.
    if not model.metric_plots['plt2_legend']:
        model.metric_plots['plt2_legend'] = model.metric_plots['plt2'].legend()

    '''
    Redraw the canvas.
    '''
    model.metric_plots['fig'].canvas.draw()


def plot_acc_loss(iter_train_loss_list,
                  iter_train_acc_list,
                  iter_val_loss_list,
                  iter_val_acc_list,
                  title,
                  save_to=None):

    """
    similar to update plots, but this one plots the accuracy/loss plots given the four lists
    """

    fig, ax = plt.subplots(2, 1, figsize=(9, 8), num=title)
    fig.suptitle(title, y=0.93)

    # Convert list to numpy array for easier indexing.
    iter_train_loss_list = np.asanyarray(iter_train_loss_list)

    # Plot the train loss.
    ax[0].plot(smooth(iter_train_loss_list[:, 0], smoothing_steps=50),
                      smooth(iter_train_loss_list[:, 1], smoothing_steps=50),
                      color="royalblue", label="train")

    # In case there is a validation list given.
    if len(iter_val_loss_list) > 0:
        # Convert list to numpy array for easier indexing.
        iter_val_loss_list = np.asanyarray(iter_val_loss_list)
        # Plot the validation loss.
        ax[0].plot(smooth(iter_val_loss_list[:, 0], smoothing_steps=15),
                          smooth(iter_val_loss_list[:, 1], smoothing_steps=15),
                          color="orange", label="validation")

    # Set scale and labels.
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Iters")

    # In case the legend is not yet initialized do it.
    ax[0].legend()

    '''
    The accuracy metric plot.
    '''

    # Convert list to numpy array for easier indexing.
    iter_train_acc_list = np.asanyarray(iter_train_acc_list)
    # Plot the train accuracy.
    ax[1].plot(iter_train_acc_list[:, 0],
               iter_train_acc_list[:, 1] * 100,
               color="royalblue", label="train")

    # In case there is a validation list given.
    if len(iter_val_acc_list) > 0:
        # Convert list to numpy array for easier indexing.
        iter_val_acc_list = np.asanyarray(iter_val_acc_list)
        # Plot the validation accuracy.
        ax[1].plot(iter_val_acc_list[:, 0],
                 iter_val_acc_list[:, 1] * 100,
                 color="orange", label="validation")

    # Set labels.
    ax[1].set_ylabel("Accuracy [%]")
    ax[1].set_xlabel("Iters")

    # In case the legend is not yet initialized do it.
    ax[1].legend()

    if save_to is not None:
        fig.savefig(save_to + '.svg', format='svg')
    else:
        plt.show()


def calculate_accuracy_np(labels, predictions, threshold=None):

    """
    needs to be revisited
    """

    assert len(labels) == len(predictions), "Labels and prediction vectors have different length"

    size = len(labels)

    if threshold:
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0
        predictions = predictions.astype(np.int8)

    else:
        predictions = np.round(predictions).astype(np.int8)

    # format labels
    labels = labels.astype(np.int8)
    acc = np.sum(predictions == labels)
    acc /= size

    return acc, predictions, labels


def calculate_accuracy(labels, predictions):

    """
    calculate accuracy of the prediction for batch
    """

    # Get the batch size.
    batch_size = predictions.shape[0]

    # Prepare predictions and labels.
    predictions = torch.flatten(predictions).round().type(torch.long)
    labels = torch.flatten(labels).type(torch.long)

    # Calculate accuracy
    acc = torch.sum(predictions == labels).item()
    acc /= batch_size

    return acc


def calculate_f1(labels, predictions, from_numpy=False):

    """
    calculate precision, recall and f1 score for predicted batch
    """

    if from_numpy:
        predictions = np.rint(predictions).astype(np.long)
        labels = labels.astype(np.long)
    else:
        predictions = torch.flatten(predictions).round().cpu().numpy().astype(np.long)
        labels = torch.flatten(labels).round().cpu().numpy().astype(np.long)

    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    return f1, precision, recall


def smooth(x, smoothing_steps):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    smoothed = (cumsum[smoothing_steps:] - cumsum[:-smoothing_steps]) / float(smoothing_steps)
    return smoothed


def plot_sample_images(ims, title=None, temporal=False, correct=None):

    title = torch.flatten(title).tolist()
    title = [np.round(prob, 2) for prob in title]

    if temporal:
        window_ims = []
        # make each tensor a list of stacked numpy arrays
        for i in range(len(ims)):
            stack = np.hstack((ims[i][0].permute(1, 2, 0),
                               ims[i][1].permute(1, 2, 0),
                               ims[i][2].permute(1, 2, 0),
                               ims[i][3].permute(1, 2, 0),
                               ims[i][4].permute(1, 2, 0)))

            stack = (stack - np.min(stack)) / np.ptp(stack)
            window_ims.append(stack)

        fig, axs = plt.subplots(nrows=len(window_ims), ncols=1, figsize=(8, 8))
        for idx, ax in enumerate(axs):
            ax.imshow(window_ims[idx])
            ax.grid(None)
            ax.set_title(title[idx])
            ax.axis('off')
    else:
        ims = [im.permute(1, 2, 0).numpy() for im in ims]
        ims = [(im - np.min(im)) / np.ptp(im) for im in ims]

        fig, axs = plt.subplots(nrows=1, ncols=len(ims), figsize=(16,4))
        for idx, ax in enumerate(axs):
            ax.imshow(ims[idx])
            ax.grid(None)
            ax.set_title(title[idx])
            ax.axis('off')

    background_color = 'green' if correct else 'red'
    fig.patch.set_facecolor(background_color)
    fig.patch.set_alpha(0.3)
    plt.tight_layout()
    plt.show()


def plot_sample_images_nopred(samples):
    """
    function that creates and plots NxN imagegrid of samples
    Args:
        samples (list): list of sample images to be plotted

    Returns: nothing but shows plot
    """

    # check that number of samples can be plotted as NxN grid
    assert int(np.sqrt(len(samples))) == np.sqrt(len(samples)), f"Cannot plot quadratic grid with sample length {len(samples)}"

    # determine grid size depending on number of samples
    nrows = ncols = int(np.sqrt(len(samples)))

    # initiate NxN image grid
    fig = plt.figure(figsize=(7, 7))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.1
    )

    # iterating over grid object returns axes
    for ax, im in zip(grid, samples):
        # only plot if face is present
        if im is not None:
            im = im.permute(1, 2, 0).numpy()
            im = (im - np.min(im)) / np.ptp(im)

            # plot
            ax.imshow(im)
            ax.grid(False)

    plt.show()


def initialize_model(model_class, path_to_model=None, pretrained=True, epoch=None,
                     device=None, dropout=0.5, gradcam_mode=False):

    assert path_to_model is not None or not pretrained, 'Trying to load pretrained model from path, but no path is specified'
    assert not pretrained or epoch is not None, 'To load pretrained model, specify a checkpoint'

    if model_class == 'effb3':
        model = EfficientNet.from_name('efficientnet-b3',
                                       override_params={
                                           'num_classes': 1,
                                           'dropout_rate': dropout,
                                           'drop_connect_rate': 0.2
                                       }
                                       )
        temporal = False
    elif model_class == 'effb0':
        model = EfficientNet.from_name('efficientnet-b0',
                                       override_params={
                                           'num_classes': 1,
                                           'dropout_rate': dropout,
                                           'drop_connect_rate': 0.2
                                       }
                                       )
        temporal = False
    elif model_class == 'meso4':
        model = MesoNet4()
        temporal = False
    elif model_class == 'effb3LSTM':
        model = EfficientLSTM(efficientnet_name='efficientnet-b3',
                              seq_len=5,
                              hidden_size=128,
                              device=device,
                              cnn_pretrained=False,
                              cnn_checkpoint=False,
                              gradcam_mode=gradcam_mode)
        temporal = True
    elif model_class == 'effb0LSTM':
        model = EfficientLSTM(efficientnet_name='efficientnet-b0',
                              seq_len=5,
                              hidden_size=256,
                              device=device,
                              cnn_pretrained=False,
                              cnn_checkpoint=False,
                              gradcam_mode=gradcam_mode)
        temporal = True
    elif model_class == 'meso4LSTM':
        model = Meso4LSTM(device=device, gradcam_mode=gradcam_mode)
        temporal = True
    else:
        assert False, 'Unknown model class specified. Valid options are: \n' \
                      'effb0, effb3, meso4, effb3LSTM, effb0LSTM, meso4LSTM'

    if pretrained:
        model.load_state_dict(torch.load(path_to_model + f'checkpoint-{epoch}.pth.tar')['model'])
    model = model.to(device)
    model.eval()

    return model, temporal


def get_model_path(model_class, model_name, pretrained=True):

    assert model_class in range(1, 5), 'Invalid model choice.'

    if model_class == 1:
        model_class = 'meso4'

    elif model_class == 2:
        model_class = 'effb0'

    elif model_class == 3:
        model_class = 'meso4LSTM'
    else:
        model_class = 'effb3LSTM'

    path_to_model = f'logs/checkpoints/{model_name}/'
    if not pretrained and not os.path.exists(path_to_model):
            os.makedirs(path_to_model)

    return model_class, path_to_model

