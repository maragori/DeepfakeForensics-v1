import os
import glob
import torch
import cv2
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import gc
import pandas as pd
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
from torchvision.utils import save_image
import fnmatch


class FaceDetection:
    """
    Facial detection pipeline class. Used to detect faces in the frames of a video file.
    """

    def __init__(self, detector, device,  n_frames=16, batch_size=2, resize=None):
        """
        Constructor for FacialDetection class

        Args:
            detector (): face detector to use
            device: cuda device
            n_frames (): the number of frames that should be extracted from the video
                         frames will be evenly spaced
                          default is None, which results in all frames being extracted
            batch_size (): batch size to use with face detector, default is 16
            resize (): fraction by which frames are resized from original to face detection
                       <1: downsampling
                       >1: upsampling
                       default is None
        """

        self.detector = detector
        self.device = device
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize

    def __call__(self, file, temporal=False):
        """
        This methods extracts frames and faces from a mp4

        Args:
            file (): path + filename of the video

        Returns: list of face images
        """

        # read video from file
        v_cap = cv2.VideoCapture(file)

        # get frame count of video
        v_frames = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # create indices for the specified amount n_frames (random window)

        if not self.n_frames:
            # if number of frames not specified, create index for all frames in video
            frame_indices = np.arange(0, v_frames)
        else:
            if v_frames < self.n_frames:
                print(f"File has less than {self.n_frames} frames. Skipping...")
                return None
            # if number of frames is specified, create n_frames equidistant indices
            if not temporal:
                frame_indices = np.arange(0, v_frames, v_frames/self.n_frames).astype(int)
            # for temporal model, create n_frames/5 equidistant frame windows of length 5
            else:
                start_indices = np.arange(0, v_frames-5, v_frames / (self.n_frames/5.)).astype(int)
                window_list = [[idx, idx+1, idx+2, idx+3, idx+4] for idx in list(start_indices)]
                frame_indices = list(np.array(window_list).flat)

        # init lists to fill with frames and faces
        faces = []
        # batch list for frames
        frame_batch = []

        # Loop through frames
        for frame_index in range(v_frames):

            # grab_frame
            _ = v_cap.grab()

            # if frame is in frame_indices
            if frame_index in frame_indices:

                # Load frame
                success, frame = v_cap.retrieve()

                # if retrieve fails, pass
                if not success:
                    continue

                # colors to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # extract from object
                frame = Image.fromarray(frame)

                # append to frame list
                frame_batch.append(frame)

                del frame
                gc.collect()

                # When batch is full or no more indices
                if len(frame_batch) % self.batch_size == 0 or frame_index == frame_indices[-1]:
                    # detect faces in frame list, append batch to face list
                    # note, if no face is present, None is appended
                    faces.extend(self.detector(frame_batch))
                    # reset frame list
                    frame_batch = []

        # release device resource
        v_cap.release()

        return faces


def only_read_CDF_from_path(files, labels_per_face, labels_per_file, label,
                            path_to_store_faces, path_to_store_faces_aug,  face_detection, csv_file_name,
                            min_face_cutoff=64, temporal=False, log_plots=False):

    with torch.no_grad():

        # repeat for all real files
        for path in tqdm(files[:]):

            # get filename
            filename = path.split('\\')[-1]

            # Detect all faces that occur in the video
            faces = face_detection(path, temporal)

            if faces:
                faces = [f for f in faces if f is not None]
                if len(faces) < min_face_cutoff:
                    continue
            else:
                continue

            print(f"[# extracted faces: {len(faces)}]")

            # augment face set
            faces_aug = add_noise_to_faces(faces)

            # for temporal data, create face window tensors of length 5
            if temporal:
                if not log_plots:
                    # if we don't want to plot sequences:
                    # because we store tensors to disk, we can already normalize them
                    faces = [face.type(torch.int32) / 255.0 for face in faces]
                    faces = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(face) for face in
                             faces]

                    faces_aug = [face.type(torch.int32) / 255.0 for face in faces_aug]
                    faces_aug = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(face) for face
                                 in faces_aug]

                face_windows = []
                face_windows_aug = []
                for i in np.arange(0, len(faces), 5):
                    try:
                        face_windows.append(torch.stack([faces[i],
                                                    faces[i+1],
                                                    faces[i+2],
                                                    faces[i+3],
                                                    faces[i+4]]))

                        face_windows_aug.append(torch.stack([faces_aug[i],
                                                            faces_aug[i+1],
                                                            faces_aug[i+2],
                                                            faces_aug[i+3],
                                                            faces_aug[i+4]]))

                    except IndexError:
                        print(f"Incomplete window indices at file {filename}")
                        pass

                print(f"[# extracted face windows: {len(face_windows)}]")

                faces = face_windows
                faces_aug = face_windows_aug

                # for logging or debugging purposes, save plots
                if log_plots:
                    create_window_plots(face_windows, filename + '_temp', save=True)
                    create_window_plots(face_windows_aug, filename + '_aug_temp', save=True)

            # for dataset creation, save images and labels to disk
            if not log_plots:

                # add label dict entries corresponding to the faces
                labels_per_face, labels_per_file = add_to_label_dicts(faces, filename, label,
                                                                      labels_per_face, labels_per_file)
                # store labels to disk
                store_labels(labels=labels_per_face, filename=csv_file_name + "_labels_per_face")
                store_labels(labels=labels_per_file, filename=csv_file_name + "_labels_per_file")

                # write raw faces to disk
                store_face_tensors(faces, path_to_store_faces, filename, temporal)

                # write augmented faces to disk
                store_face_tensors(faces_aug, path_to_store_faces_aug, filename, temporal)

    return labels_per_face, labels_per_file


def create_window_plots(face_windows, filename=None, save=False, unnormalize=False):

    # if plot should be saved, filename must be specified
    if save:
        assert filename, 'Filename for window plot has to be specified'

    fig, ax = plt.subplots(figsize=(40, 40))

    windows = []
    for i in range(len(face_windows)):
        stack = np.hstack((face_windows[i][0].permute(1, 2, 0),
                        face_windows[i][1].permute(1, 2, 0),
                        face_windows[i][2].permute(1, 2, 0),
                        face_windows[i][3].permute(1, 2, 0),
                        face_windows[i][4].permute(1, 2, 0)))

        if unnormalize:
            stack = (stack - np.min(stack)) / np.ptp(stack)
        windows.append(stack)

    stack = np.vstack(tuple(windows))
    if unnormalize:
        ax.imshow(stack)
    else:
        ax.imshow(stack / 255)

    ax.axis('off')
    if save:
        plt.savefig(f'logs/preprocess_plots/{filename}.svg', format='svg')
    plt.show()


def get_CDF_per_folder(path_to_data, path_to_store_faces, path_to_store_faces_aug,
                       face_detection, label, csv_file_name, sample=False, size=2,
                       min_face_cutoff=64, temporal=False, log_plots=False):

    videos = glob.glob(os.path.join(path_to_data, '*.mp4'))
    print(f"Extracting  faces from {len(videos)} {label} files")

    # for dev purposes. get only first 20 samples
    if sample:
        videos = videos[:int(size)]

    # label dict
    labels_per_face = {}
    labels_per_file = {}

    label_per_face, labels_per_file = only_read_CDF_from_path(videos,
                                                              labels_per_face,
                                                              labels_per_file,
                                                              label,
                                                              path_to_store_faces,
                                                              path_to_store_faces_aug,
                                                              face_detection,
                                                              csv_file_name,
                                                              min_face_cutoff,
                                                              temporal,
                                                              log_plots)

    return labels_per_face, labels_per_file


def add_to_label_dicts(faces, filename, label, labels_per_face, labels_per_file):

    # overall label
    labels_per_file[filename] = 1 if label == 'Fake' else 0

    for idx, _ in enumerate(faces):
        # label per face
        labels_per_face[f"{str(idx).zfill(3)}_{filename}"] = 1 if label == 'Fake' else 0

    return labels_per_face, labels_per_file


def store_face_tensors(faces, path_to_store, filename, temporal=False):

    """
    this will need to be edited for the temporal version
    """

    # here we also unsqueeze  the tensors
    if not os.path.exists(path_to_store):
        os.makedirs(path_to_store)

    for idx, im in enumerate(faces):

        if temporal:
            # add batch dim
            im = im.unsqueeze(0)

            # store each face tensor with idx
            torch.save(im, path_to_store + f"{str(idx).zfill(3)}_{filename}.pt")

        else:
            save_image(im/255, path_to_store + f"{str(idx).zfill(3)}_{filename}.jpg")


def store_labels(labels, path_to_store='Labels/', filename=''):

    if not os.path.exists(path_to_store):
        os.makedirs(path_to_store)

    label_df = pd.DataFrame.from_dict(labels, orient='index').reset_index()
    label_df.columns = ['file', 'label']
    label_df.to_csv(path_to_store + filename + '.csv')


def combine_labels(path, file1, file2, file3=None):
    """
    combine labels of celeb_real, youtube_real and celeb_synthesis
    """
    if file3 is None:
        dfs = [pd.read_csv(path + file, index_col=0) for file in [file1, file2]]
    else:
        dfs = [pd.read_csv(path + file, index_col=0) for file in [file1, file2, file3]]

    full = pd.concat(dfs, ignore_index=True)

    return full


def to_numpy_hwc(im):
    """
    convert from CxHxW torch tensor to HxWxC numpy array
    Args:
        im: CxHxW torch tensor

    Returns: HxWxC numpy array

    """
    return im.permute(1, 2, 0).int().numpy().astype('uint8')


def to_torch_chw(im):
    """
    inverse to the transform function
    Args:
        im: HxWxC numpy array

    Returns: CxHxW torch tensor

    """
    return torch.from_numpy(im).permute(2, 0, 1).type(torch.float32)


def augment_ims(ims, augmentations):
    """
    define the image augmentations that are added in augmented dataset
    - random changes in jpeg compression
    - random additive gaussian noise
    - random changes to brightness
    - random changes to hue and saturation
    - random changes to contrast
    Args:
        ims: list of image arrays HxWxC (RGB)

    Returns: list of augmented image arrays (RGB)

    """
    # make deterministic, because we want all images from one file to undergo the same augmentations
    augmentations_det = augmentations.to_deterministic()

    ims = [augmentations_det.augment_image(im) for im in ims]

    return ims


def add_noise_to_faces(ims):
    """
    given a list of image tensors of shape (1, 3, 224, 224), augment each image and return the augmented list
    Args:
        ims: list of image tensors (1, 3, 224, 224)

    Returns: list of augmented image arrays (1, 3, 224, 224)

    """

    # define augmentations
    seq = iaa.Sequential([
        iaa.JpegCompression(compression=(40, 80)),
        iaa.AdditiveGaussianNoise(scale=(4, 9)),
        iaa.GaussianBlur(sigma=1.2),
        iaa.MultiplyAndAddToBrightness(mul=(0.6, 1.5), add=(-10, 10)),
        iaa.AddToHueAndSaturation((-15, 15), per_channel=True),
        iaa.GammaContrast((0.6, 1.6))
    ])

    # transform images into the correct format for augmentation: (224, 224, 3)
    ims = [to_numpy_hwc(im) for im in ims]

    # augment images
    ims = augment_ims(ims, seq)

    # visual check for debugging
    #ia.imshow(np.hstack(ims[:4]))
    #plt.show()

    ims = [to_torch_chw(im) for im in ims]
    return ims


def plot_sample_images(samples):
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
        axes_pad=0.0
    )

    # iterating over grid object returns axes
    for ax, im in zip(grid, samples):
        # only plot if face is present
        if im is not None:
            im = im[0].permute(1, 2, 0).numpy()
            im = (im - np.min(im)) / np.ptp(im)

            # plot
            ax.imshow(im)
            ax.grid(False)
            ax.axis('off')

    plt.show()


def extract_single_actor_clips(path_to_videos, face_detector, n_frames=8, threshold=1.2,
                               path_to_store_single_actor='DFDC/single_actor2/',
                               path_to_store_multiple_actor='DFDC/multiple_actor/'):

    """
    used to extract the subset of DFDC clips that feature a single actor
    algorithm has the occasional false positive and false negative, so manual checking afterwards is required
    """

    videos = glob.glob(os.path.join(path_to_videos, '*.mp4'))

    # create directories
    if not os.path.exists(path_to_store_single_actor):
        os.makedirs(path_to_store_single_actor)

    if not os.path.exists(path_to_store_multiple_actor):
        os.makedirs(path_to_store_multiple_actor)

    # repeat for all real files
    for path in tqdm(videos[:]):

        # get filename
        filename = path.split('\\')[-1]
        print(f"File: {filename}")

        # read video from file
        v_cap = cv2.VideoCapture(path)

        # get frame count of video
        v_frames = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # extract n_frames equidistant frames
        if v_frames < n_frames:
            print(f"File has less than {n_frames} frames. Skipping...")
            continue

        # if number of frames is specified, create n_frames indices
        frame_indices = np.arange(0, v_frames, v_frames / n_frames).astype(int)

        sum = 0
        counter = 0

        # Loop through frames
        for frame_index in range(v_frames):

            # grab_frame
            _ = v_cap.grab()

            # if frame is in frame_indices
            if frame_index in frame_indices:
                # Load frame
                success, frame = v_cap.retrieve()

                if not success:
                    continue

                face = face_detector(frame)

                if face is None:
                    continue

                n = face.size(0)
                sum += n
                counter += 1

        # release device resource
        v_cap.release()

        try:
            avg = sum / counter
            print(f"Average: {avg:.3f} extracted from {counter} frames")
        except ZeroDivisionError:
            print("Could not extract any frames. Skipping...")
            continue

        # if average of detected frames per face is lower than threshold store in single actor folder
        if avg < threshold:
            os.rename(path, path_to_store_single_actor + filename)
        # else store in multiple actor folder
        else:
            os.rename(path, path_to_store_multiple_actor + filename)


def balance_labels_per_file(labels_per_file):
    """
    balance labels by oversampling the minority class, real files

    > THIS FUNCTION IS NOT NEEDED ANYMORE

    """
    print('Balancing the per file labels according to split...')

    # find real and fake files for the split
    real_files = labels_per_file.loc[labels_per_file['label'] == 0]
    fake_files = labels_per_file.loc[labels_per_file['label'] == 1]

    # oversampling factor
    proportion = int(len(fake_files) / len(real_files))

    print(f"Split contained {len(real_files)} real files and {len(fake_files)} fake files")
    print(f"Oversampling real files by a factor of {proportion}")

    # oversample real files by factor
    temp = pd.DataFrame(np.repeat(real_files.values, proportion, axis=0))
    temp.columns = labels_per_file.columns

    # concat oversampled real files with fake files
    labels_per_file_balanced = pd.concat([labels_per_file, temp], ignore_index=True)

    # reshuffle dataframe
    labels_per_file_balanced = labels_per_file_balanced.sample(frac=1).reset_index(drop=True)

    print(f"Balanced dataset has length {len(labels_per_file_balanced)}")
    print('Done')

    return labels_per_file_balanced


def get_face_labels_from_file_labels(labels_per_file, labels_per_face):
    """
    given a pandas df containing labels on file level, derive the corresponding face level label df

    Args:
        labels_per_file:

    Returns:

    """

    files = list(labels_per_file.file)
    faces = list(labels_per_face.file)

    labels_per_face_split = {}
    print('Deriving the correct face labels for the split...')
    for file in tqdm(files):

        # find the matching face files for the filename
        faces_per_file = fnmatch.filter(faces, "???_" + file)

        # get label for file
        label = int(labels_per_file.loc[labels_per_file['file'] == file]['label'])

        # create label per face entry for each face
        for face in faces_per_file:

            # fix .pt in file name, remove when new data
            #face = '.'.join(face.split('.')[:-1])

            labels_per_face_split[face] = label

    # to pandas
    labels_per_face_split = pd.DataFrame.from_dict(labels_per_face_split, orient='index').reset_index()
    labels_per_face_split.columns = labels_per_file.columns

    return labels_per_face_split


def get_DFDC_training_data(path_to_data, path_to_store_faces, path_to_store_faces_aug,
                           face_detection_real, face_detection_fake, csv_file_name,
                           min_face_cutoff_real, min_face_cutoff_fake, temporal=False, log_plots=False):


    # get path to all training videos
    all_train_videos = glob.glob(os.path.join(path_to_data, '*.mp4'))

    # Get path of metadata.json
    metadata_path = path_to_data + 'DFDC_metadata.csv'

    # Get metadata
    metadata = pd.read_csv(metadata_path, index_col=1).drop('Unnamed: 0', axis=1)['label'].to_dict()

    # initialize label dicts
    labels_per_face = {}
    labels_per_file = {}

    with torch.no_grad():
        # repeat for all files
        for path in tqdm(all_train_videos):

            # get filename
            filename = path.split('\\')[-1]

            # check label
            label = 'Fake' if metadata[filename] == 'FAKE' else 'Real'
            min_face_cutoff = min_face_cutoff_fake if label == 'Fake' else min_face_cutoff_real

            # Detect all faces that occur in the video, number of frames to use depends on the label
            if label == 'Fake':
                faces = face_detection_fake(path)
            elif label == 'Real':
                faces = face_detection_real(path)
            else:
                assert False, f"Invalid type of label encountered: {label}"

            if faces:
                faces = [f for f in faces if f is not None]
                if len(faces) < min_face_cutoff:
                    continue
            else:
                continue

            print(f"[extracted {len(faces)} faces from {label} file {filename}]")

            # augment face set
            faces_aug = add_noise_to_faces(faces)


            if not log_plots:
                # normalize
                faces = [face.type(torch.int32) / 255.0 for face in faces]
                faces = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(face) for face in
                         faces]

                faces_aug = [face.type(torch.int32) / 255.0 for face in faces_aug]
                faces_aug = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(face) for face
                             in faces_aug]

            # for temporal data, create face window tensors of length 5
            if True:
                face_windows = []
                face_windows_aug = []
                for i in np.arange(0, len(faces), 5):
                    try:
                        face_windows.append(torch.stack([faces[i],
                                                         faces[i + 1],
                                                         faces[i + 2],
                                                         faces[i + 3],
                                                         faces[i + 4]]))

                        face_windows_aug.append(torch.stack([faces_aug[i],
                                                             faces_aug[i + 1],
                                                             faces_aug[i + 2],
                                                             faces_aug[i + 3],
                                                             faces_aug[i + 4]]))
                    except IndexError:
                        print(f"Incomplete window indices at file {filename}")
                        pass

                print(f"[# extracted face windows: {len(face_windows)}]")
                faces = face_windows
                faces_aug = face_windows_aug

                if log_plots:
                    create_window_plots(face_windows, 'DFDC/' + filename)
                    create_window_plots(face_windows_aug, 'DFDC/' + filename + '_aug')

            if not log_plots:
                # add label dict entries corresponding to the faces
                labels_per_face, labels_per_file = add_to_label_dicts(faces, filename, label,
                                                                      labels_per_face, labels_per_file)
                # store labels to disk
                store_labels(labels=labels_per_face, filename=csv_file_name + "DFDC_labels_per_face")
                store_labels(labels=labels_per_file, filename=csv_file_name + "DFDC_labels_per_file")

                # write raw faces to disk
                store_face_tensors(faces, path_to_store_faces, filename)

                # write augmented faces to disk
                store_face_tensors(faces_aug, path_to_store_faces_aug, filename)

        return labels_per_face, labels_per_file