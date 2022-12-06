import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
import torch


class GRFDataset(Dataset):
    def __init__(self, images, truths, img_transform=None, truth_transform=None, z=0, z_type='zero_out'):
        self.images = images
        self.truths = truths
        self.img_transform = img_transform
        self.truth_transform = truth_transform
        self.z = z
        self.z_type = z_type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        truth = self.truths[idx]
        
        if self.img_transform:
            image = self.img_transform(image)
        if self.truth_transform:
            truth = self.truth_transform(truth)

        if self.z > 0:
            if self.z_type == 'crop':
                image = transforms.functional.crop(image, top=self.z, left=self.z, height=image.shape[-1]-2*self.z, width=image.shape[-1]-2*self.z)
                truth = transforms.functional.crop(truth, top=self.z, left=self.z, height=truth.shape[-1]-2*self.z, width=truth.shape[-1]-2*self.z)
                # image, mask = image[self.z:-self.z, self.z:-self.z], mask[self.z:-self.z, self.z:-self.z]
            elif self.z_type == 'zero_out':
                msk = torch.zeros(image.size())
                msk[:,self.z:-self.z, self.z:-self.z] = 1.0

                image *= msk
                truth *= msk

        return image, truth



def prepare_data(datafile, size=None, shuffle=False, z=0, normalization='standard'):
    with open(datafile, 'rb') as f:
        truths = np.load(f)
        inputs = np.load(f)

    if size is None:
        size = len(inputs)

    if shuffle is True:
        p = np.random.permutation(len(inputs))
        inputs, truths = inputs[p], truths[p]

    # Select required data
    inputs  = inputs[:size]
    truths = truths[:size]


    # Define Normalization Transforms
    if normalization == 'standard':
        # Calculate mean and std for standardization
        input_mean, input_std = np.mean(inputs), np.std(inputs)
        truth_mean, truth_std = np.mean(truths), np.std(truths)


        # Define Transforms
        input_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=input_mean, std=input_std)
        ])

        truth_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=truth_mean, std=truth_std)
        ])


        # Define Inverse Transforms
        inv_input_trans = transforms.Compose([
            transforms.Normalize(mean=0.0, std=1.0/input_std),
            transforms.Normalize(mean=-input_mean, std=1.0)
        ])

        inv_truth_trans = transforms.Compose([
            transforms.Normalize(mean=0.0, std=1.0/truth_std),
            transforms.Normalize(mean=-truth_mean, std=1.0)
        ])


    elif normalization == 'log_normal':
        # Calculate min and max for normalization
        input_min, input_max = np.min(inputs), np.max(inputs)
        truth_min, truth_max = np.min(truths), np.max(truths)


        # Define Transforms
        input_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=input_min, std=(input_max - input_min)),
            transforms.Lambda(torch.log)
        ])

        truth_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=truth_min, std=(truth_max - truth_min)),
            transforms.Lambda(torch.log)
        ])


        # Define Inverse Transforms
        inv_input_trans = transforms.Compose([
            transforms.Lambda(torch.exp),
            transforms.Normalize(mean=0.0, std=1.0/(input_max - input_min)),
            transforms.Normalize(mean=-input_min, std=1.0)
        ])

        inv_truth_trans = transforms.Compose([
            transforms.Lambda(torch.exp),
            transforms.Normalize(mean=0.0, std=1.0/(truth_max - truth_min)),
            transforms.Normalize(mean=-truth_min, std=1.0)
        ])

    
    elif normalization is None:
        # Define Transforms
        input_trans = transforms.ToTensor()
        truth_trans = transforms.ToTensor()

        # Define Inverse Transforms
        inv_input_trans = transforms.Normalize(mean=0.0, std=1.0)
        inv_truth_trans = transforms.Normalize(mean=0.0, std=1.0)


    transdict = {
        'input_transform': input_trans,
        'truth_transform': truth_trans,
        'inv_input_transform': inv_input_trans,
        'inv_truth_transform': inv_truth_trans
    }


    # Create Datasets
    train_data = GRFDataset(inputs, truths, img_transform=input_trans, truth_transform=truth_trans, z=z)

    return train_data, transdict



# def prepare_data_train_and_test(datafile, train_size=None, train_frac=None):
#     with open(datafile, 'rb') as f:
#         all_masks = np.load(f)
#         all_images = np.load(f)

#     p = np.random.permutation(len(all_masks))
#     all_masks, all_images = all_masks[p], all_images[p]

#     if train_size is not None:
#         pass
#     elif train_frac is not None:
#         train_size = int(train_frac*len(all_masks))
#     else:
#         raise ValueError("train_prop and train_frac can''t be both None")

#     train_images, test_images = all_images[:train_size], all_images[train_size:]
#     train_masks, test_masks = all_masks[:train_size], all_masks[train_size:]


#     # Calculate mean and std for standardization
#     train_img_m, train_img_s = np.mean(train_images), np.std(train_images)
#     train_msk_m, train_msk_s = np.mean(train_masks), np.std(train_masks)

#     test_img_m, test_img_s = np.mean(test_images), np.std(test_images)
#     test_msk_m, test_msk_s = np.mean(test_masks), np.std(test_masks)


#     # Define Transforms
#     train_img_trans = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(train_img_m), std=(train_img_s))
#     ])

#     train_msk_trans = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(train_msk_m), std=(train_msk_s))
#     ])

#     test_img_trans = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(test_img_m), std=(test_img_s))
#     ])

#     test_msk_trans = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(test_msk_m), std=(test_msk_s))
#     ])


#     # Create Datasets
#     train_data = GRFDataset(train_images, train_masks, img_transform=train_img_trans, msk_transform=train_msk_trans)
#     test_data = GRFDataset(test_images, test_masks, img_transform=test_img_trans, msk_transform=test_msk_trans)

#     return train_data, test_data

