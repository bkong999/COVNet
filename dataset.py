import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
import SimpleITK as sitk
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms


class NCovDataset(data.Dataset):
    def __init__(self, root_dir, stage='train'):
        super().__init__()
        self.root_dir = root_dir
        self.stage = stage
        assert stage in ['train', 'val', 'test']

        if stage == 'train':
            split_file = 'train.csv'
        elif stage == 'val':
            split_file = 'val.csv'
        elif stage == 'test':
            # We just assume validation set is the same as test set
            split_file = 'val.csv'

        df = pd.read_csv(os.path.join(root_dir, split_file),
                              converters={'case': str, 'label': int})
        df = df.set_index('case')
        self.case_ids = list(df.index)
        self.labels = df['label'].values.tolist()

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        fn = os.path.join(self.root_dir, self.case_ids[index], 'masked_ct.nii')
        image = sitk.ReadImage(fn)
        array = sitk.GetArrayFromImage(image)

        mask_fn = os.path.join(self.root_dir, self.case_ids[index],
                               'mask.nii.gz')
        mask_image = sitk.ReadImage(mask_fn)
        mask = sitk.GetArrayFromImage(mask_image)

        array, mask = array[None, ...], mask[None, ...]
        if self.stage == 'train':
            # Default randomly mirroring the second and third axes
            array, mask = spatial_transforms.augment_mirroring(
                array, sample_seg=mask, axes=(1, 2))
        array, mask = array[0], mask[0]

        ######################################################
        #  Preprocessing for both train and validation data  #
        ######################################################
        min_value, max_value = -1250, 250
        np.clip(array, min_value, max_value, out=array)
        array = (array - min_value) / (max_value - min_value)

        # data should be a numpy array with shape [x, y, z] or [c, x, y, z]
        # seg should be a numpy array with shape [x, y, z]
        full_channel = np.stack([array, array, array])

        if self.stage == 'train':
            full_channel, mask = self.do_augmentation(full_channel, mask)
        else:
            mask = mask[None, ...]

        # remove the noise in the non-lung regions
        mask = mask[0]
        full_channel[0][mask == 0] = 0
        full_channel[1][mask == 0] = 0
        full_channel[2][mask == 0] = 0
        label = self.labels[index]
        full_channel = torch.FloatTensor(full_channel).permute((1, 0, 2, 3))

        return full_channel, label, self.case_ids[index]

    def do_augmentation(self, array, mask):
        """Augmentation for the training data.

        :array: A numpy array of size [c, x, y, z]
        :returns: augmented image and the corresponding mask

        """
        # normalize image to range [0, 1], then apply this transform
        patch_size = np.asarray(array.shape)[1:]
        augmented = noise_transforms.augment_gaussian_noise(
            array, noise_variance=(0, .015))

        # need to become [bs, c, x, y, z] before augment_spatial
        augmented = augmented[None, ...]
        mask = mask[None, None, ...]
        r_range = (0, (3 / 360.) * 2 * np.pi)
        cval = 0.

        augmented, mask = spatial_transforms.augment_spatial(
            augmented, seg=mask, patch_size=patch_size,
            do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
            do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
            do_scale=True, scale=(.9, 1.1),
            border_mode_data='constant', border_cval_data=cval,
            order_data=3,
            p_el_per_sample=0.5,
            p_scale_per_sample=.5,
            p_rot_per_sample=.5,
            random_crop=False
        )
        mask = mask[0]
        return augmented[0], mask

    def make_weights_for_balanced_classes(self):
        """Making sampling weights for the data samples
        :returns: sampling weigghts for dealing with class imbalance problem

        """
        n_samples = len(self.labels)
        unique, cnts = np.unique(self.labels, return_counts=True)
        cnt_dict = dict(zip(unique, cnts))

        weights = []
        for label in self.labels:
            weights.append(n_samples / float(cnt_dict[label]))
        return weights
