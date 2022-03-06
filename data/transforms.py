import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2


class DomainTransforms:
    '''
        Base class for DomainTransforms, perform augs only for labels in "filter_labels"
    '''
    def __init__(self, filter_labels, transforms):

        self.filter_labels = filter_labels
        self.transforms = transforms

    def __call__(self, image, y):

        if y in self.filter_labels:
            sample = self.transforms(image=image)
            return sample['image']
        return image

class NightTransforms(DomainTransforms):
    def __init__(self, filter_labels):

        transforms = A.Compose([
                A.ToGray(p=1.0)
            ], p=1.0)

        super().__init__(filter_labels, transforms=transforms)

class BirdsTransforms(DomainTransforms):
    def __init__(self, filter_labels):

        transforms = A.Compose([
                A.Rotate(limit=(90, 90), p=0.3),
            ], p=1.0)

        super().__init__(filter_labels, transforms=transforms)


def create_transforms(opt, mode, post_transforms=None):

    assert mode in ['train', 'val'], 'Mode must be train or val'

    if mode == 'val':
        transforms =  A.Compose([
                A.Resize(height=opt.resolution, width=opt.resolution, p=1.0),
                A.Normalize(),
            ], p=1.0)
    elif mode == 'train':

        pre = []

        if not opt.no_augs:
            pre.append(A.HorizontalFlip(p=opt.hor_flip))

            pre.append(
                    A.OneOf([A.GaussNoise(p=opt.gauss_noise),
                             A.ISONoise(p=opt.iso_noise),
                ], p=1.0)
            )

            pre.append(A.RandomBrightness(limit=opt.brightness, p=0.25))

            pre.append(A.RandomContrast(limit=opt.contrast, p=0.25))

            pre.append(A.augmentations.transforms.ColorJitter(p=opt.color_jitter))

            pre.append(A.augmentations.transforms.CLAHE(p=opt.clahe))

            pre.append(A.Affine(
                scale=(1.0 - opt.scale, 1.0 + opt.scale),
                translate_percent=(0.0, opt.translate),
                rotate=(-opt.angle, opt.angle),
                shear=(-opt.shear, opt.shear)
            ))


        resize_tr = [A.Resize(height=opt.resolution, width=opt.resolution, p=1.0)]
        if opt.resize_strategy == 'resized_crop':
            resize_tr = [A.RandomResizedCrop(height=opt.resolution, width=opt.resolution, scale=(0.9, 1.0), p=1.0)]
        elif opt.resize_strategy == 'resize&crop':
            up_size = int(opt.resolution * 1.2)
            resize_tr = [A.Resize(height=up_size, width=up_size, p=1.0),
                        A.RandomCrop(height=opt.resolution, width=opt.resolution, p=1.0)
            ]

        post = [
            *resize_tr,
            A.Normalize(), # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ]

        transforms = A.Compose(pre + post, p=1.0)

        if post_transforms:
            transforms.transforms.transforms.extend(post_transforms)
  
    return transforms
