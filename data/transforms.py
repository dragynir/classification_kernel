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

def TTA_5_cropps(image, resize_size=1024, target_size=256):

    target_shape = (target_size, target_size, 3)

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],
              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],]

    images = []
    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + target_w >= resize_size:
            x = resize_size - target_w-1
        if y + target_h >= resize_size:
            y = resize_size - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]
        image_ = zeros.copy()
        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]).squeeze())

    return images

def create_transforms(opt, mode, post_transforms=None):

    assert mode in ['train', 'val'], 'Mode must be train or val'

    if mode == 'val':
        resize_size = opt.milti_input_resize if opt.multi_image else opt.resolution
        transforms =  A.Compose([
                A.Resize(height=resize_size, width=resize_size, p=1.0),
                A.Normalize(),
            ], p=1.0)
    elif mode == 'train':

        pre = []

        if not opt.no_augs:
            pre.append(A.HorizontalFlip(p=opt.hor_flip))

            # pre.append(
            #         A.OneOf([A.GaussNoise(p=opt.gauss_noise),
            #                  A.ISONoise(p=opt.iso_noise),
            #     ], p=1.0)
            # )

            # pre.append(A.RandomBrightness(limit=opt.brightness, p=0.25))

            # pre.append(A.RandomContrast(limit=opt.contrast, p=0.25))

            # pre.append(A.augmentations.transforms.ColorJitter(p=opt.color_jitter))

            # pre.append(A.augmentations.transforms.CLAHE(p=opt.clahe))
            # pre.append(A.Affine(
            #     scale=(1.0 - opt.scale, 1.0 + opt.scale),
            #     translate_percent=(0.0, opt.translate),
            #     rotate=(-opt.angle, opt.angle),
            #     shear=(-opt.shear, opt.shear)
            # ))

        if opt.multi_image:
            resize_tr = [A.Resize(height=opt.milti_input_resize, width=opt.milti_input_resize, p=1.0)]
        else:
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
