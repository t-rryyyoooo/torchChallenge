import SimpleITK as sitk
import numpy as np
from utils import *

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)

        return image, label

class ReadImage(object):
    def __call__(self, image_file, label_file):
        image = sitk.ReadImage(image_file)
        label = sitk.ReadImage(label_file)

        return image, label

class AffineTransform(object):
    def __init__(self, translate_range, rotate_range, shear_range, scale_range, bspline=None):
        self.translate_range = translate_range
        self.rotate_range = rotate_range
        self.shear_range = shear_range
        self.scale_range = scale_range
        self.bspline = None

    def __call__(self, image, label):
        """
        image : 256 * 256 * x
        label : 256 * 256 * 1
        """
        parameters = makeAffineParameters(image, self.translate_range, self.rotate_range, self.shear_range, self.scale_range)
        affine = makeAffineMatrix(*parameters)

        minval = getMinimumValue(image)
        transformed_image = transforming(image, self.bspline, affine, sitk.sitkLinear, minval)

        transformed_label = transforming(label, self.bspline, affine, sitk.sitkNearestNeighbor, 0)

        return transformed_image, transformed_label

class GetArrayFromImage(object):
    def __call__(self, image, label):
        imageArray = sitk.GetArrayFromImage(image)
        labelArray = sitk.GetArrayFromImage(label)

        if image.GetDimension() != 3:
            imageArray = imageArray[..., np.newaxis]

        labelArray = np.squeeze(labelArray)
        imageArray = imageArray.transpose((2, 0, 1))

        return imageArray, labelArray









