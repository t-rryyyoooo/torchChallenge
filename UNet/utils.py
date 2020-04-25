import SimpleITK as sitk
import numpy as np
from pathlib import Path

def separateData(dataset_path, criteria, phase): 
    dataset = []
    for number in criteria[phase]:
        data_path = Path(dataset_path) / ("case_00" + number) 

        image_list = data_path.glob("image*")
        label_list = data_path.glob("label*")
        
        image_list = sorted(image_list)
        label_list = sorted(label_list)

        for img, lab in zip(image_list, label_list):
            dataset.append((str(img), str(lab)))

    return dataset


def makeAffineParameters(image, translate, rotate, shear, scale):
    dimension = image.GetDimension()
    translation = np.random.uniform(-translate, translate, dimension)
    rotation = np.radians(np.random.uniform(-rotate, rotate))
    shear = np.random.uniform(-shear, shear, 2)
    scale = np.random.uniform(1 - scale, 1 + scale)
    center = (np.array(image.GetSize()) * np.array(image.GetSpacing()) / 2)[::-1]
    
    return [translation, rotation, scale, shear, center]

def makeAffineMatrix(translate, rotate, scale, shear, center):
    a = sitk.AffineTransform(3)

    a.SetCenter(center)
    a.Rotate(1, 0, rotate)
    a.Shear(1, 0, shear[0])
    a.Shear(0, 1, shear[1])
    a.Scale((scale, scale, 1))
    a.Translate(translate)

    return a

def transforming(image, bspline, affine, interpolator, minval):
    # B-spline transformation
    if bspline is not None:
        transformed_b = sitk.Resample(image, bspline, interpolator, minval)

    # Affine transformation
        transformed_a = sitk.Resample(transformed_b, affine, interpolator, minval)

    else:
        transformed_a = sitk.Resample(image, affine, interpolator, minval)

    return transformed_a

def getMinimumValue(image):
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(image)
    return minmax.GetMinimum()


