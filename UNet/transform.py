from .preprocessing import *

class UNetTransform():
    def __init__(self, translate_range=0, rotate_range=0, shear_range=0, scale_range=0):
        self.translate_range = translate_range
        self.rotate_range = rotate_range
        self.shear_range = shear_range
        self.scale_range = scale_range

        self.transforms = {
                "train" : Compose([
                    ReadImage(), 
                    AffineTransform(self.translate_range, self.rotate_range, self.shear_range, self.scale_range),
                    GetArrayFromImage()
                    ]), 

                "val" : Compose([
                    ReadImage(), 
                    GetArrayFromImage()
                    ])
                }

    def __call__(self, phase, image, label):

        return self.transforms[phase](image, label)


