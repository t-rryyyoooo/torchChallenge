import SimpleITK as sitk
import numpy as np
import argparse
from functions import createParentPath, getImageWithMeta
from pathlib import Path
from slicer import slicer as sler
from tqdm import tqdm
import torch
import cloudpickle
from UNet.model import UNetModel


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("imageDirectory", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("savePath", help="Segmented label file.(.mha)")
    parser.add_argument("-g", "--gpuid", nargs="+", help="ID of GPU to be used for segmentation.", type=int)

    parser.add_argument("--widthSize", default=15, type=int)
    parser.add_argument("--paddingSize", default=100, type=int)
    parser.add_argument("--patchSize", help="256-256-3", default="256-256-3")
    parser.add_argument("--noFlip", action="store_true")


    
    args = parser.parse_args()
    return args

def main(args):
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    """ Slice module. """
    labelFile = Path(args.imageDirectory) / "segmentation.nii.gz"
    imageFile = Path(args.imageDirectory) / "imaging.nii.gz"

    label = sitk.ReadImage(str(labelFile))
    image = sitk.ReadImage(str(imageFile))

    slicer = sler(
            image = image, 
            label = label, 
            outputImageSize = args.patchSize, 
            widthSize = args.widthSize, 
            paddingSize = args.paddingSize, 
            noFlip = args.noFlip
            )

    slicer.execute()
    _, cuttedImageArrayList = slicer.output("Array")

    """ Load model. """

    with open(args.modelweightfile, 'rb') as f:
        model = cloudpickle.load(f)
        model = torch.nn.DataParallel(model, device_ids=args.gpuid)

    model.eval()

    """ Segmentation module. """

    segmentedArrayList = [[] for _ in range(2)]
    for i in range(2):
        length = len(cuttedImageArrayList[i])
        for x in tqdm(range(length), desc="Segmenting images...", ncols=60):
            imageArray = cuttedImageArrayList[i][x]
            imageArray = imageArray.transpose((2, 0, 1))
            imageArray = torch.from_numpy(imageArray[np.newaxis, ...]).to(device, dtype=torch.float)
            
            segmentedArray = model(imageArray)
            segmentedArray = segmentedArray.to("cpu").detach().numpy().astype(np.float)
            segmentedArray = np.squeeze(segmentedArray)
            segmentedArray = np.argmax(segmentedArray, axis=-1).astype(np.uint8)
            segmentedArrayList[i].append(segmentedArray)

    """ Restore module. """
    segmentedArray = slicer.restore(segmentedArrayList)

    segmented = getImageWithMeta(segmentedArray, label)
    createParentPath(args.savePath)
    print("Saving image to {}".format(args.savePath))
    sitk.WriteImage(segmented, args.savePath, True)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
