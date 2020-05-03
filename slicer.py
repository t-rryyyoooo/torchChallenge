import sys
import SimpleITK as sitk
import numpy as np
from cut import searchBound, caluculateClipSize, caluculateArea, getCenterOfGravity
from functions import getImageWithMeta, resampleSize, write_file, createParentPath
from tqdm import tqdm
import re
from pathlib import Path

class slicer():
    def __init__(self, image, label, outputImageSize="256-256-3", widthSize=15, paddingSize=100, onlyCancer=False, noFlip=False):
        self.image = image
        self.label = label
        self.outputImageSize = outputImageSize
        self.widthSize = widthSize
        self.paddingSize = paddingSize
        self.onlyCancer = onlyCancer
        self.noFlip = noFlip

    def execute(self):
        self.meta = [{} for _ in range(2)]
        labelArray = sitk.GetArrayFromImage(self.label)
        imageArray = sitk.GetArrayFromImage(self.image)

        matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", self.outputImageSize)
        if matchobj is None:
            print("[ERROR] Invalid patch size : {}".format(self.outputImageSize))
            sys.exit()

        self.outputImageSize = [int(s) for s in matchobj.groups()]
        self.outputLabelSize = self.outputImageSize[:2] + [1]
        
        startIndex, endIndex = searchBound(labelArray, "sagittal")
        if len(startIndex) != 2:
            print("The patient has horse shoe kidney")
            sys.exit()

        largestKidneyLabelArray = []
        largestKidneyImageArray = []

        largestSlice = slice(endIndex[0], labelArray.shape[0])
        largestKidneyLabelArray.append(labelArray[largestSlice, ...])
        largestKidneyImageArray.append(imageArray[largestSlice, ...])
        self.meta[0]["largestSlice"] = largestSlice

        largestSlice = slice(0, startIndex[1])
        largestKidneyLabelArray.append(labelArray[largestSlice, ...])
        largestKidneyImageArray.append(imageArray[largestSlice, ...])
        self.meta[1]["largestSlice"] = largestSlice

        if not self.noFlip:
            largestKidneyLabelArray[1] = largestKidneyLabelArray[1][::-1, ...]
            largestKidneyImageArray[1] = largestKidneyImageArray[1][::-1, ...]
        else:
            print("The kidney doesn't flip.")


        self.cuttedLabelArrayList = [[] for _ in range(2)]
        self.cuttedStackedLabelArrayList = [[] for _ in range(2)]
        self.cuttedImageArrayList = [[] for _ in range(2)]

        for i in range(2):
            p = (self.paddingSize, self.paddingSize)
            largestKidneyLabelArray[i] = np.pad(largestKidneyLabelArray[i], [p, p, (0, 0)], "minimum")
            largestKidneyImageArray[i] = np.pad(largestKidneyImageArray[i], [p, p, (0, 0)], "minimum")

            startIndex, endIndex = caluculateClipSize(largestKidneyLabelArray[i], "axial")
            axialSlice = slice(startIndex, endIndex)
            
            largestKidneyLabelArray[i] = largestKidneyLabelArray[i][..., axialSlice]
            largestKidneyImageArray[i] = largestKidneyImageArray[i][..., axialSlice]
            self.meta[i]["axialSlice"] = axialSlice

            length = largestKidneyLabelArray[i].shape[2]
            area = []
            for x in range(length):
                area.append(caluculateArea(largestKidneyLabelArray[i][..., x]))
                maxArea = np.argmax(area)

            maxAreaLabelArray = largestKidneyLabelArray[i][..., maxArea]
            
            s, e = caluculateClipSize(maxAreaLabelArray[..., np.newaxis], "sagittal")
            width = e - s
            s, e = caluculateClipSize(maxAreaLabelArray[..., np.newaxis], "coronal")
            height = e - s
            wh = max(width, height)

            margin = self.outputImageSize[2] // 2
            top, bottom = caluculateClipSize(largestKidneyLabelArray[i], "axial", widthSize = 0)
            topSliceArray = largestKidneyLabelArray[i][..., top]
            bottomSliceArray = largestKidneyLabelArray[i][..., bottom - 1]

            check = False
            sagittalSlices = []
            coronalSlices = []
            size = []
            for x in tqdm(range(length), desc="Slicing images...", ncols = 60):
                a = caluculateArea(largestKidneyLabelArray[i][..., x])
                if a == 0:
                    if not check:
                        sliceLabelArray = topSliceArray
                    else:
                        sliceLabelArray = bottomSliceArray

                else:
                    sliceLabelArray = largestKidneyLabelArray[i][..., x]
                    check = True

                center = getCenterOfGravity(sliceLabelArray)
                x0 = center[0] - wh // 2
                x1 = center[0] + wh // 2
                y0 = center[1] - wh // 2
                y1 = center[1] + wh // 2

                minLabelArray = np.zeros((x1 - x0, y1 - y0)) + largestKidneyLabelArray[i].min()
                minImageArray = np.zeros((x1 - x0, y1 - y0)) + largestKidneyImageArray[i].min()

                sagittalSlice = slice(x0, x1)
                coronalSlice = slice(y0, y1)
                sagittalSlices.append(sagittalSlice)
                coronalSlices.append(coronalSlice)

                cuttedImageArray = []
                cuttedStackedLabelArray = []
                for y in range(-margin, margin + 1):
                    if 0 <= x + y < length:
                        cuttedStackedLabelArray.append(largestKidneyLabelArray[i][sagittalSlice, coronalSlice, x + y])
                        cuttedImageArray.append(largestKidneyImageArray[i][sagittalSlice, coronalSlice, x + y])

                    else:
                        cuttedStackedLabelArray.append(minLabelArray)
                        cuttedImageArray.append(minImageArray)

                cuttedLabelArray = largestKidneyLabelArray[i][sagittalSlice, coronalSlice, x]
                cuttedStackedLabelArray = np.dstack(cuttedStackedLabelArray)
                cuttedImageArray = np.dstack(cuttedImageArray)

                self.cuttedLabelArrayList[i].append(cuttedLabelArray)
                self.cuttedStackedLabelArrayList[i].append(cuttedStackedLabelArray)
                self.cuttedImageArrayList[i].append(cuttedImageArray)

                size.append(cuttedLabelArray.shape)

            self.meta[i]["sagittalSlice"] = sagittalSlices
            self.meta[i]["coronalSlice"] = coronalSlices
            self.meta[i]["size"] = size

        """ For resampling, get direction, spacing, origin and minimun value in image and label in 2D. """
        extractSliceFilter = sitk.ExtractImageFilter()
        size = list(self.image.GetSize())
        size[0] = 0
        index = (0, 0, 0)
        extractSliceFilter.SetSize(size)
        extractSliceFilter.SetIndex(index)
        self.sliceImage = extractSliceFilter.Execute(self.image)

        self.cuttedLabelList = [[] for _ in range(2)]
        self.cuttedImageList = [[] for _ in range(2)]
        for i in range(2):
            length = len(self.cuttedLabelArrayList[i])
            for x in tqdm(range(length), desc="Transforming images...", ncols=60):
                cuttedLabel = getImageWithMeta(self.cuttedLabelArrayList[i][x][..., np.newaxis], self.image)
                cuttedImage = getImageWithMeta(self.cuttedImageArrayList[i][x], self.image)
                cuttedLabel = resampleSize(cuttedLabel, self.outputLabelSize[::-1], is_label = True)
                cuttedImage = resampleSize(cuttedImage, self.outputImageSize[::-1])

                cuttedLabelArray = sitk.GetArrayFromImage(cuttedLabel)
                cuttedImageArray = sitk.GetArrayFromImage(cuttedImage)

                self.cuttedLabelArrayList[i][x] = cuttedLabelArray
                self.cuttedImageArrayList[i][x] = cuttedImageArray

                self.cuttedLabelList[i].append(cuttedLabel)
                self.cuttedImageList[i].append(cuttedImage)

    def output(self, kind = "Array"):
        if kind == "Array":
            return self.cuttedLabelArrayList, self.cuttedImageArrayList

        elif kind == "Image":
            return self.cuttedLabelList, self.cuttedImageList

        else:
            print("Argument error kind = [Array / Image]")
            sys.exit()


    def save(self, savePath, patientID):
        if self.onlyCancer:
            print("Saving only images with cancer.")


        savePath = Path(savePath)
        saveImagePath = savePath / "image" / patientID / "dummy.mha"
        saveTextPath = savePath / "path" / (patientID + ".txt")

        if not saveImagePath.parent.exists():
            createParentPath(str(saveImagePath))

        if not saveTextPath.parent.exists():
            createParentPath(str(saveTextPath))

        for i in range(2):
            length = len(self.cuttedLabelList[i])
            for x in tqdm(range(length), desc="Saving images...", ncols=60):
                if self.onlyCancer and not (self.cuttedStackedLabelArrayList[i][x] == 2).any():
                    continue

                saveImagePath = savePath / "image" / patientID / "image_{}_{:02d}.mha".format(i, x)
                saveLabelPath = savePath / "image" / patientID / "label_{}_{:02d}.mha".format(i, x)

                sitk.WriteImage(self.cuttedLabelList[i][x], str(saveLabelPath), True)
                sitk.WriteImage(self.cuttedImageList[i][x], str(saveImagePath), True)

                write_file(str(saveTextPath) ,str(saveImagePath) + "\t" + str(saveLabelPath))


    def restore(self, predictArrayList):
        labelArray = sitk.GetArrayFromImage(self.label)
        outputArray = np.zeros_like(labelArray)
        for i in range(2):
            length = len(predictArrayList[i])
            largestSlice = self.meta[i]["largestSlice"]
            axialSlice = self.meta[i]["axialSlice"]
            paddingSize = self.paddingSize
            largestArray = outputArray[largestSlice, ...]
            p = (paddingSize, paddingSize)
            largestArray = np.pad(largestArray, [p, p, (0, 0)], "minimum")
            largestArray = largestArray[..., axialSlice]
            for x in tqdm(range(length), desc="Restoring images...", ncols=60):
                pre = getImageWithMeta(predictArrayList[i][x], self.sliceImage)
                sagittalSlice = self.meta[i]["sagittalSlice"][x]
                coronalSlice = self.meta[i]["coronalSlice"][x]
                size = self.meta[i]["size"][x]

                pre = resampleSize(pre, size[::-1], is_label = True)
                preArray = sitk.GetArrayFromImage(pre)
                largestArray[sagittalSlice, coronalSlice, x] = preArray

            largestArray = largestArray[paddingSize : -paddingSize, paddingSize : -paddingSize, :]

            if not self.noFlip:
                if i == 1:
                    largestArray = largestArray[::-1, ...]

            outputArray[largestSlice, :, axialSlice] += largestArray

        return outputArray














