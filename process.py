
import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import UniqueImagesValidator

# added imports
#import tensorflow as tf
from typing import Tuple, List
from pathlib import Path
import re

import subprocess

from evalutils.io import (ImageLoader, SimpleITKLoader)


class MixLacune(SegmentationAlgorithm):
    def __init__(self):
        self.input_modalities = ['T1', 'T2', 'FLAIR']
        self.first_modality = self.input_modalities[0]

        self.flag_save_uncertainty = True

        super().__init__(
            # (Skip UniquePathIndicesValidator, because this will error when there are multiple images
            # for the same subject)
            validators=dict(input_image=(UniqueImagesValidator(),)),
            # Indicate with regex which image to load as input, e.g. T1 scan
            file_filters={'input_image':
                          re.compile("/input/sub-.*_space-.*_desc-masked_%s.nii.gz" % self.first_modality)}
        )

        print("==> Initializing model")

        # --> Load model
        # TODO add code to load model

        print("==> Model loaded")

    def _load_input_image(self, *, case) -> Tuple[List[SimpleITK.Image], List[Path]]:
        input_image_file_path = case["path"]

        input_image_file_loader = self._file_loaders["input_image"]
        if not isinstance(input_image_file_loader, ImageLoader):
            raise RuntimeError(
                "The used FileLoader was not of subclass ImageLoader"
            )
        input_images = []
        input_path_list = []

        # Load the image(s) for this case
        for modality in self.input_modalities:
            # Load all input images, e.g. T1, T2 and FLAIR
            scan_name = Path(input_image_file_path.name.replace('%s.nii.gz' % self.first_modality,
                                                                '%s.nii.gz' % modality))
            modality_path = input_image_file_path.parent / scan_name
            input_images.append(input_image_file_loader.load_image(modality_path))
            input_path_list.append(modality_path)

        # Check that it is the expected image
        if input_image_file_loader.hash_image(input_images[0]) != case["hash"]:
            raise RuntimeError("Image hashes do not match")

        return input_images, input_path_list

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_images, input_path_list = self._load_input_image(case=case)

        # Segment case
        list_results = self.predict(input_images=input_images)

        if self.flag_save_uncertainty:
            assert len(list_results) == 2, "Error, predict function should return a list containing 2 images, " \
                                           "the predicted segmentation and the predicted uncertainty map. " \
                                           "Or change flag_save_uncertainty to False"
        else:
            assert len(list_results) == 1, "Error, predict function should return a list containing 1 image, " \
                                           "only the predicted segmentation. " \
                                           "Or change flag_save_uncertainty to True"

        # Write resulting segmentation to output location
        if not self._output_path.exists():
            self._output_path.mkdir()

        save_description = ['prediction', 'uncertaintymap']
        output_path_list = []

        for i, outimg in enumerate(list_results):
            output_name = Path(input_path_list[0].name.split("desc-masked_")[0] + "%s.nii.gz" % save_description[i])
            segmentation_path = self._output_path / output_name
            print(segmentation_path)
            output_path_list.append(segmentation_path)
            SimpleITK.WriteImage(outimg, str(segmentation_path), True)

        input_name_list = [p.name for p in input_path_list]
        output_name_list = [p.name for p in output_path_list]

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [
                dict(type="metaio_image", filename=output_name_list)
            ],
            "inputs": [
                dict(type="metaio_image", filename=input_name_list)
            ],
            "error_messages": [],
        }

    def predict(self, *, input_images: List[SimpleITK.Image]) -> List[SimpleITK.Image]:
        print("==> Running prediction")
        
        # Process-lacunes.py assumes images are in a sub-folder and the first seven characters are the subject ID
        for i in range(len(self.input_modalities)):
            SimpleITK.WriteImage(input_images[i], '/home/input_data/lacunes/lacunes_'+self.input_modalities[i]+'.nii.gz')
        
        subprocess.run(["sh", "/home/run.sh"])
        
        prediction_image = SimpleITK.ReadImage('/home/output_data/lacunes_space-T1_binary_prediction.nii.gz')
        
        # Compute a (fake) uncertainty image. Uncertainty is usually at the lesion boundaries. By using dilation,
        # a 1-pixel boundary at the border of every segmentation (donut-shape) is created. This is done in 
        # 2D (3x3x0 kernel), because most uncertainty between raters is within-slice and not through-slice.
        # Only dilation, because our method tends to under-segment and thus errors are on the outside.
        #
        # I feel dirty doing this, I'm sorry.     
        dilated_prediction_image = SimpleITK.BinaryDilate(prediction_image, [1, 1, 0])
        uncertainty_image = SimpleITK.Subtract(dilated_prediction_image, prediction_image)
                
        print("==> Prediction done")
        
        return [prediction_image, uncertainty_image]

    
if __name__ == "__main__":
    MixLacune().process()
