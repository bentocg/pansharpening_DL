import argparse
import os
from itertools import product

import cv2
import numpy as np
import rasterio
from rasterio import windows


def parse_args():
    parser = argparse.ArgumentParser(description='tiles a pair of processed scenes into pairs of training patches')
    parser.add_argument('--file_pan', type=str, help="processed panchromatic scene file name")
    parser.add_argument('--patch_size', type=int, default=450, help="training image dimensions")
    parser.add_argument('--stride', type=float, default=1, help="stride for sliding window, ")
    parser.add_argument('--thresholds', type=str, default='5_250', help="threshold to throw away "
                                                                        "images without interesting "
                                                                        "content ")
    parser.add_argument('--dataset', type=str, default="training", help="dataset name where tiles"
                                                                        "will be saved to")
    parser.add_argument('--RGB', type=str, default=1, help="whether training tiles will be RGB"
                                                           "or multispectral")
    return parser.parse_args()


def tile_training_scene(file_pan: str, patch_size: int = 450, stride: float = 1, thresholds: tuple = (5, 255),
                        dataset: str = 'training', RGB=1):
    """
    Creates training patches from processed scenes. Patches are grouped in pairs of panchromatic and
    multispectral images.

    :param file_pan: file name for panchromatic image.
    :param patch_size: dimensions for training images.
    :param stride: sliding window stride parametrized as a multiple of patch size.
    :param thresholds: thresholds to throw away patches with no content.
    :param dataset: whether patches will be saved on training, validation or test sets
    :param RGB: whether patches will be RGB or 8 band multispectral
    :return:
    """
    # check if a ground-truth exists
    scn_gt = [scn for scn in [ele for ele in os.listdir("raw_scenes") if 'M1BS' in ele] if
              file_pan.split('_')[0] in scn]
    assert len(scn_gt) > 0, "no groundtruth scene found for input raster"

    # read groundtruth, panchromatic and multispectral versions of the scene
    scn_pan = rasterio.open(f"processed_scenes/{file_pan}")
    scn_ms = rasterio.open(f"processed_scenes/{file_pan.split('_')[0]}_ms.tif")
    scn_gt = rasterio.open(f"raw_scenes/{scn_gt[0]}")

    # check input
    assert scn_ms.shape == scn_pan.shape, "panchromatic and multispectral input dimensions do not match"

    # create output folder
    output_path = f"training_set/{dataset}/"
    for ele in ['x', 'y']:
        os.makedirs(f"{output_path}/{ele}", exist_ok=True)

    # template for output filename
    output_filename = output_path + '/{}/{}_{}_{}.tif'

    # get width and height and prepare iterator for sliding window
    nrows, ncols = scn_pan.shape
    offsets = product(range(0, nrows, int(patch_size * stride)), range(0, ncols, int(patch_size * stride)))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)

    # extract patches on sliding window
    for row, col in offsets:
        window = windows.Window(col_off=col, row_off=row, width=patch_size, height=patch_size).intersection(big_window)
        patch_pan = scn_pan.read(window=window)

        # check content with thresholds
        if np.max(patch_pan) < thresholds[0] or np.min(patch_pan) > thresholds[1]:
            continue

        if RGB:
            # stack panchromatic (1 channel) and multispectral -- need to extract RGB bands from multispectral
            patch_ms = scn_ms.read(window=window)[[2, 3, 5], :, :]
            patch_merged = np.vstack([patch_ms, patch_pan]).transpose([1, 2, 0])
            # opencv saves images with a BGR encoding, panchromatic is added as an alpha channel
            cv2.imwrite(output_filename.format('x', file_pan.split('_')[0], row, col), patch_merged)

            # save ground-truth correspondent
            patch_gt = scn_gt.read(window=window)[[2, 3, 5], :, :].transpose([1, 2, 0])
            cv2.imwrite(output_filename.format('y', file_pan.split('_')[0], row, col), patch_gt)

        else:
            # TODO add support for multispectral patches
            continue


def main():
    args = parse_args()
    thresholds = tuple(([int(ele) for ele in args.thresholds.split('_')]))
    tile_training_scene(args.file_pan, args.patch_size, args.stride, thresholds, args.dataset)


if __name__ == "__main__":
    main()
