import numpy as np
import rasterio
import argparse
import os
from rasterio import Affine


def parse_args():
    parser = argparse.ArgumentParser(description='creates input for CNN pan-sharpening')
    parser.add_argument('--pan_scene', type=str, help='path to panchromatic scene to be merged')
    parser.add_argument('--ms_scene', type=str, help='path to multispectral scene to be merged')
    parser.add_argument('--out_folder', type=str, help='output folder to save processed images')
    return parser.parse_args()


def rescale(src_scene, scale, target_shape, output='out.tif'):
    """
    Rescales a scene to a given scale, writes it out to an output .tif file keeping all metadata
    except the new affine matrix scales and raster dimensions.
    :param src_scene: input scene to be rescaled, str
    :param scale: scaling factor -- new dimensions = scale * old dimensions, tuple
    :param target_shape: target shape for projected image, tuple
    :param output: output file name, str
    :return: None
    """
    # check if output file naming is valid
    assert output.split('.')[1] == 'tif', 'output file must be a .tif'
    assert len(target_shape) == 2, 'invalid dimensions for target shape'
    assert len(scale) == 2, 'invalid dimensions for warping scale'

    # create affine matrix for rescaled image
    affine = src_scene.transform
    rescaled_affine = Affine(affine.a / scale[0], affine.b, affine.c,
                             affine.d, affine.e / scale[1], affine.f)

    scene = src_scene.read()
    # create empty array for rescaled image
    rescaled = np.zeros(shape=(scene.shape[0],
                               target_shape[0],
                               target_shape[1]),
                        dtype=np.uint8)

    kwargs = src_scene.meta
    kwargs['transform'] = rescaled_affine
    kwargs['width'] = target_shape[1]
    kwargs['height'] = target_shape[0]

    # reproject by reading into array
    src_scene.read(out=rescaled)

    # write rescaled raster to file
    with rasterio.open(output, mode='w', **kwargs) as out:
        out.write(rescaled)


def generate_training_scene(pan_scene, ms_scene, out_folder):
    """
    Rescales a panchromatic and a multispectral scene and writes processed rasters to folder.

    :param pan_scene: path to panchromatic scene to be merged, str
    :param ms_scene: path to multispectral scene to be merged, str
    :param out
    :return: None
    """
    # create output folder
    os.makedirs(out_folder, exist_ok=True)
    # get target resolution from multispectral scene
    with rasterio.open(pan_scene) as src_pan:
        with rasterio.open(ms_scene) as src_ms:
            # target resolution to downscale panchromatic to multispectral resolution
            target_scale = [src_ms.shape[1] / src_pan.shape[1], src_ms.shape[0] / src_pan.shape[0]]
            # target shape = multispectral shape
            target_shape = np.array(src_ms.shape)

            # downsample panchromatic to multispectral resolution
            rescale(src_scene=src_pan, scale=target_scale, target_shape=target_shape,
                    output=f"{out_folder}/{pan_scene.split('_')[3].split('-')[0]}_pan.tif")

            # downsample then upsample multispectral
            rescale(src_scene=src_ms, scale=[0.5, 0.5], target_shape=target_shape // 2, output='temp.tif')
            with rasterio.open('temp.tif') as ms_down:
                rescale(src_scene=ms_down, scale=[2, 2], target_shape=target_shape,
                        output=f"{out_folder}/{ms_scene.split('_')[3].split('-')[0]}_ms.tif")

    # remove temporary file
    os.remove('temp.tif')


def main():
    args = parse_args()
    generate_training_scene(args.pan_scene, args.ms_scene, args.out_folder)


if __name__ == "__main__":
    main()
