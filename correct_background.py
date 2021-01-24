#!/usr/bin/env python

import numpy as np
import skimage.io as io
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.util as util
import argparse
import os
import errno
import glob
import re
import parse_indexes as pidx

#Define the functions to use to convert different formats
FORMAT_MAP = {
    np.dtype('float64'): util.img_as_float,
    np.dtype('uint16'): util.img_as_uint,
    np.dtype('uint8'): util.img_as_ubyte
}


def parse_args():
    parser = argparse.ArgumentParser(description='correct uneven background in\
                                     TEM images')
    parser.add_argument('bkgd_image', 
                        help='path to the background image')
    parser.add_argument('target_folder', 
                        help='path to a folder of images to \
                        correct with the background image')
    parser.add_argument('-m', '--multiplier', type=float,
                        help='multiply background by this \
                        number prior to subtraction. If the -o flag is set \
                        then this value is used as the initial guess for \
                        the optimization routine.')
    parser.add_argument('-o', '--optimize-multiplier', action='store_true', 
                        help='Optimize the multiplier using gradient descent,\
                        seeking to minimize the sum of the squares of\
                        background pixel intensities after subtraction.')
    parser.add_argument('-s', '--output-suffix', default='_corrected',
                        help='suffix added to the name of each output image\
                        (default is "%(default)s").')
    parser.add_argument('--target-substring',metavar='SUBSTRING',
                        help='a substring contained within each file in \
                        the target_folder that idenfies the subset of files \
                        within that folder that should be processed. Files \
                        in target_folder that do NOT contain this string \
                        will NOT be processed. If this is not specified, all\
                        .tif files in the folder will be processed.')
    parser.add_argument('-i', '--indexes', type=pidx.parseNumList,
                        action=pidx.concat, nargs='+',
                        help='EM images are numbered at the end of the \
                        filename. Specify a range of numbers to get a list \
                        of numbers that will match a subset of items in the \
                        target_folder. This is in addition to any matching \
                        based on a target_substring. -i 0 2 5-7 will return \
                        a list [0 2 5 6 7]')

    args = parser.parse_args()
    #If multiplier is not supplied, then assume it must be found with optimizer
    if args.multiplier is None:
        print('no multiplier specified, will run optimizer to find one...')
        args.optimize_multiplier = True
    #Check that both the background file and target folder exist
    if not os.path.isfile(args.bkgd_image):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                args.bkgd_image)
    if not os.path.isdir(args.target_folder):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                args.target_folder)

    return args


def get_background_in_target(target):
    '''Find the background region of the given image (i.e. areas of slow
    signal variation.
    Input: 
        - target: target image, containing foreground and background
    Output:
        - mask: a boolean image same size as target where 1 is the background
        and 0 is the foreground
    '''
    #Get the appropriate datatype
    target_float = util.img_as_float(target)
    #blur the image
    target_blur = filters.gaussian(target_float, sigma=2)
    #Threshold based on edge detection
    #(ansatz: in EM images foreground is highly textured)
    target_edge = filters.sobel(target_blur)
    thresh = filters.threshold_otsu(target_edge)
    target_binary = target_edge >= thresh
    #Clean up the binary image (doesn't need to be perfect)
    disk = morphology.disk(10)
    target_close = morphology.closing(target_binary, disk)
    target_fill = morphology.remove_small_holes(target_close,
                                                area_threshold=10000)
    target_fill = morphology.remove_small_objects(target_fill, min_size=5000)
    mask = np.logical_not(target_fill)
    return mask


def df(target, bkgd, x):
    '''Gradient for the optimization function (target - x * bkgd)^2/N, where
    N is the length of the target (or bkgd) vector.
    Input:
        - target, bkgd: vectors of target and bkgd
        - x: scalar to be optimized
    '''
    N = len(target)
    return -2 * np.sum(bkgd * (target - x * bkgd)) / N


def find_multiplier_gradient_descent(target, bkgd, guess):
    '''Apply gradient descent to find the optimum multiplier for background 
    subtraction
    Input:
        - target, bkgd: the target image and background image (or vector)
        - guess: scalar, initial guess for the algorithm
    Output: 
        - multiplier: scalar, final quantity to be multiplied
    Code taken from https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1
    '''
    cur_multiplier = guess
    rate = 0.5
    precision = 1e-5
    previous_step_size = 1
    max_iters = 1000
    iters = 0

    while previous_step_size > precision and iters < max_iters:
        prev_multiplier = cur_multiplier
        cur_multiplier = cur_multiplier - rate * df(target, bkgd,
                                                    prev_multiplier)
        previous_step_size = abs(cur_multiplier - prev_multiplier)
        iters = iters + 1

    return cur_multiplier


def find_optimal_multiplier(target, bkgd, initial_guess):
    '''Process the target image to find the multiplier that best subtracts
    the background of the target image.
    '''
    mask = get_background_in_target(target)
    #Ensure that the images are floating-point
    target_float = util.img_as_float(target)
    bkgd_float = util.img_as_float(bkgd)
    target_masked = target_float[mask]
    bkgd_masked = bkgd_float[mask]
    #Downsample to save computing time
    target_masked = target_masked[::2]
    bkgd_masked = bkgd_masked[::2]
    if initial_guess is None:
        initial_guess = round(np.mean(target_masked) / np.mean(bkgd_masked), 1)
    multiplier = find_multiplier_gradient_descent(target_masked,
                                                  bkgd_masked,
                                                  initial_guess)
    print('Multiplier found by optimization: {}'.format(multiplier))
    return multiplier


def subtract_background(target, bkgd, x):
    '''Subtract the background from target image using multiplier x, and
    add back a constant value. Preserve datatype of target image after
    subtraction
    Input:
        - target, bkgd: Target and background images
        - x: scalar, multiplier to multiply background by before subtraction
    '''
    #Check datatype of target
    target_dtype = target.dtype
    target = util.img_as_float(target)
    bkgd = util.img_as_float(bkgd)
    
    #Subtract background
    new = target - x * bkgd + 0.5 #0.5 is a fill value to allow re-conversion
    #Convert back to the original datatype of target
    target_subtracted = FORMAT_MAP[target_dtype](new)
    
    return target_subtracted


def save_file_with_suffix(im_to_save, old_name, save_suffix):
    '''Save the indicated array, using a suffix appended before the
    extension
    '''
    (dirname, basename) = os.path.split(old_name)
    (base, ext) = os.path.splitext(basename)
    new_basename = base + save_suffix + ext
    new_name = os.path.join(dirname, new_basename)
    io.imsave(new_name, im_to_save, check_contrast=False)


def specify_file_subset_in_folder(folder, substring, indexes):
    '''For a folder with a lot of files, specify a subset with a matching
    substring and matching given indexes. If both are None, just return
    the whole folder
    Input:
        - folder: path to a folder
        - substring: a substring that should be contained in any file from
        folder that will be processed
        - indexes: a list of integers that can further refine the list of 
        files containing the substring
    Output:
        - file_list: a list of paths to files to process
    '''
    if substring is None:
        file_specifier = os.path.join(folder,'*.tif')
    else:
        file_specifier = os.path.join(folder,'*{}*.tif'.format(substring))
    candidate_file_list = sorted(glob.glob(file_specifier))
    if len(candidate_file_list) == 0:
        raise Exception(("no files in {} specified by the given "
                         "substring '{}'").format(folder, substring))
    if isinstance(indexes, list):
        file_list = []
        for f in candidate_file_list:
            m = re.match(r'.*(\d{2})\.tif$', f)
            if m:
                candidate_index = int(m.group(1))
                if int(m.group(1)) in indexes:
                    file_list.append(f)
    else:
        file_list = candidate_file_list
    return file_list


def main():
    args = parse_args()

    bkgd = io.imread(args.bkgd_image)
    #Get the list of files to process
    target_file_list = specify_file_subset_in_folder(args.target_folder,
                                                     args.target_substring,
                                                     args.indexes)
    #Find optimum multiplier using the first target image as an example
    if args.optimize_multiplier:
        guess = args.multiplier
        target = io.imread(target_file_list[0])
        multiplier = find_optimal_multiplier(target, bkgd, guess)
    else:
        multiplier = args.multiplier

    print('processing target files...')
    for target_file in target_file_list:
        print('processing file {}...'.format(os.path.basename(target_file)))
        target = io.imread(target_file)
        target_subtracted = subtract_background(target, bkgd, multiplier)
        save_file_with_suffix(target_subtracted, target_file, 
                              args.output_suffix)


if __name__ == '__main__':
    main()
