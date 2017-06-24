#!/usr/bin/env python

import os
import sys
import glob
import re
import argparse

import nibabel as nib
import numpy as np


def fparse(fname):
    """
    Filename parser for NIFTI and AFNI files.

    Returns prefix from input datafiles (e.g., sub_001) and filetype (e.g.,
    .nii). If AFNI format is supplied, extracts space (+tlrc, +orig, or +mni)
    as filetype.

    Parameters
    ----------
    fname : str

    Returns
    -------
    str, str: prefix of filename, type of filename
    """

    if '.' in fname:
        if '+' in fname:  # i.e., if AFNI format
            prefix = fname.split('+')[0]
            suffix = ''.join(fname.split('+')[-1:])
            ftype  = '+' + suffix.split('.')[0]
        else:
            if fname.endswith('.gz'):  # if gzipped, two matches for ftype
                prefix = '.'.join(fname.split('.')[:-2])
                ftype  = '.' + '.'.join(fname.split('.')[-2:])
            else:
                prefix = '.'.join(fname.split('.')[:-1])
                ftype  = '.' + ''.join(fname.split('.')[-1:])
    else:
        prefix = fname
        ftype  = ''

    return prefix, ftype


def format_inset(inset, tes=None, e=0):
    """
    Parse input file specification as short- or longhand.

    Given a dataset specification (usually from options.input_ds),
    parse and create a setname for all output files. Also optionally takes
    a label to append to all created files.

    Parameters
    ----------
    inset : str
    tes   : str
    e     : echo to process, default 1

    Returns
    -------
    str: dataset name
    str: name to be "set" for any output files
    """
    import re

    try:
        # Parse shorthand input file specification
        if '[' in inset:
            fname, ftype = fparse(inset)
            if '+' in ftype:  # if AFNI format, call .HEAD file
                ftype = ftype + '.HEAD'
            prefix       = re.split(r'[\[\],]',fname)[0]
            echo_nums    = re.split(r'[\[\],]',fname)[1:-1]
            trailing     = re.split(r'[\]+]',fname)[-1]
            dsname       = prefix + echo_nums[e] + trailing + ftype
            setname      = prefix + ''.join(echo_nums) + trailing

        # Parse longhand input file specificiation
        else:
            datasets_in   = inset.split(',')
            prefix, ftype = fparse(datasets_in[0])
            echo_nums     = [str(vv+1) for vv in range(len(tes))]
            trailing      = ''
            dsname        = datasets_in[e].strip()
            setname       = prefix + ''.join(echo_nums[1:]) + trailing
            assert len(echo_nums) == len(datasets_in)

    except AssertionError as err:
            print("*+ Can't understand dataset specification. "            +
                  "Number of TEs and input datasets must be equal and "    +
                  "matched in order. Try double quotes around -d argument.")
            raise err

    return dsname, setname


def check_obliquity(fname):
    """
    Generates obliquity (in degrees) of `fname`

    Parameters
    ----------
    fname : str
        path to neuroimage

    Returns
    -------
    float : angle from plumb (in degrees)
    """
    import nibabel as nib
    import numpy as np

    # get abbreviated affine matrix (3x3)
    aff = nib.load(fname).affine[:3,:-1]

    # generate offset (rads) and convert to degrees
    fig_merit = np.min(np.sqrt((aff**2).sum(axis=0)) / np.abs(aff).max(axis=0))
    ang_merit = (np.arccos(fig_merit) * 180) / np.pi

    return ang_merit


def find_CM(fname):
    """
    Generates center of mass for `fname`

    Will only use the first volume if a 4D image is supplied

    Parameters
    ----------
    fname : str
        path to neuroimage

    Returns
    -------
    float, float, float : x, y, z coordinates of center of mass
    """
    import nibabel as nib
    import numpy as np

    im = nib.load(fname)
    data = im.get_data()
    if data.ndim > 3: data = data[:,:,:,0]  # use first volume in 4d series
    data_sum = data.sum()  # to ensure the dataset is non-zero

    # if dataset is not simply zero, then calculate CM in i, j, k values
    # otherwise, just pick the dead center of the array
    if data_sum > 0:
        cm = []  # to hold CMs
        for n, dim in enumerate(im.shape[:3]):
            res = np.ones(3,dtype='int')  # create matrix for reshaping
            res[n] = dim                  # set appropriate dimension to dim
            cm.append((np.arange(dim).reshape(res) * data).sum()/data_sum)
    else:
        cm = 0.5*(np.array(im.shape[:3])-1)

    # for some reason, the affine as read by nibabel is different than AFNI;
    # specifically, the first two rows are inverted, so we invert them back
    # (since we're trying to do this as AFNI would)
    # then, we calculate the orientation, reindex the affine matrix, and
    # generate the centers of mass from there based on the above calculations
    affine = im.affine * [[-1],[-1],[1],[1]]             # fix affine
    orient = np.abs(affine).argsort(axis=0)[-1,:3]       # get orientation
    affine = affine[orient]                              # reindex affine
    cx, cy, cz = affine[:,-1] + cm * np.diag(affine)     # calculate centers
    cx, cy, cz = np.array([cx,cy,cz])[orient.argsort()]  # reorient centers

    return cx, cy, cz


def run(options):

    get_cm = pe.Node(util.Function(input_names=['fname'],
                                   output_names=['x', 'y', 'z'],
                                   function=find_CM),
                     name='get_cm')
    meica_wf.connect(subj_iterable, 'subject_id', get_cm, 'fname')


def get_options(_debug=None):
    """
    Parses command line inputs

    Returns
    -------
    argparse dic
    """

    # Configure options and help dialog
    parser = argparse.ArgumentParser()

    # Base processing options
    parser.add_argument('-e',
                        dest='tes',
                        help="Echo times in ms. ex: -e 14.5,38.5,62.5",
                        default="")
    parser.add_argument('-d',
                        dest='input_ds',
                        help="Input datasets. ex: -d RESTe[123].nii.gz",
                        default='')
    parser.add_argument('-a',
                        dest='anat',
                        help="(Optional) anatomical dataset. " +
                             "ex: -a mprage.nii.gz",
                        default='')
    parser.add_argument('-b',
                        dest='basetime',
                        help="Time to steady-state equilibration in " +
                             "seconds(s) or volumes(v). Default 0. ex: -b 4v",
                        default='0')
    parser.add_argument('--MNI',
                        dest='mni',
                        action='store_true',
                        help="Warp to MNI standard space.",
                        default=False)
    parser.add_argument('--strict',
                        dest='strict',
                        action='store_true',
                        help="Use strict component selection, suitable with" +
                             " large-voxel, high-SNR data",
                        default=False)

    # Extended options for processing
    extopts = parser.add_argument_group("Additional processing options")
    extopts.add_argument("--qwarp",
                         dest='qwarp',
                         action='store_true',
                         help="Nonlinear warp to standard space using QWarp," +
                              " requires --MNI or --space).",
                         default=False)
    extopts.add_argument("--native",
                         dest='native',
                         action='store_true',
                         help="Output native space results in addition to " +
                              "standard space results.",
                         default=False)
    extopts.add_argument("--space",
                         dest='space',
                         help="Path to specific standard space template for " +
                              "affine anatomical normalization.",
                         default=False)
    extopts.add_argument("--fres",
                         dest='fres',
                         help="Specify functional voxel dimensions in mm " +
                              "(iso.) for resampling during preprocessing." +
                              "ex: --fres=2.5",
                         default=False)
    extopts.add_argument("--no_skullstrip",
                         action="store_true",
                         dest='no_skullstrip',
                         help="Anat is intensity-normalized and " +
                              "skull-stripped (for use with '-a' flag).",
                         default=False)
    extopts.add_argument("--no_despike",
                         action="store_true",
                         dest='no_despike',
                         help="Do not de-spike functional data. " +
                              "Default is to de-spike, recommended.",
                         default=False)
    extopts.add_argument("--no_axialize",
                         action="store_true",
                         dest='no_axialize',
                         help="Do not re-write dataset in axial-first order." +
                              " Default is to axialize, recommended.",
                         default=False)
    extopts.add_argument("--mask_mode",
                         dest='mask_mode',
                         help="Mask functional with help from anatomical or" +
                              " standard space images." +
                              " Options: 'anat' or 'template'.",
                         default='func')
    extopts.add_argument("--coreg_mode",
                         dest='coreg_mode',
                         help="Coregistration with Local Pearson and T2* weights "+
                              "(default), or use align_epi_anat.py (edge method)."+
                              "Options: 'lp-t2s' or 'aea'",
                         default='lp-t2s')
    extopts.add_argument("--smooth",
                         dest='FWHM',
                         help="FWHM smoothing with 3dBlurInMask. Default none. " +
                              "ex: --smooth 3mm ",
                         default='0mm')
    extopts.add_argument("--align_base",
                         dest='align_base',
                         help="Explicitly specify base dataset for volume " +
                              "registration",
                         default='')
    extopts.add_argument("--TR",
                         dest='TR',
                         help="TR. Read by default from dataset header",
                         default='')
    extopts.add_argument("--tpattern",
                         dest='tpattern',
                         help="Slice timing (i.e. alt+z, see 3dTshift -help)." +
                              " Default from header.",
                         default='')
    extopts.add_argument("--align_args",
                         dest='align_args',
                         help="Additional arguments to anatomical-functional" +
                              " co-registration routine",
                         default='')
    extopts.add_argument("--ted_args",
                         dest='ted_args',
                         help="Additional arguments to " +
                              "TE-dependence analysis",
                         default='')

    # Additional, extended preprocessing options
    #  no help provided, caveat emptor
    extopts.add_argument("--select_only",
                         dest='select_only',
                         action='store_true',
                         help=argparse.SUPPRESS,
                         default=False)
    extopts.add_argument("--tedica_only",
                         dest='tedica_only',
                         action='store_true',
                         help=argparse.SUPPRESS,
                         default=False)
    extopts.add_argument("--export_only",
                         dest='export_only',
                         action='store_true',
                         help=argparse.SUPPRESS,
                         default=False)
    extopts.add_argument("--daw",
                         dest='daw',
                         help=argparse.SUPPRESS,
                         default='10')
    extopts.add_argument("--tlrc",
                         dest='space',
                         help=argparse.SUPPRESS,
                         default=False)  # For backwards compat.
    extopts.add_argument("--highpass",
                         dest='highpass',
                         help=argparse.SUPPRESS,
                         default=0.0)
    extopts.add_argument("--detrend",
                         dest='detrend',
                         help=argparse.SUPPRESS,
                         default=0.)
    extopts.add_argument("--initcost",
                         dest='initcost',
                         help=argparse.SUPPRESS,
                         default='tanh')
    extopts.add_argument("--finalcost",
                         dest='finalcost',
                         help=argparse.SUPPRESS,
                         default='tanh')
    extopts.add_argument("--sourceTEs",
                         dest='sourceTEs',
                         help=argparse.SUPPRESS,
                         default='-1')
    parser.add_argument_group(extopts)

    # Extended options for running
    runopts = parser.add_argument_group("Run options")
    runopts.add_argument("--prefix",
                         dest='prefix',
                         help="Prefix for final ME-ICA output datasets.",
                         default='')
    runopts.add_argument("--cpus",
                         dest='cpus',
                         help="Maximum number of CPUs (OpenMP threads) to use. " +
                         "Default 2.",
                         default='2')
    runopts.add_argument("--label",
                         dest='label',
                         help="Label to tag ME-ICA analysis folder.",
                         default='')
    runopts.add_argument("--test_proc",
                         action="store_true",
                         dest='test_proc',
                         help="Align, preprocess 1 dataset then exit.",
                         default=False)
    runopts.add_argument("--pp_only",
                         action="store_true",
                         dest='preproc_only',
                         help="Preprocess only, then exit.",
                         default=False)
    runopts.add_argument("--keep_int",
                         action="store_true",
                         dest='keep_int',
                         help="Keep preprocessing intermediates. " +
                              "Default delete.",
                              default=False)
    runopts.add_argument("--RESUME",
                         dest='resume',
                         action='store_true',
                         help="Attempt to resume from normalization step " +
                              "onwards, overwriting existing")
    runopts.add_argument("--OVERWRITE",
                         dest='overwrite',
                         action="store_true",
                         help="Overwrite existing meica directory.",
                         default=False)
    parser.add_argument_group(runopts)

    if _debug is not None: return parser.parse_args(_debug)
    else: return parser.parse_args()


if __name__ == '__main__':
    options = get_options()
    run(options)
