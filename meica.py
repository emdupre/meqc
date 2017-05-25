import os
import sys
import os.path
import re
from re import split as resplit
import commands
from optparse import OptionParser,OptionGroup,SUPPRESS_HELP
# import ipdb

__version__="v3.2 beta1"
welcome_block="""
# Multi-Echo ICA, Version %s
#
# Kundu, P., Brenowitz, N.D., Voon, V., Worbe, Y., Vertes, P.E., Inati, S.J.,
# Saad, Z.S., Bandettini, P.A. & Bullmore, E.T. Integrated strategy for
# improving functional connectivity mapping using multiecho fMRI. PNAS (2013).
# http://dx.doi.org/10.1073/pnas.1301725110
#
# Kundu, P., Inati, S.J., Evans, J.W., Luh, W.M. & Bandettini, P.A.
# Differentiating BOLD and non-BOLD signals in fMRI time series using
# multi-echo EPI. NeuroImage (2011).
# http://dx.doi.org/10.1016/j.neuroimage.2011.12.028
#
# meica.py version %s (c) 2014 Prantik Kundu
# PROCEDURE 1 : Preprocess multi-echo datasets and apply multi-echo ICA based
# on spatial concatenation
# -Check arguments, input filenames, and filesystem for dependencies
# -Calculation of motion parameters based on images with highest constrast
# -Application of motion correction and coregistration parameters
# -Misc. EPI preprocessing (temporal alignment, smoothing, etc) in appropriate
# order
# -Compute PCA and ICA in conjuction with TE-dependence analysis
""" % (__version__,__version__)


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


# def getdsname(e_ii,prefixonly=False, shorthand_dsin=True):
#     if shorthand_dsin:
#         dsname = '%s%s%s%s' % (prefix,datasets[e_ii],trailing,isf)
#     else:
#         dsname = datasets_in[e_ii]
#     if prefixonly:
#         return fparse(dsname)
#     else: return dsname


# Configure options and help dialog
parser = OptionParser()
# Base processing options
parser.add_option('-e',"",
                  dest='tes',
                  help="Echo times in ms. ex: -e 14.5,38.5,62.5  ",default='')
parser.add_option('-d',"",
                  dest='dsinputs',
                  help="Input datasets. ex: -d RESTe[123].nii.gz",default='')
parser.add_option('-a',"",
                  dest='anat',
                  help="(Optional) anatomical dataset. " +
                       "ex: -a mprage.nii.gz",default='')
parser.add_option('-b',"",
                  dest='basetime',
                  help="Time to steady-state equilibration in seconds(s) or " +
                       "volumes(v). Default 0. ex: -b 4v",default='0')
parser.add_option('',"--MNI",
                  dest='mni',
                  action='store_true',
                  help="Warp to MNI standard space.",default=False)
parser.add_option('',"--strict",
                  dest='strict',
                  action='store_true',
                  help="Use strict component selection, suitable with " +
                       "large-voxel, high-SNR data",default=False)

# Extended options for processing
extopts = OptionGroup(parser,"Additional processing options")
extopts.add_option('',"--qwarp",
                   dest='qwarp',
                   action='store_true',
                   help="Nonlinear warp to standard space using QWarp, " +
                        "requires --MNI or --space).",default=False)
extopts.add_option('',"--native",
                   dest='native',
                   action='store_true',
                   help="Output native space results in addition to " +
                        "standard space results.",default=False)
extopts.add_option('',"--space",
                   dest='space',
                   help="Path to specific standard space template for " +
                        "affine anatomical normalization.",default=False)
extopts.add_option('',"--fres",
                   dest='fres',
                   help="Specify functional voxel dimensions in mm (iso.) " +
                        "for resampling during preprocessing." +
                        "ex: --fres=2.5", default=False)
extopts.add_option('',"--no_skullstrip",
                   action="store_true",
                   dest='no_skullstrip',
                   help="Anat is intensity-normalized and skull-stripped" +
                        "(for use with '-a' flag).",default=False)
extopts.add_option('',"--no_despike",
                   action="store_true",
                   dest='no_despike',
                   help="Do not de-spike functional data. " +
                        "Default is to de-spike, recommended.",default=False)
extopts.add_option('',"--no_axialize",
                   action="store_true",
                   dest='no_axialize',
                   help="Do not re-write dataset in axial-first order. " +
                        "Default is to axialize, recommended.",default=False)
extopts.add_option('',"--mask_mode",
                   dest='mask_mode',
                   help="Mask functional with help from anatomical or " +
                        "standard space images. " +
                        "Options: 'anat' or 'template'.",default='func')
extopts.add_option('',"--coreg_mode",
                   dest='coreg_mode',
                   help="Coregistration with Local Pearson and T2* weights " +
                        "(default), or use align_epi_anat.py (edge method)." +
                        "Options: 'lp-t2s' or 'aea'",default='lp-t2s')
extopts.add_option('',"--smooth",
                   dest='FWHM',
                   help="FWHM smoothing with 3dBlurInMask. Default none. " +
                        "ex: --smooth 3mm ",default='0mm')
extopts.add_option('',"--align_base",
                   dest='align_base',
                   help="Explicitly specify base dataset for volume " +
                        "registration",default='')
extopts.add_option('',"--TR",
                   dest='TR',
                   help="TR. Read by default from dataset header",default='')
extopts.add_option('',"--tpattern",
                   dest='tpattern',
                   help="Slice timing (i.e. alt+z, see 3dTshift -help). " +
                   "Default from header.",default='')
extopts.add_option('',"--align_args",
                   dest='align_args',
                   help="Additional arguments to anatomical-functional " +
                        "co-registration routine",default='')
extopts.add_option('',"--ted_args",
                   dest='ted_args',
                   help="Additional arguments to TE-dependence analysis",default='')

# Additional, extended preprocessing options-- no help provided, caveat emptor
extopts.add_option('',"--select_only",
                   dest='select_only',
                   action='store_true',
                   help=SUPPRESS_HELP,default=False)
extopts.add_option('',"--tedica_only",
                   dest='tedica_only',
                   action='store_true',
                   help=SUPPRESS_HELP,default=False)
extopts.add_option('',"--export_only",
                   dest='export_only',
                   action='store_true',
                   help=SUPPRESS_HELP,default=False)
extopts.add_option('',"--daw",
                   dest='daw',
                   help=SUPPRESS_HELP,default='10')
extopts.add_option('',"--tlrc",
                   dest='space',
                   help=SUPPRESS_HELP,default=False)  # For backwards compat.
extopts.add_option('',"--highpass",
                   dest='highpass',
                   help=SUPPRESS_HELP,default=0.0)
extopts.add_option('',"--detrend",
                   dest='detrend',
                   help=SUPPRESS_HELP,default=0.)
extopts.add_option('',"--initcost",
                   dest='initcost',
                   help=SUPPRESS_HELP,default='tanh')
extopts.add_option('',"--finalcost",
                   dest='finalcost',
                   help=SUPPRESS_HELP,default='tanh')
extopts.add_option('',"--sourceTEs",
                   dest='sourceTEs',
                   help=SUPPRESS_HELP,default='-1')
parser.add_option_group(extopts)

# Extended options for running
runopts = OptionGroup(parser,"Run options")
runopts.add_option('',"--prefix",
                   dest='prefix',
                   help="Prefix for final ME-ICA output datasets.",default='')
runopts.add_option('',"--cpus",
                   dest='cpus',
                   help="Maximum number of CPUs (OpenMP threads) to use. " +
                   "Default 2.",default='2')
runopts.add_option('',"--label",
                   dest='label',
                   help="Label to tag ME-ICA analysis folder.",default='')
runopts.add_option('',"--test_proc",
                   action="store_true",
                   dest='test_proc',
                   help="Align, preprocess 1 dataset then exit.",default=False)
runopts.add_option('',"--script_only",
                   action="store_true",
                   dest='script_only',
                   help="Generate script only, then exit",default=0)
runopts.add_option('',"--pp_only",
                   action="store_true",
                   dest='pp_only',
                   help="Preprocess only, then exit.",default=False)
runopts.add_option('',"--keep_int",
                   action="store_true",
                   dest='keep_int',
                   help="Keep preprocessing intermediates. " +
                   "Default delete.",default=False)
runopts.add_option('',"--RESUME",
                   dest='resume',
                   action='store_true',
                   help="Attempt to resume from normalization step onwards, " +
                   "overwriting existing")
runopts.add_option('',"--OVERWRITE",
                   dest='overwrite',
                   action="store_true",
                   help="Overwrite existing meica directory.",default=False)
parser.add_option_group(runopts)

(options,args) = parser.parse_args()

if __name__ == '__main__':

    # Show a welcome message
    print(welcome_block)

    # Parse dataset input names
    # if options.dsinputs=='' or options.TR==0:
    #     print("Need at least dataset inputs and TE. Try meica.py -h")
    #     sys.exit()
    # if os.path.abspath(os.path.curdir).__contains__('meica.'):
    #     print("*+ You are inside a ME-ICA directory! " +
    #           "Please leave this directory and rerun.")
    #     sys.exit()

    # Parse shorthand input file specification and TEs
    tes = options.tes.split(',')
    outprefix=options.prefix
    if '[' in options.dsinputs:
        shorthand_dsin = True
        dsinputs=prefix(options.dsinputs)
        prefix=resplit(r'[\[\],]',dsinputs)[0]
        datasets=resplit(r'[\[\],]',dsinputs)[1:-1]
        trailing=resplit(r'[\]+]',dsinputs)[-1]
        isf= dssuffix(options.dsinputs)
        setname=prefix+''.join(datasets)+trailing+options.label
    else:
        # Parse longhand input file specificiation
        shorthand_dsin = False
        datasets_in = options.dsinputs.split(',')
        datasets = [str(vv+1) for vv in range(len(tes))]
        prefix = prefix(datasets_in[0])
        isf = dssuffix(datasets_in[0])
        if '.nii' in isf: isf='.nii'
        trailing=''
        setname=prefix+options.label

    if not shorthand_dsin and len(datasets)!=len(datasets_in):
        print "*+ Can't understand dataset specification. Try double quotes around -d argument."
        sys.exit()

    if len(options.tes.split(','))!=len(datasets):
        print "*+ Number of TEs and input datasets must be equal and matched in order. Or try double quotes around -d argument."
        sys.exit()

    # Prepare script
    startdir=rstrip(popen('pwd').readlines()[0])
    meicadir=os.path.dirname(os.path.abspath(os.path.expanduser(sys.argv[0])))
    headsl = []  # Header lines and command list
    sl = []    # Script command list
    runcmd = " ".join(sys.argv).replace(options.dsinputs,r'"%s"' % options.dsinputs ).replace('"',r'"')
    headsl.append('#'+runcmd)
    headsl.append(welcome_block)
    osf='.nii.gz'  # Using NIFTI outputs

    # Check if input files exist
    notfound=0
    for ds_ii in range(len(datasets)):
        if commands.getstatusoutput('3dinfo %s' % (getdsname(ds_ii)))[0]!=0:
            print "*+ Can't find/load dataset %s !" % (getdsname(ds_ii))
            notfound+=1
    if options.anat!='' and commands.getstatusoutput('3dinfo %s' % (options.anat))[0]!=0:
        print "*+ Can't find/load anatomical dataset %s !" % (options.anat)
        notfound+=1
    if notfound!=0:
        print "++ EXITING. Check dataset names."
        sys.exit()

    # Check dependencies
    grayweight_ok = 0
    if not options.skip_check:
        dep_check()
        print "++ Continuing with preprocessing."
    else:
        print "*+ Skipping dependency checks."
        grayweight_ok = 1

    # Parse timing arguments
    if options.TR!='':tr=float(options.TR)
    else:
        tr=float(os.popen('3dinfo -tr %s' % (getdsname(0))).readlines()[0].strip())
        options.TR=str(tr)
    if 'v' in str(options.basetime):
        basebrik = int(options.basetime.strip('v'))
    else:
        timetoclip=0
        timetoclip = float(options.basetime.strip('s'))
        basebrik=int(round(timetoclip/tr))

    # Misc. command parsing
    if options.mni: options.space='MNI_caez_N27+tlrc'
    if options.qwarp and (options.anat=='' or not options.space):
        print "*+ Can't specify Qwarp nonlinear coregistration without anatomical and SPACE template!"
        sys.exit()

    if not options.mask_mode in ['func','anat','template']:
        print "*+ Mask mode option '%s' is not recognized!" % options.mask_mode
        sys.exit()
    if options.mask_mode=='' and options.space:
        options.mask_mode='template'

    # Parse alignment options
    if options.coreg_mode == 'aea': options.t2salign=False
    elif 'lp' in options.coreg_mode : options.t2salign=True
    align_base = basebrik
    align_interp='cubic'
    align_interp_final='wsinc5'
    oblique_epi_read = 0
    oblique_anat_read = 0
    zeropad_opts = " -I %s -S %s -A %s -P %s -L %s -R %s " % (tuple([1]*6))
    if options.anat!='':
        oblique_anat_read = int(os.popen('3dinfo -is_oblique %s' % (options.anat)).readlines()[0].strip())
        epicm = [float(coord) for coord in os.popen("3dCM %s" % (getdsname(0))).readlines()[0].strip().split()]
        anatcm = [float(coord) for coord in os.popen("3dCM %s" % (options.anat)).readlines()[0].strip().split()]
        maxvoxsz = float(os.popen("3dinfo -dk %s" % (getdsname(0))).readlines()[0].strip())
        deltas = [abs(epicm[0]-anatcm[0]),abs(epicm[1]-anatcm[1]),abs(epicm[2]-anatcm[2])]
        cmdist = 20+sum([dd**2. for dd in deltas])**.5
        cmdif =  max(abs(epicm[0]-anatcm[0]),abs(epicm[1]-anatcm[1]),abs(epicm[2]-anatcm[2]))
        addslabs = abs(int(cmdif/maxvoxsz))+10
           zeropad_opts=" -I %s -S %s -A %s -P %s -L %s -R %s " % (tuple([addslabs]*6))
    oblique_epi_read = int(os.popen('3dinfo -is_oblique %s' % (getdsname(0))).readlines()[0].strip())
    if oblique_epi_read or oblique_anat_read: 
        oblique_mode = True
        headsl.append("echo Oblique data detected.")
    else: oblique_mode = False
    if options.fres:
        if options.qwarp: qwfres="-dxyz %s" % options.fres
        alfres = "-mast_dxyz %s" % options.fres
    else:
        if options.qwarp: qwfres="-dxyz ${voxsize}"  # See section called "Preparing functional masking for this ME-EPI run"
        alfres="-mast_dxyz ${voxsize}"
    if options.anat=='' and options.mask_mode!='func':
        print "*+ Can't do anatomical-based functional masking without an anatomical!"
        sys.exit()
    if options.anat and options.space and options.qwarp: valid_qwarp_mode = True
    else: valid_qwarp_mode = False

    # Detect if current AFNI has old 3dNwarpApply
    if " -affter aaa  = *** THIS OPTION IS NO LONGER AVAILABLE" in commands.getstatusoutput("3dNwarpApply -help")[1]: old_qwarp = False
    else: old_qwarp = True

    # Detect AFNI direcotry
    afnidir = os.path.dirname(os.popen('which 3dSkullStrip').readlines()[0])

    # Prepare script and enter MEICA directory
    logcomment("Set up script run environment",level=1)
    headsl.append('set -e')
    headsl.append('export OMP_NUM_THREADS=%s' % (options.cpus))
    headsl.append('export MKL_NUM_THREADS=%s' % (options.cpus))
    headsl.append('export DYLD_FALLBACK_LIBRARY_PATH=%s' % (afnidir))
    headsl.append('export AFNI_3dDespike_NEW=YES')
    if not options.resume and not options.tedica_only and not options.select_only and not options.export_only:
        if options.overwrite:
            headsl.append('rm -rf meica.%s' % (setname))
        else:
            headsl.append("if [[ -e meica.%s ]]; then echo ME-ICA directory exists, exiting; exit; fi" % (setname))
        headsl.append('mkdir -p meica.%s' % (setname))
    if options.resume:
        headsl.append('if [ ! -e meica.%s/_meica.orig.sh ]; then mv `ls meica.%s/_meica*sh` meica.%s/_meica.orig.sh; fi' % (setname,setname,setname))
    if not options.tedica_only and not options.select_only: headsl.append("cp _meica_%s.sh meica.%s/" % (setname,setname))
    headsl.append("cd meica.%s" % setname)
    thecwd= "%s/meica.%s" % (getcwd(),setname)

    ica_datasets = sorted(datasets)

    # Parse anatomical processing options, process anatomical
    if options.anat != '':
        logcomment("Deoblique, unifize, skullstrip, and/or autobox anatomical, in starting directory (may take a little while)", level=1)
        nsmprage = options.anat
        anatprefix=prefix(nsmprage)
        pathanatprefix="%s/%s" % (startdir,anatprefix)
        if oblique_mode:
            sl.append("if [ ! -e %s_do.nii.gz ]; then 3dWarp -overwrite -prefix %s_do.nii.gz -deoblique %s/%s; fi" % (pathanatprefix,pathanatprefix,startdir,nsmprage))
            nsmprage="%s_do.nii.gz" % (anatprefix)
        if not options.no_skullstrip: 
            sl.append("if [ ! -e %s_ns.nii.gz ]; then 3dUnifize -overwrite -prefix %s_u.nii.gz %s/%s; 3dSkullStrip  -shrink_fac_bot_lim 0.3 -orig_vol -overwrite -prefix %s_ns.nii.gz -input %s_u.nii.gz; 3dAutobox -overwrite -prefix %s_ns.nii.gz %s_ns.nii.gz; fi" % (pathanatprefix,pathanatprefix,startdir,nsmprage,pathanatprefix,pathanatprefix,pathanatprefix,pathanatprefix))
            nsmprage="%s_ns.nii.gz" % (anatprefix)

    # Copy in functional datasets as NIFTI (if not in NIFTI already), calculate rigid body alignment
    vrbase=getdsname(0,True)
    logcomment("Copy in functional datasets, reset NIFTI tags as needed", level=1)
    for e_ii in range(len(datasets)):
        ds = datasets[e_ii]
        sl.append("3dcalc -a %s/%s -expr 'a' -prefix ./%s.nii" % (startdir,getdsname(e_ii),getdsname(e_ii,True) )   )
        if '.nii' in isf: 
            sl.append("nifti_tool -mod_hdr -mod_field sform_code 1 -mod_field qform_code 1 -infiles ./%s.nii -overwrite" % (  getdsname(e_ii,True)  ))
    isf = '.nii'

    logcomment("Calculate and save motion and obliquity parameters, despiking first if not disabled, and separately save and mask the base volume",level=1)
    # Determine input to volume registration
    vrAinput = "./%s%s" % (vrbase,isf)
    # Compute obliquity matrix
    if oblique_mode:
        if options.anat!='': sl.append("3dWarp -verb -card2oblique %s[0] -overwrite  -newgrid 1.000000 -prefix ./%s_ob.nii.gz %s/%s | \grep  -A 4 '# mat44 Obliquity Transformation ::'  > %s_obla2e_mat.1D" % (vrAinput,anatprefix,startdir,nsmprage,prefix))
        else: sl.append("3dWarp -overwrite -prefix %s -deoblique %s" % (vrAinput,vrAinput))
    # Despike and axialize
    if not options.no_despike:
        sl.append("3dDespike -overwrite -prefix ./%s_vrA%s %s "  % (vrbase,osf,vrAinput))
        vrAinput = "./%s_vrA%s" % (vrbase,osf)
    if not options.no_axialize:
        sl.append("3daxialize -overwrite -prefix ./%s_vrA%s %s" % (vrbase,osf,vrAinput))
        vrAinput = "./%s_vrA%s" % (vrbase,osf)
    # Set eBbase
    external_eBbase=False
    if options.align_base!='':
        if options.align_base.isdigit():
            basevol = '%s[%s]' % (vrAinput,options.align_base)
        else:
            basevol = options.align_base
            external_eBbase=True
    else:
        basevol = '%s[%s]' % (vrAinput,basebrik)
    sl.append("3dcalc -a %s  -expr 'a' -prefix eBbase.nii.gz "  % (basevol))
    if external_eBbase:
        if oblique_mode: sl.append("3dWarp -overwrite -deoblique eBbase.nii.gz eBbase.nii.gz")
        if not options.no_axialize: sl.append("3daxialize -overwrite -prefix eBbase.nii.gz eBbase.nii.gz")
    # Compute motion parameters
    sl.append("3dvolreg -overwrite -tshift -quintic  -prefix ./%s_vrA%s -base eBbase.nii.gz -dfile ./%s_vrA.1D -1Dmatrix_save ./%s_vrmat.aff12.1D %s" % \
              (vrbase,osf,vrbase,prefix,vrAinput))
    vrAinput = "./%s_vrA%s" % (vrbase,osf)
    sl.append("1dcat './%s_vrA.1D[1..6]{%s..$}' > motion.1D " % (vrbase,basebrik))
    e2dsin = prefix+datasets[0]+trailing

    logcomment("Preliminary preprocessing of functional datasets: despike, tshift, deoblique, and/or axialize",level=1)
    # Do preliminary preproc for this run
    if shorthand_dsin: datasets.sort()
    for echo_ii in range(len(datasets)):
        # Determine dataset name
        echo = datasets[echo_ii]
        indata = getdsname(echo_ii)
        dsin = 'e'+echo
        if echo_ii==0: e1_dsin = dsin
        logcomment("Preliminary preprocessing dataset %s of TE=%sms to produce %s_ts+orig" % (indata,str(tes[echo_ii]),dsin) )
        # Pre-treat datasets: De-spike, RETROICOR in the future?
        intsname = '%s%s' % (prefix(indata),isf)
        if not options.no_despike:
            intsname = "./%s_pt.nii.gz" % prefix(indata)
            sl.append("3dDespike -overwrite -prefix %s %s%s" % (intsname,prefix(indata),isf))
        # Time shift datasets
        if options.tpattern!='':
            tpat_opt = ' -tpattern %s ' % options.tpattern
        else:
            tpat_opt = ''
        sl.append("3dTshift -heptic %s -prefix ./%s_ts+orig %s" % (tpat_opt,dsin,intsname) )
        # Force +orig label on dataset
        sl.append("3drefit -view orig %s_ts*HEAD" % (dsin))
        if oblique_mode and options.anat=="":
            sl.append("3dWarp -overwrite -deoblique -prefix ./%s_ts+orig ./%s_ts+orig" % (dsin,dsin))
        # Axialize functional dataset
        if not options.no_axialize:
            sl.append("3daxialize  -overwrite -prefix ./%s_ts+orig ./%s_ts+orig" % (dsin,dsin))
        if oblique_mode: sl.append("3drefit -deoblique -TR %s %s_ts+orig" % (options.TR,dsin))
        else: sl.append("3drefit -TR %s %s_ts+orig" % (options.TR,dsin))

    # Compute T2*, S0, and OC volumes from raw data
    logcomment("Prepare T2* and S0 volumes for use in functional masking and (optionally) anatomical-functional coregistration (takes a little while).",level=1)
    dss = datasets
    dss.sort()
    stackline=""
    for echo_ii in range(len(dss)):
        echo = datasets[echo_ii]
        dsin = 'e'+echo
        sl.append("3dAllineate -overwrite -final NN -NN -float -1Dmatrix_apply %s_vrmat.aff12.1D'{%i..%i}' -base eBbase.nii.gz -input %s_ts+orig'[%i..%i]' -prefix %s_vrA.nii.gz" % \
                    (prefix,int(basebrik),int(basebrik)+20,dsin,int(basebrik),int(basebrik)+20,dsin))
        stackline+=" %s_vrA.nii.gz" % (dsin)
    sl.append("3dZcat -prefix basestack.nii.gz %s" % (stackline))
    sl.append("%s %s -d basestack.nii.gz -e %s" % (sys.executable, '/'.join([meicadir,'meica.libs','t2smap.py']),options.tes))
    sl.append("3dUnifize -prefix ./ocv_uni+orig ocv.nii")
    sl.append("3dSkullStrip -no_avoid_eyes -prefix ./ocv_ss.nii.gz -overwrite -input ocv_uni+orig")
    sl.append("3dcalc -overwrite -a t2svm.nii -b ocv_ss.nii.gz -expr 'a*ispositive(a)*step(b)' -prefix t2svm_ss.nii.gz" )
    sl.append("3dcalc -overwrite -a s0v.nii -b ocv_ss.nii.gz -expr 'a*ispositive(a)*step(b)' -prefix s0v_ss.nii.gz" )
    if not options.no_axialize:
        sl.append("3daxialize -overwrite -prefix t2svm_ss.nii.gz t2svm_ss.nii.gz")
        sl.append("3daxialize -overwrite -prefix ocv_ss.nii.gz ocv_ss.nii.gz")
        sl.append("3daxialize -overwrite -prefix s0v_ss.nii.gz s0v_ss.nii.gz")

    # Resume from here on
    if options.resume:
        sl = []
        sl.append("export AFNI_DECONFLICT=OVERWRITE")

    # Calculate affine anatomical warp if anatomical provided, then combine motion correction and coregistration parameters 
    if options.anat!='':
        # Copy in anatomical and make sure its in +orig space
        logcomment("Copy anatomical into ME-ICA directory and process warps",level=1)
        sl.append("cp %s/%s* ." % (startdir,nsmprage))
        abmprage = nsmprage
        refanat = nsmprage
        if options.space:
            sl.append("afnibinloc=`which 3dSkullStrip`")
            if '/' in options.space:
                sl.append("ll=\"%s\"; templateloc=${ll%%/*}/" % options.space)
                options.space=options.space.split('/')[-1]
              else:
                sl.append("templateloc=${afnibinloc%/*}")
            atnsmprage = "%s_at.nii.gz" % (prefix(nsmprage))
            if not dssuffix(nsmprage).__contains__('nii'): sl.append("3dcalc -float -a %s -expr 'a' -prefix %s.nii.gz" % (nsmprage,prefix(nsmprage)))
            logcomment("If can't find affine-warped anatomical, copy native anatomical here, compute warps (takes a while) and save in start dir. ; otherwise link in existing files")
            sl.append("if [ ! -e %s/%s ]; then \@auto_tlrc -no_ss -init_xform AUTO_CENTER -base ${templateloc}/%s -input %s.nii.gz -suffix _at" % (startdir,atnsmprage,options.space,prefix(nsmprage)))
            sl.append("cp %s.nii %s" % (prefix(atnsmprage),startdir))
            sl.append("gzip -f %s/%s.nii" % (startdir,prefix(atnsmprage)))
            sl.append("else if [ ! -e %s/%s ]; then ln -s %s/%s .; fi" % (startdir,atnsmprage,startdir,atnsmprage))
            refanat = '%s/%s' % (startdir,atnsmprage)
            sl.append("fi")
            sl.append("3dcopy %s/%s.nii.gz %s" % (startdir,prefix(atnsmprage),prefix(atnsmprage)))
            sl.append("rm -f %s+orig.*; 3drefit -view orig %s+tlrc " % (prefix(atnsmprage),prefix(atnsmprage)) )
            sl.append("3dAutobox -overwrite -prefix ./abtemplate.nii.gz ${templateloc}/%s" % options.space)
            abmprage = 'abtemplate.nii.gz'
            if options.qwarp:
                logcomment("If can't find non-linearly warped anatomical, compute, save back; otherwise link")
                nlatnsmprage="%s_atnl.nii.gz" % (prefix(nsmprage))
                sl.append("if [ ! -e %s/%s ]; then " % (startdir,nlatnsmprage))
                logcomment("Compute non-linear warp to standard space using 3dQwarp (get lunch, takes a while) ")
                sl.append("3dUnifize -overwrite -GM -prefix ./%su.nii.gz %s/%s" % (prefix(atnsmprage),startdir,atnsmprage))  
                sl.append("3dQwarp -iwarp -overwrite -resample -useweight -blur 2 2 -duplo -workhard -base ${templateloc}/%s -prefix %s/%snl.nii.gz -source ./%su.nii.gz" % (options.space,startdir,prefix(atnsmprage),prefix(atnsmprage)))
                sl.append("fi")
                sl.append("if [ ! -e %s/%s ]; then ln -s %s/%s .; fi" % (startdir,nlatnsmprage,startdir,nlatnsmprage))
                refanat = '%s/%snl.nii.gz' % (startdir,prefix(atnsmprage))
        
        # Set anatomical reference for anatomical-functional co-registration
        if oblique_mode: alnsmprage = "./%s_ob.nii.gz" % (anatprefix)
        else: alnsmprage = "%s/%s" % (startdir,nsmprage)
        if options.coreg_mode=='lp-t2s': 
            ama_alnsmprage = alnsmprage
            if not options.no_axialize:
                ama_alnsmprage = os.path.basename(alnsmprage)
                sl.append("3daxialize -overwrite -prefix ./%s %s" % (ama_alnsmprage,alnsmprage))
            t2salignpath = 'meica.libs/alignp_mepi_anat.py'
            sl.append("%s %s -t t2svm_ss.nii.gz -a %s -p mepi %s" % \
                (sys.executable, '/'.join([meicadir,t2salignpath]),ama_alnsmprage,options.align_args))
            sl.append("cp alignp.mepi/mepi_al_mat.aff12.1D ./%s_al_mat.aff12.1D" % anatprefix)
        elif options.coreg_mode=='aea':
            logcomment("Using AFNI align_epi_anat.py to drive anatomical-functional coregistration ")
            sl.append("3dcopy %s ./ANAT_ns+orig " % alnsmprage)
            sl.append("align_epi_anat.py -anat2epi -volreg off -tshift off -deoblique off -anat_has_skull no -save_script aea_anat_to_ocv.tcsh -anat ANAT_ns+orig -epi ocv_uni+orig -epi_base 0 %s" % (options.align_args) )
            sl.append("cp ANAT_ns_al_mat.aff12.1D %s_al_mat.aff12.1D" % (anatprefix))
        if options.space: 
            tlrc_opt = "%s/%s::WARP_DATA -I" % (startdir,atnsmprage)
            inv_tlrc_opt = "%s/%s::WARP_DATA" % (startdir,atnsmprage)
            sl.append("cat_matvec -ONELINE %s > %s/%s_xns2at.aff12.1D" % (tlrc_opt,startdir,anatprefix))
            sl.append("cat_matvec -ONELINE %s > %s_xat2ns.aff12.1D" % (inv_tlrc_opt,anatprefix))
        else: tlrc_opt = ""
        if oblique_mode: oblique_opt = "%s_obla2e_mat.1D" % prefix
        else: oblique_opt = ""
        # pre-Mar 3, 2017, included tlrc affine warp in preprocessing. For new export flexiblity, will do tlrc_opt at export.
        # pre-Mar 3, 2017 version: sl.append("cat_matvec -ONELINE  %s %s %s_al_mat.aff12.1D -I > %s_wmat.aff12.1D" % (tlrc_opt,oblique_opt,anatprefix,prefix))
        sl.append("cat_matvec -ONELINE  %s %s_al_mat.aff12.1D -I > %s_wmat.aff12.1D" % (oblique_opt,anatprefix,prefix))
        if options.anat: sl.append("cat_matvec -ONELINE  %s %s_al_mat.aff12.1D -I  %s_vrmat.aff12.1D  > %s_vrwmat.aff12.1D" % (oblique_opt,anatprefix,prefix,prefix))
    else: sl.append("cp %s_vrmat.aff12.1D %s_vrwmat.aff12.1D" % (prefix,prefix))

    # Preprocess datasets
    if shorthand_dsin: datasets.sort()
    logcomment("Extended preprocessing of functional datasets",level=1)

    # Compute grand mean scaling factor
    sl.append("3dBrickStat -mask eBbase.nii.gz -percentile 50 1 50 %s_ts+orig[%i] > gms.1D" % (e1_dsin,basebrik))
    sl.append("gms=`cat gms.1D`; gmsa=($gms); p50=${gmsa[1]}")

    # Set resolution variables
    sl.append("voxsize=`ccalc .85*$(3dinfo -voxvol eBbase.nii.gz)**.33`") #Set voxel size for decomp to slightly upsampled version of isotropic appx of native resolution so GRAPPA artifact is not at Nyquist
    sl.append("voxdims=\"`3dinfo -adi eBbase.nii.gz` `3dinfo -adj eBbase.nii.gz` `3dinfo -adk eBbase.nii.gz`\"")
    sl.append("echo $voxdims > voxdims.1D")
    sl.append("echo $voxsize > voxsize.1D")

    for echo_ii in range(len(datasets)):

        # Determine dataset name
        echo = datasets[echo_ii]
        indata = getdsname(echo_ii)
        dsin = 'e'+echo  # Note using same dsin as in time shifting 

        if echo_ii == 0: 
            logcomment("Preparing functional masking for this ME-EPI run",2 )
            # abmprage = refanat = nsmprage  # Update as of Mar 3, 2017, to move to all native analysis        
            if options.anat: almaster="-master %s" % nsmprage #abmprage
            else: almaster=""
            #print 'almaster line is', almaster  # DEBUG
            #print 'refanat line is', refanat  # DEBUG
            sl.append("3dZeropad %s -prefix eBvrmask.nii.gz ocv_ss.nii.gz[0]" % (zeropad_opts))
            # Create base mask
            #if valid_qwarp_mode: 
            #    if old_qwarp: nwarpstring = " -nwarp '%s/%s_WARP.nii.gz' -affter '%s_wmat.aff12.1D'" % (startdir,prefix(nlatnsmprage),prefix)
            #    else: nwarpstring = " -nwarp '%s/%s_WARP.nii.gz %s_wmat.aff12.1D' " % (startdir,prefix(nlatnsmprage),prefix)
            if options.anat:
                sl.append("3dAllineate -overwrite -final %s -%s -float -1Dmatrix_apply %s_wmat.aff12.1D -base %s -input eBvrmask.nii.gz -prefix ./eBvrmask.nii.gz %s %s" % \
                ('NN','NN',prefix,nsmprage,almaster,alfres))
                if options.t2salign or options.mask_mode!='func':
                    sl.append("3dAllineate -overwrite -final %s -%s -float -1Dmatrix_apply %s_wmat.aff12.1D -base eBvrmask.nii.gz -input t2svm_ss.nii.gz -prefix ./t2svm_ss_vr.nii.gz %s %s" % \
                    ('NN','NN',prefix,almaster,alfres))
                    sl.append("3dAllineate -overwrite -final %s -%s -float -1Dmatrix_apply %s_wmat.aff12.1D -base eBvrmask.nii.gz -input ocv_uni+orig -prefix ./ocv_uni_vr.nii.gz %s %s" % \
                    ('NN','NN',prefix,almaster,alfres))
                    sl.append("3dAllineate -overwrite -final %s -%s -float -1Dmatrix_apply %s_wmat.aff12.1D -base eBvrmask.nii.gz -input s0v_ss.nii.gz -prefix ./s0v_ss_vr.nii.gz %s %s" % \
                    ('NN','NN',prefix,almaster,alfres))
            # Fancy functional masking
            if options.anat and options.mask_mode != 'func':
                if options.space and options.mask_mode == 'template':
                    sl.append("3dfractionize -overwrite -template eBvrmask.nii.gz -input abtemplate.nii.gz -prefix ./anatmask_epi.nii.gz -clip 1")
                    sl.append("3dAllineate -overwrite -float -1Dmatrix_apply %s_xat2ns.aff12.1D -base eBvrmask.nii.gz -input anatmask_epi.nii.gz -prefix anatmask_epi.nii.gz -overwrite" % (anatprefix) )
                    logcomment("Preparing functional mask using information from standard space template (takes a little while)")
                if options.mask_mode == 'anat':
                    sl.append("3dfractionize -template eBvrmask.nii.gz -input %s -prefix ./anatmask_epi.nii.gz -clip 0.5" %  (nsmprage) )
                    logcomment("Preparing functional mask using information from anatomical (takes a little while)")
                sl.append("3dBrickStat -mask eBvrmask.nii.gz -percentile 50 1 50 t2svm_ss_vr.nii.gz > t2s_med.1D")
                sl.append("3dBrickStat -mask eBvrmask.nii.gz -percentile 50 1 50 s0v_ss_vr.nii.gz > s0v_med.1D")
                sl.append("t2sm=`cat t2s_med.1D`; t2sma=($t2sm); t2sm=${t2sma[1]}")
                sl.append("s0vm=`cat s0v_med.1D`; s0vma=($s0vm); s0vm=${s0vma[1]}")
                sl.append("3dcalc -a ocv_uni_vr.nii.gz -b anatmask_epi.nii.gz -c t2svm_ss_vr.nii.gz -d s0v_ss_vr.nii.gz -expr \"a-a*equals(equals(b,0)+isnegative(c-${t2sm})+ispositive(d-${s0vm}),3)\" -overwrite -prefix ocv_uni_vr.nii.gz ")
                sl.append("3dSkullStrip -no_avoid_eyes -overwrite -input ocv_uni_vr.nii.gz -prefix eBvrmask.nii.gz ")
                if options.fres: resstring = "-dxyz %s %s %s" % (options.fres,options.fres,options.fres)
                else: resstring = "-dxyz ${voxsize} ${voxsize} ${voxsize}"
                sl.append("3dresample -overwrite -master %s %s -input eBvrmask.nii.gz -prefix eBvrmask.nii.gz" % (nsmprage,resstring))

            logcomment("Trim empty space off of mask dataset and/or resample")
            sl.append("3dAutobox -overwrite -prefix eBvrmask%s eBvrmask%s" % (osf,osf) ) 
            resstring = "-dxyz ${voxsize} ${voxsize} ${voxsize}"
            sl.append("3dresample -overwrite -master eBvrmask.nii.gz %s -input eBvrmask.nii.gz -prefix eBvrmask.nii.gz" % (resstring)) #want this isotropic so spatial ops in select_model not confounded
            sl.append("3dcalc -float -a eBvrmask.nii.gz -expr 'notzero(a)' -overwrite -prefix eBvrmask.nii.gz")

        #logcomment("Extended preprocessing dataset %s of TE=%sms to produce %s_in.nii.gz" % (indata,str(tes[echo_ii]),dsin),level=2 )
        logcomment("Apply combined co-registration/motion correction parameter set to %s_ts+orig" % dsin)

        sl.append("3dAllineate -final %s -%s -float -1Dmatrix_apply %s_vrwmat.aff12.1D -base eBvrmask%s -input  %s_ts+orig -prefix ./%s_vr%s" % \
            (align_interp_final,align_interp,prefix,osf,dsin,dsin,osf))
        if echo_ii == 0:
            sl.append("3dTstat -min -prefix ./%s_vr_min%s ./%s_vr%s" % (dsin,osf,dsin,osf) )
            sl.append("3dcalc -a eBvrmask.nii.gz -b %s_vr_min%s -expr 'step(a)*step(b)' -overwrite -prefix eBvrmask.nii.gz " % (dsin,osf))
        if options.FWHM=='0mm':  sl.append("3dcalc -float -overwrite -a eBvrmask.nii.gz -b ./%s_vr%s[%i..$] -expr 'step(a)*b' -prefix ./%s_sm%s " % (dsin,osf,basebrik,dsin,osf))
        else:  sl.append("3dBlurInMask -fwhm %s -mask eBvrmask%s -prefix ./%s_sm%s ./%s_vr%s[%i..$]" % (options.FWHM,osf,dsin,osf,dsin,osf,basebrik))
        sl.append("3dcalc -float -overwrite -a ./%s_sm%s -expr \"a*10000/${p50}\" -prefix ./%s_sm%s" % (dsin,osf,dsin,osf))
        sl.append("3dTstat -prefix ./%s_mean%s ./%s_sm%s" % (dsin,osf,dsin,osf))
        if options.detrend: sl.append("3dDetrend -polort %s -overwrite -prefix ./%s_sm%s ./%s_sm%s " % (options.detrend,dsin,osf,dsin,osf) )
        if options.highpass: sl.append("3dBandpass -prefix ./%s_in%s %f 99 ./%s_sm%s " % (dsin,osf,float(options.highpass),dsin,osf) )
        else: sl.append("mv %s_sm%s %s_in%s" % (dsin,osf,dsin,osf))
        sl.append("3dcalc -float -overwrite -a ./%s_in%s -b ./%s_mean%s -expr 'a+b' -prefix ./%s_in%s" % (dsin,osf,dsin,osf,dsin,osf))
        sl.append("3dTstat -stdev -prefix ./%s_std%s ./%s_in%s" % (dsin,osf,dsin,osf))
        
        if options.test_proc: sl.append("exit")
        if not (options.test_proc or options.keep_int): sl.append("rm -f %s_pt.nii.gz %s_vr%s %s_sm%s" % (dsin,dsin,osf,dsin,osf))


    # Spatial concatenation of datasets, this needs to get removed in future versions based on the argparse feature. 
    ica_input = "zcat_ffd.nii.gz" 
    ica_mask = "zcat_mask.nii.gz"
    zcatstring=""
    for echo in ica_datasets: 
        dsin ='e'+echo
        zcatstring = "%s ./%s_in%s" % (zcatstring,dsin,osf)
    sl.append("3dZcat -overwrite -prefix %s  %s" % (ica_input,zcatstring) )
    sl.append("3dcalc -float -overwrite -a %s[0] -expr 'notzero(a)' -prefix %s" % (ica_input,ica_mask))

    if options.pp_only: tedflag='#'
    else: tedflag = ''

    if options.resume: sl.append('rm -f TED/pcastate.pklbz')
    if options.tedica_only: sl = []

    strict_setting = ''
    if options.strict: strict_setting = '--strict'

    if os.path.exists('%s/meica.libs' % (meicadir)): tedanapath = 'meica.libs/tedana.py'
    else: tedanapath = 'tedana.py'
    logcomment("Perform TE-dependence analysis (takes a good while)",level=1)
    interactive_flag=''
    if 'IPYTHON' in args: interactive_flag =' -i -- '
    sl.append("%s%s %s %s -e %s  -d %s --sourceTEs=%s --kdaw=%s --rdaw=1 --initcost=%s --finalcost=%s --conv=2.5e-5 %s %s" % (tedflag,sys.executable,interactive_flag, '/'.join([meicadir,tedanapath]),options.tes,ica_input,options.sourceTEs,options.daw,options.initcost,options.finalcost,strict_setting, options.ted_args))
    if outprefix=='': outprefix=setname

    if options.select_only: 
        sl = []
        sl.append("%s %s %s -e %s  -d %s --sourceTEs=%s --kdaw=%s --rdaw=1 --initcost=%s --finalcost=%s --conv=2.5e-5 --mix=meica_mix.1D %s %s " % (sys.executable,interactive_flag, '/'.join([meicadir,tedanapath]),options.tes,ica_input,options.sourceTEs,options.daw,options.initcost,options.finalcost,strict_setting, options.ted_args))

    mask_dict = {}  # Need this here
    def export_result(infile,outfileprefix,comment='Created by %s' % runcmd,interp='wsinc5',disable=False,plaintext=False,export_mask='./export_mask.nii.gz'):
        tedflag = ''
        if disable == True : tedflag='#'
        native_export = options.native
        if plaintext: 
            sl.append('cp %s %s/%s' %  (infile,startdir,outfileprefix) )
            return 
        # Set output resolution parameters, either specified fres or original voxel dimensions
        if options.fres: resstring = "-dxyz %s %s %s" % (options.fres,options.fres,options.fres)
        else: resstring = "-dxyz ${voxdims}"
        to_export = []
        if options.space : export_master ="-master %s" % abmprage  #
        
        # If Qwarp, do Nwarpapply
        if valid_qwarp_mode: 
            warp_code = 'nlw'
            this_nwarpstring =" -nwarp %s/%s_xns2at.aff12.1D '%s/%s_WARP.nii.gz' " % (startdir,anatprefix,startdir,prefix(nlatnsmprage))
            sl.append("%s3dNwarpApply -overwrite %s %s %s -source %s -interp %s -prefix %s_nlw.nii " % \
                    (tedflag,this_nwarpstring,export_master,qwfres,infile,interp,outfileprefix))
            if not warp_code in mask_dict.keys(): 
                sl.append("%s3dNwarpApply -overwrite %s %s %s -source %s -interp %s -prefix %s_export_mask.nii " % \
                    (tedflag,this_nwarpstring,export_master,qwfres,'export_mask.nii.gz',interp,warp_code))
                sl.append("%snifti_tool -mod_hdr -mod_field sform_code 2 -mod_field qform_code 2 -infiles %s_export_mask.nii -overwrite" % (tedflag,warp_code))            
                mask_dict[warp_code] = '%s_export_mask.nii' % warp_code
            to_export.append(('%s_%s' % (outfileprefix,warp_code),'%s_export_mask.nii' % warp_code))
        # If there's a template space, allineate result to that space
        elif options.space:
            warp_code = 'afw'
            sl.append("%s3dAllineate -overwrite -final %s -%s -float -1Dmatrix_apply %s/%s_xns2at.aff12.1D -input %s -prefix ./%s_afw.nii %s %s" % \
                (tedflag,interp,align_interp,startdir,anatprefix,infile,outfileprefix,export_master,alfres))    
            if not warp_code in mask_dict.keys(): 
                sl.append("%s3dAllineate -overwrite -final %s -%s -float -1Dmatrix_apply %s/%s_xns2at.aff12.1D -input %s -prefix ./%s_export_mask.nii %s %s" % \
                    (tedflag,interp,align_interp,startdir,anatprefix,'export_mask.nii.gz',warp_code,export_master,alfres))    
                sl.append("%snifti_tool -mod_hdr -mod_field sform_code 2 -mod_field qform_code 2 -infiles %s_export_mask.nii -overwrite" % (tedflag,warp_code))            
                mask_dict[warp_code] = '%s_export_mask.nii' % warp_code
            to_export.append(('%s_%s' % (outfileprefix,warp_code),'%s_export_mask.nii' % warp_code))
        # Otherwise resample    
        else: native_export= True
        if native_export:
            if options.anat: 
                native_suffix = 'nat'
                export_master = "-master %s" % nsmprage
            else: 
                native_suffix = 'epi'
                export_master = ''
            sl.append("%s3dresample -rmode Li -overwrite %s %s -input %s -prefix %s_%s.nii" % (tedflag,export_master,resstring,infile,outfileprefix,native_suffix))
            warp_code = native_suffix
            if not warp_code in mask_dict.keys(): 
                #sl.append('3dresample -overwrite -prefix %s_export_mask.nii -rmode Li -master %s_%s.nii -input export_mask.nii.gz' % (warp_code,outfileprefix,warp_code))
                sl.append("%s3dresample -rmode Li -overwrite %s %s -input %s -prefix %s_export_mask.nii" % (tedflag,export_master,resstring,'export_mask.nii.gz',warp_code))        
                mask_dict[warp_code] = '%s_export_mask.nii' % warp_code
            to_export.append(('%s_%s' % (outfileprefix,warp_code),'%s_export_mask.nii' % warp_code))
        for P,P_mask in to_export:
            sl.append("%s3dNotes -h \'%s\' %s.nii" % (tedflag,comment,P))
            if options.anat!='' and options.space!=False and '_nat' not in P:
                sl.append("%snifti_tool -mod_hdr -mod_field sform_code 2 -mod_field qform_code 2 -infiles %s.nii -overwrite" % (tedflag,P))
            sl.append("3dcalc -overwrite -a %s -b %s.nii -expr 'ispositive(a-.5)*b' -prefix %s.nii ; gzip -f %s.nii; mv %s.nii.gz %s" % (P_mask,P,P,P,P,startdir)  )

    if options.export_only:
            sl = []

    sl.append("voxdims=\"`3dinfo -adi eBbase.nii.gz` `3dinfo -adj eBbase.nii.gz` `3dinfo -adk eBbase.nii.gz`\"")
    sl.append("echo $voxdims > voxdims.1D")
    # Make the export mask
    sl.append("3dcalc -float -a TED/ts_OC.nii[0] -overwrite -expr 'notzero(a)' -prefix ./export_mask.nii.gz")
    logcomment("Copying results to start directory",level=1)
    export_result('TED/ts_OC.nii','%s_tsoc' % (outprefix), "T2* weighted average of ME time series, produced by ME-ICA %s" % __version__, disable=options.pp_only) 
    export_result('TED/dn_ts_OC.nii','%s_medn' % (outprefix), "Denoised timeseries (including thermal noise), produced by ME-ICA %s" % __version__, disable=options.pp_only) 
    export_result('TED/dn_ts_OC_T1c.nii','%s_T1c_medn' % (outprefix), "Denoised timeseries with T1 equilibration correction (including thermal noise), produced by ME-ICA %s" % __version__, disable=options.pp_only) 
    export_result('TED/hik_ts_OC_T1c.nii','%s_hikts' % (outprefix), "Denoised timeseries with T1 equilibration correction (no thermal noise), produced by ME-ICA %s" % __version__, disable=options.pp_only) 
    export_result('TED/betas_hik_OC.nii','%s_mefc' % (outprefix), "Denoised ICA coeff. set for ME-ICR seed-based FC analysis, produced by ME-ICA %s" % __version__, disable=options.pp_only) 
    export_result('TED/betas_OC.nii','%s_mefl' % (outprefix), "Full ICA coeff. set for component assessment, produced by ME-ICA %s" % __version__, disable=options.pp_only) 
    export_result('TED/feats_OC2.nii','%s_mefcz' % (outprefix), "Z-normalized spatial component maps, produced by ME-ICA %s" % __version__, disable=options.pp_only) 
    export_result('TED/comp_table.txt','%s_ctab.txt' % (outprefix), plaintext=True)
    export_result('TED/meica_mix.1D','%s_mmix.1D' % (outprefix), plaintext=True)


    # Write the preproc script and execute it
    ofh = open('_meica_%s.sh' % setname ,'w')
    print "++ Writing script file: _meica_%s.sh" % (setname)
    ofh.write("\n".join(headsl + sl)+"\n")
    ofh.close()
    if not options.script_only: 
        print "++ Executing script file: _meica_%s.sh" % (setname)
        system('bash _meica_%s.sh' % setname)
