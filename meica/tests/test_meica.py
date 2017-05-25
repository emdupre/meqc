from meica import format_inputs, fparse

def test_fparse():
    ('sub.001', '.nii.gz') == fparse('sub.001.nii.gz')
    ('sub.001', '+tlrc')   == fparse('sub.001+tlrc.BRIK.gz')

def test_format_inputs():
    'sub_001.e0123' == format_inputs('sub_001.e0[1,2,3].nii.gz', '',
                                     [12.2, 24.6, 30])
