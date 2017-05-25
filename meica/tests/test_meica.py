from meica import fparse

def test_fparse():
    ('sub.001', '.nii.gz') == fparse('sub.001.nii.gz')
    ('sub.001', '+tlrc') == fparse('sub.001+tlrc.BRIK.gz')
