from meica.meica import format_inset, fparse

def test_fparse():
    ('sub.001', '.nii.gz') == fparse('sub.001.nii.gz')
    ('sub.001', '+tlrc')   == fparse('sub.001+tlrc.BRIK.gz')


def test_format_inset():
    ('sub_001.e02.nii.gz',
     'sub_001.e0123') == format_inset('sub_001.e0[1,2,3].nii.gz')

    ('sub_001.e02.nii.gz',
     'sub_001.e0123') == format_inset('sub_001.e01.nii.gz, ' +
                                      'sub_001.e02.nii.gz, ' +
                                      'sub_001.e03.nii.gz',
                                      [12.2, 24.6, 30])

    ('sub_001.e02+tlrc.HEAD',
     'sub_001.e0123') == format_inset('sub_001.e0[1,2,3]+tlrc.BRIK.gz',
                                      [12.2, 24.6, 30])

    ('sub_001.e02+tlrc.BRIK.gz',
     'sub_001.e0123') == format_inset('sub_001.e01+tlrc.BRIK.gz, ' +
                                      'sub_001.e02+tlrc.BRIK.gz, ' +
                                      'sub_001.e03+tlrc.BRIK.gz',
                                      [12.2, 24.6, 30])


# def test_find_CM():
#     [0.343872,
#      -9.33318,
#      -24.6354] == find_CM('../resources/sub-001_T1w.nii.gz')
