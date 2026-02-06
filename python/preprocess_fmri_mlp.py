
"""
This script preprocesses the CAT-D fMRI dataset using a Nipype pipeline.

The preprocessing pipeline performs the following steps:
1.  Anatomical Preprocessing:
    -   Skull-stripping: Removes the skull and non-brain tissue from the T1-weighted anatomical images using FSL's BET.
2.  Functional Preprocessing:
    -   Motion Correction: Corrects for head motion using FSL's MCFLIRT.
    -   Slice-Timing Correction: Corrects for differences in acquisition time between slices using FSL's SliceTimer.
    -   Coregistration: Aligns the functional and anatomical images using FSL's FLIRT.
    -   Normalization: Warps the data into a standard (MNI) brain space using FSL's FLIRT and FNIRT.
    -   Spatial Smoothing: Applies a Gaussian filter to improve the signal-to-noise ratio using FSL's SUSAN.
"""

import os
from nipype.interfaces import fsl
from nipype.pipeline import engine as pe
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface
# --- --- --- --- --- --- --- --- --- ---
# Configuration
# --- --- --- --- --- --- --- --- --- ---

# Base directory of the dataset
DATA_DIR = os.path.abspath('ds004627-download')
# Directory for the output of the preprocessing
OUTPUT_DIR = os.path.abspath('derivatives')
# List of subjects to process. An empty list will process all subjects.
SUBJECTS = []

# --- --- --- --- --- --- --- --- --- ---
# Preprocessing pipeline
# --- --- --- --- --- --- --- --- --- ---

def get_preprocessing_workflow():
    """
    Creates the fMRI preprocessing workflow.
    """
    workflow = pe.Workflow(name='preprocessing')
    workflow.base_dir = OUTPUT_DIR

    # --- --- --- --- --- --- --- --- --- ---
    # 1. Anatomical preprocessing
    # --- --- --- --- --- --- --- --- --- ---

    # Brain extraction
    bet_anat = pe.Node(interface=fsl.BET(), name='bet_anat')
    bet_anat.inputs.frac = 0.5
    bet_anat.inputs.robust = True
    bet_anat.inputs.output_type = 'NIFTI_GZ'

    # --- --- --- --- --- --- --- --- --- ---
    # 2. Functional preprocessing
    # --- --- --- --- --- --- --- --- --- ---

    # Motion correction
    mcflirt = pe.Node(interface=fsl.MCFLIRT(), name='mcflirt')
    mcflirt.inputs.mean_vol = True
    mcflirt.inputs.output_type = 'NIFTI_GZ'

    # Slice timing correction
    slicetimer = pe.Node(interface=fsl.SliceTimer(), name='slicetimer')
    slicetimer.inputs.output_type = 'NIFTI_GZ'
    slicetimer.inputs.time_repetition = 2.0  # Please adjust this to your TR

    # Coregistration (functional to anatomical)
    coreg = pe.Node(interface=fsl.FLIRT(), name='coreg')
    coreg.inputs.dof = 6
    coreg.inputs.output_type = 'NIFTI_GZ'

    # Normalization (anatomical to MNI)
    normalize = pe.Node(interface=fsl.FLIRT(), name='normalize')
    normalize.inputs.reference = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
    normalize.inputs.output_type = 'NIFTI_GZ'

    # Apply normalization warp to functional data
    apply_warp = pe.Node(interface=fsl.ApplyWarp(), name='apply_warp')
    apply_warp.inputs.ref_file = fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
    apply_warp.inputs.output_type = 'NIFTI_GZ'

    # Spatial smoothing
    smoothing = pe.Node(interface=fsl.SUSAN(), name='smoothing')
    smoothing.inputs.fwhm = 6.0
    smoothing.inputs.brightness_threshold = 2000.0

    # --- --- --- --- --- --- --- --- --- ---
    # Connect the nodes
    # --- --- --- --- --- --- --- --- --- ---

    workflow.connect([
        (bet_anat, coreg, [('out_file', 'reference')]),
        (mcflirt, slicetimer, [('out_file', 'in_file')]),
        (slicetimer, coreg, [('slice_time_corrected_file', 'in_file')]),
        (coreg, normalize, [('out_file', 'in_file')]),
        (normalize, apply_warp, [('out_matrix_file', 'premat')]),
        (slicetimer, apply_warp, [('slice_time_corrected_file', 'in_file')]),
        (apply_warp, smoothing, [('out_file', 'in_file')]),
    ])

    return workflow


# --- --- --- --- --- --- --- --- --- ---
# Main function
# --- --- --- --- --- --- --- --- --- ---

if __name__ == '__main__':
    import datalad.api as dl

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Infosource: to iterate over subjects and sessions
    if not SUBJECTS:
        SUBJECTS = [s for s in os.listdir(DATA_DIR) if s.startswith('sub-')]

    # Datalad get: to download the data on the fly
    for subject_id in SUBJECTS:
        anat_file = os.path.join(DATA_DIR, subject_id, 'ses-v1', 'anat', f'{subject_id}_ses-v1_acq-mprage_rec-prenorm_run-1_T1w.nii.gz')
        func_file = os.path.join(DATA_DIR, subject_id, 'ses-v1', 'func', f'{subject_id}_ses-v1_task-rest_bold.nii.gz')
        # if os.path.exists(os.path.dirname(anat_file)):
        #     # dl.get(anat_file)
        # if os.path.exists(os.path.dirname(func_file)):
        #     # dl.get(func_file)

    infosource = pe.Node(
        interface=IdentityInterface(fields=['subject_id']),
        name='infosource'
    )
    infosource.iterables = ('subject_id', SUBJECTS)

    # SelectFiles: to grab the data
    templates = {
        'anat': '{subject_id}/ses-v1/anat/{subject_id}_ses-v1_acq-mprage_rec-prenorm_run-1_T1w.nii.gz',
        'func': '{subject_id}/ses-v1/func/{subject_id}_ses-v1_task-rest_bold.nii.gz'
    }
    selectfiles = pe.Node(
        SelectFiles(templates, base_directory=DATA_DIR),
        name='selectfiles'
    )

    # Datasink: to store the results
    datasink = pe.Node(DataSink(), name='datasink')
    datasink.inputs.base_directory = OUTPUT_DIR

    # Get the preprocessing workflow
    preprocessing_workflow = get_preprocessing_workflow()

    # Create the main workflow
    main_workflow = pe.Workflow(name='main_workflow')
    main_workflow.base_dir = OUTPUT_DIR

    # Connect the nodes
    main_workflow.connect([
        (infosource, selectfiles, [('subject_id', 'subject_id')]),
        (selectfiles, preprocessing_workflow, [
            ('anat', 'bet_anat.in_file'),
            ('func', 'mcflirt.in_file'),
        ]),
        (preprocessing_workflow, datasink, [
            ('smoothing.smoothed_file', 'preprocessed_func'),
            ('normalize.out_file', 'normalized_anat'),
            ('mcflirt.par_file', 'motion_parameters'),
        ]),
    ])

    # Run the workflow
    main_workflow.run('MultiProc', plugin_args={'n_procs': 4})
