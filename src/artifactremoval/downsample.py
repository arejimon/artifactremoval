import logging
import shutil
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from artifactremoval.registration import resample_image
from artifactremoval.imgproc import npy_to_sitk

# =============================================================================
# SINGLE STUDY PROCESSING: Downsampling and Multiplication
# =============================================================================

def perform_downsampling(study_folder, overwrite=False):
    """
    Downsamples the following files for a single study to the siref original size:
      - Flair segmentation (assumed file: "binary_segmentation_flair_after_resize.nii.gz" in "skullstrip_seg")
      - Flair volume (assumed file: "flair_skullstrip.nii.gz" in "03_skullstrip")
      - SIREF (assumed file: "siref_reg.nii.gz" in "02_registered")
      - Brain mask (assumed file: "brainmask_reg.nii.gz" in "02_registered")
      - SIREF Original (file: "unprocessed_siref.nii.gz in "01_unprocessed")
      
    Saves both the original copies (labeled as upsampled) and the downsampled versions in a new folder 
    called "04_downsample" within the study folder.
    
    Then, multiplies the downsampled flair segmentation, siref, and brain mask to produce a final mask.
    A marker file ("downsampling_complete.txt") is created to indicate processing is complete.
    
    Parameters:
      study_folder (Path): Path to the study folder (e.g., interim_directory/subject_id/study_date).
      overwrite (bool): If False, skip processing if marker file exists. If True, delete files and start over. 
    """
    # Define output folder
    downsample_folder = study_folder / "06_downsample"
    
    # If overwrite is True and the output folder exists, remove it entirely.
    if overwrite and downsample_folder.exists():
        shutil.rmtree(downsample_folder)
        logging.info(f"Overwrite True: Removed previous output folder {downsample_folder}.")
    
    downsample_folder.mkdir(parents=True, exist_ok=True)
    
    marker_file = downsample_folder / "downsampling_complete.txt"
    if not overwrite and marker_file.exists():
        logging.info(f"Downsampling already complete in {downsample_folder}. Skipping.")
        return

    # Define input subfolder paths
    registered_folder = study_folder / "02_registration"
    skullstrip_folder = study_folder / "04_cerebellumstrip"
    segmentation_folder = study_folder / "05_segmentation"
    unprocessed_folder = study_folder / "01_unprocessed"
    
    # Define input file paths
    flair_seg_file      = segmentation_folder / "binary_seg_flair_ssres.nii.gz"
    flair_vol_file      = skullstrip_folder / "flair_cerebellum_removed.nii.gz"
    siref_file          = registered_folder / "siref_reg.nii.gz"
    brain_mask_file     = registered_folder / "brainmask_reg.nii.gz"
    siref_original_file = unprocessed_folder / "unprocessed_siref.nii.gz"
    
    try:
        flair_seg_img   = sitk.ReadImage(str(flair_seg_file))
        flair_vol_img   = sitk.ReadImage(str(flair_vol_file))
        siref_img       = sitk.ReadImage(str(siref_file))
        brain_mask_img  = sitk.ReadImage(str(brain_mask_file))
        siref_og_img    = sitk.ReadImage(str(siref_original_file))
        logging.info(f"Loaded input images from {study_folder}")
    except Exception as e:
        logging.error(f"Error loading input images in {study_folder}: {e}")
        return
    
    # Save original (upsampled) copies in the downsample folder
    try:
        shutil.copy(str(flair_seg_file),  str(downsample_folder / "flair_seg_upsampled.nii.gz"))
        shutil.copy(str(flair_vol_file),  str(downsample_folder / "flair_vol_upsampled.nii.gz"))
        shutil.copy(str(siref_file),      str(downsample_folder / "siref_upsampled.nii.gz"))
        shutil.copy(str(brain_mask_file), str(downsample_folder / "brain_mask_upsampled.nii.gz"))
        logging.info("Copied original files as upsampled versions.")
    except Exception as e:
        logging.error(f"Error copying upsampled files: {e}")
    
    # Downsample the images to the target size (using your resample_image helper)
    target_size = siref_og_img.GetSize()
    try:
        flair_seg_ds = resample_image(flair_seg_img, mask=False, size=target_size)
        flair_vol_ds = resample_image(flair_vol_img, mask=False, size=target_size)
        siref_ds     = resample_image(siref_img, mask=False, size=target_size)
        brain_mask_ds= resample_image(brain_mask_img, mask=False, size=target_size)
        logging.info(f"Downsampled images to {target_size}.")
    except Exception as e:
        logging.error(f"Error during downsampling: {e}")
        return
    
    # Save the downsampled images
    try:
        sitk.WriteImage(flair_seg_ds, str(downsample_folder / "flair_seg_downsampled.nii.gz"))
        sitk.WriteImage(flair_vol_ds, str(downsample_folder / "flair_vol_downsampled.nii.gz"))
        sitk.WriteImage(siref_ds,     str(downsample_folder / "siref_downsampled.nii.gz"))
        sitk.WriteImage(brain_mask_ds,str(downsample_folder / "brain_mask_downsampled.nii.gz"))
        logging.info("Saved downsampled files.")
    except Exception as e:
        logging.error(f"Error saving downsampled files: {e}")
        return

    # Multiply the downsampled flair segmentation, siref, and brain mask to obtain the final mask.
    try:
        flair_seg_arr  = sitk.GetArrayFromImage(flair_seg_ds)
        siref_arr      = sitk.GetArrayFromImage(siref_ds)
        brain_mask_arr = sitk.GetArrayFromImage(brain_mask_ds)
        final_mask_arr = flair_seg_arr * siref_arr * brain_mask_arr
        final_mask_img = npy_to_sitk(final_mask_arr, flair_seg_ds)
        sitk.WriteImage(final_mask_img, str(downsample_folder / "final_mask.nii.gz"))
        logging.info("Computed and saved final mask from multiplied downsampled images.")
    except Exception as e:
        logging.error(f"Error computing final mask: {e}")
        return
    
    marker_file.touch()
    logging.info(f"Downsampling complete for {study_folder}. Marker created at {marker_file}")

# =============================================================================
# PER-STUDY PROCESSING
# =============================================================================

def process_study_downsampling(subject, study, interim_directory, overwrite=False):
    """
    Processes downsampling for one study of a subject.
    
    Constructs the study folder path from the subject ID and study date, and calls perform_downsampling.
    
    Parameters:
      subject: Subject object (with attribute id).
      study: Study object (with attribute date).
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, re-run processing even if marker exists.
    """
    study_folder = interim_directory / subject.id / study.date.replace("/", ".")
    logging.info(f"Processing downsampling for subject {subject.id}, study {study.date} in {study_folder}")
    perform_downsampling(study_folder, overwrite=overwrite)

# =============================================================================
# BATCH PROCESSING
# =============================================================================

def batch_process_downsampling(project, interim_directory, overwrite=False):
    """
    Batch processes downsampling for all subjects and studies.
    
    Parameters:
      project: Project object with method all_subject() returning a list of subjects.
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, re-run processing even if marker files exist.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for downsampling.")
    for subject in all_subjects:
        logging.info(f"Processing subject: {subject.id}")
        studies = subject.all_study()
        for study in studies:
            try:
                process_study_downsampling(subject, study, interim_directory, overwrite=overwrite)
            except Exception as e:
                logging.error(f"Error processing downsampling for subject {subject.id}, study {study.date}: {e}")

# =============================================================================
# BATCH VISUALIZATION
# =============================================================================

def inspect_downsampling_for_study(subject, study, interim_directory, save_fig=False, overwrite=False):
    """
    Visually inspects downsampling results for one study.
    
    Loads the downsampled files from the "04_downsample" folder and creates a montage:
      - Top panel: Downsampled flair volume with the downsampled flair segmentation overlaid.
      - Bottom panel: Downsampled siref image with the downsampled brain mask overlaid.
    Uses the middle slice for visualization.
    
    A marker file ("downsampling_inspection_complete.txt") is created in the downsample folder.
    
    Parameters:
      subject: Subject object with attribute id.
      study: Study object with attribute date.
      interim_directory (Path): Base directory where subject folders reside.
      save_fig (bool): If True, save the inspection figure instead of displaying it.
      overwrite (bool): If True, re-run inspection even if marker exists.
    """
    study_folder = interim_directory / subject.id / study.date.replace("/", ".")
    downsample_folder = study_folder / "06_downsample"
    downsample_folder.mkdir(parents=True, exist_ok=True)
    
    marker_file = downsample_folder / "downsampling_inspection_complete.txt"
    if not overwrite and marker_file.exists():
        logging.info(f"Skipping downsampling inspection for subject {subject.id}, study {study.date} (marker exists).")
        return

    try:
        flair_vol_ds  = sitk.ReadImage(str(downsample_folder / "flair_vol_downsampled.nii.gz"))
        flair_seg_ds  = sitk.ReadImage(str(downsample_folder / "flair_seg_downsampled.nii.gz"))
        siref_ds      = sitk.ReadImage(str(downsample_folder / "siref_downsampled.nii.gz"))
        brain_mask_ds = sitk.ReadImage(str(downsample_folder / "brain_mask_downsampled.nii.gz"))
        final_mask_ds = sitk.ReadImage(str(downsample_folder / "final_mask.nii.gz"))    
        logging.info(f"Loaded downsampled images for inspection for subject {subject.id}, study {study.date}")
    except Exception as e:
        logging.error(f"Error loading downsampled images for subject {subject.id}, study {study.date}: {e}")
        return

    # Convert images to numpy arrays
    flair_vol_arr  = sitk.GetArrayFromImage(flair_vol_ds)
    flair_seg_arr  = sitk.GetArrayFromImage(flair_seg_ds)
    siref_arr      = sitk.GetArrayFromImage(siref_ds)
    brain_mask_arr = sitk.GetArrayFromImage(brain_mask_ds)
    final_mask_arr = sitk.GetArrayFromImage(final_mask_ds)
    
    # Use the middle slice
    slice_idx = flair_vol_arr.shape[0] // 2

    # Create a figure with 4 columns
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Column 1: Flair volume with flair segmentation overlay
    axs[0].imshow(flair_vol_arr[slice_idx, :, :], cmap="gray")
    axs[0].imshow(np.ma.masked_where(flair_seg_arr[slice_idx, :, :] == 0, flair_seg_arr[slice_idx, :, :]),
                  cmap="Reds", alpha=0.5)
    axs[0].set_title("Flair + FlairSeg")
    axs[0].axis("off")
    
    # Column 2: SIREF with brain mask overlay
    axs[1].imshow(siref_arr[slice_idx, :, :], cmap="gray")
    axs[1].imshow(np.ma.masked_where(brain_mask_arr[slice_idx, :, :] == 0, brain_mask_arr[slice_idx, :, :]),
                  cmap="Blues", alpha=0.5)
    axs[1].set_title("Siref + BrainMask")
    axs[1].axis("off")
    
    # Column 3: SIREF with final mask overlay
    axs[2].imshow(siref_arr[slice_idx, :, :], cmap="gray")
    axs[2].imshow(np.ma.masked_where(final_mask_arr[slice_idx, :, :] == 0, final_mask_arr[slice_idx, :, :]),
                  cmap="Greens", alpha=0.5)
    axs[2].set_title("Siref + FinalMask")
    axs[2].axis("off")
    
    # Column 4: Flair volume with final mask overlay
    axs[3].imshow(flair_vol_arr[slice_idx, :, :], cmap="gray")
    axs[3].imshow(np.ma.masked_where(final_mask_arr[slice_idx, :, :] == 0, final_mask_arr[slice_idx, :, :]),
                  cmap="Greens", alpha=0.5)
    axs[3].set_title("Flair + FinalMask")
    axs[3].axis("off")
    
    plt.suptitle(f"Downsampling Inspection - Subject: {subject.id} | Study: {study.date} | Slice: {slice_idx}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_fig:
        fig_path = downsample_folder / "downsampling_inspection.png"
        plt.savefig(str(fig_path))
        logging.info(f"Saved downsampling inspection figure to {fig_path}")
    else:
        plt.show()
    plt.close(fig)
    
    marker_file.touch()
    logging.info(f"Downsampling inspection complete for subject {subject.id}, study {study.date}. Marker created at {marker_file}")

def batch_inspect_downsampling(project, interim_directory, overwrite=False, save_fig=False):
    """
    Batch visual inspection for downsampling results.
    
    Iterates through each subject and study and calls inspect_downsampling_for_study.
    
    Parameters:
      project: Project object with method all_subject() returning list of subjects.
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, re-run inspection even if marker exists.
      save_fig (bool): If True, save inspection figures; otherwise, display interactively.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for downsampling inspection.")
    for subject in all_subjects:
        studies = subject.all_study()
        for study in studies:
            try:
                inspect_downsampling_for_study(subject, study, interim_directory, save_fig=save_fig, overwrite=overwrite)
            except Exception as e:
                logging.error(f"Error during downsampling inspection for subject {subject.id}, study {study.date}: {e}")