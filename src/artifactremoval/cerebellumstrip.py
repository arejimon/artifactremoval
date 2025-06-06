import logging
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from artifactremoval.imgproc import sitk_to_itk, itk_to_sitk
from artifactremoval.midas import align_map
from artifactremoval.registration import rigid_registration_best_transform


#--- Assume a rigid registration function is defined elsewhere ---
# def rigid_registration(fixed, moving): ...
# For example purposes, we assume it's imported:
# from registration_module import rigid_registration

#--------------------------------------------------------------------------
# Single-Input Function: Perform Cerebellum Stripping
#--------------------------------------------------------------------------

def perform_cerebellum_stripping(input_path, overwrite=False):
    """
    For a single study, remove the cerebellum from the skull-stripped images.
    
    This function loads the skull-stripped T1 (and FLAIR, if available) from input_path
    (which should be the "03_skullstripped" folder), registers a brain atlas and a cerebellum atlas
    to the skull-stripped T1, resamples the cerebellum atlas (using the same transform), and applies
    the cerebellum mask to remove cerebellar voxels. The resulting images (for T1 and FLAIR) and the
    resampled cerebellum mask are saved in a new folder ("03_cerebellum_removed") within the study folder.
    
    A marker file ("cerebellum_stripping_complete.txt") is created in the output folder.
    
    Parameters:
      input_path (Path): Path to the "03_skullstripped" folder.
      overwrite (bool): If True, deletes previous outputs before processing.
    """
    #--- Set your atlas paths (adjust as needed) ---
    BRAIN_ATLAS_PATH = Path(r"X:\ArtifactRemovalProject\data\raw\Atlases\lpba40\brain.nii")
    CEREBELLUM_ATLAS_PATH = Path(r"X:\ArtifactRemovalProject\data\raw\Atlases\lpba40\cerebellum.nii")

    # Define output folder within the study folder
    study_folder = input_path.parent  # assuming input_path is "03_skullstripped"
    output_folder = study_folder / "04_cerebellumstrip"
    if overwrite and output_folder.exists():
        shutil.rmtree(output_folder)
        logging.info(f"Overwrite enabled: Removed previous output folder {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    marker_file = output_folder / "cerebellum_stripping_complete.txt"
    if not overwrite and marker_file.exists():
        logging.info(f"Cerebellum stripping already complete in {output_folder}. Skipping.")
        return

    # Load skull-stripped T1 (required) and FLAIR (if available)
    t1_path = input_path / "t1_skullstrip.nii.gz"
    try:
        t1_img = sitk.ReadImage(str(t1_path))
        logging.info(f"Loaded skull-stripped T1 from {t1_path}")
        logging.info(f"Type of t1_img: {type(t1_img)}; Size: {t1_img.GetSize()}")
    except Exception as e:
        logging.error(f"Error loading skull-stripped T1 from {t1_path}: {e}")
        return

    flair_path = input_path / "flair_skullstrip.nii.gz"
    try:
        flair_img = sitk.ReadImage(str(flair_path))
        logging.info(f"Loaded skull-stripped FLAIR from {flair_path}")
        logging.info(f"Type of flair_img: {type(flair_img)}; Size: {flair_img.GetSize()}")        
    except Exception as e:
        logging.warning(f"FLAIR not available or error loading: {e}. Proceeding with T1 only.")
        flair_img = None

    # Loading the atlases
    try:
        # Convert the atlas images to ITK images
        brain_atlas_itk = sitk_to_itk(sitk.ReadImage(str(BRAIN_ATLAS_PATH)))
        cerebellum_atlas_itk = sitk_to_itk(sitk.ReadImage(str(CEREBELLUM_ATLAS_PATH)))
        # Convert the fixed image (t1_img) to ITK image as well.
        t1_img_itk = sitk_to_itk(t1_img)
        
        # Now call align_map with both images as ITK images.
        brain_atlas_aligned_itk = align_map(brain_atlas_itk, t1_img_itk)
        cerebellum_atlas_aligned_itk = align_map(cerebellum_atlas_itk, t1_img_itk)
        
        # Convert the aligned atlases back to SimpleITK images.
        brain_atlas = itk_to_sitk(brain_atlas_aligned_itk)
        cerebellum_atlas = itk_to_sitk(cerebellum_atlas_aligned_itk)
        
        logging.info("Loaded brain and cerebellum atlases.")
    except Exception as e:
        logging.error(f"Error loading atlases: {e}")
        return

    # Register the brain atlas to the skull-stripped T1.
    try:
        transform = rigid_registration_best_transform(sitk.Cast(t1_img, sitk.sitkFloat32), brain_atlas)
        logging.info("Computed registration transform.")
    except Exception as e:
        logging.error(f"Error during registration: {e}")
        return

    # Resample both atlases to T1 space using the computed transform.
    try:
        atlas_resampled = sitk.Resample(brain_atlas, t1_img, transform, sitk.sitkLinear, 0.0, brain_atlas.GetPixelID())
        cerebellum_resampled = sitk.Resample(cerebellum_atlas, t1_img, transform, sitk.sitkLinear, 0.0, cerebellum_atlas.GetPixelID())
        logging.info("Resampled atlases to T1 space.")
    except Exception as e:
        logging.error(f"Error during resampling: {e}")
        return

    # Save the resampled cerebellum mask for inspection (convert to UInt8 and threshold)
    cerebellum_mask = sitk.Cast(cerebellum_resampled, sitk.sitkUInt8)==0
    # We assume that non-zero values indicate cerebellum; we create a binary mask.
    try:
        cerebellum_mask_path = output_folder / "cerebellum_mask.nii.gz"
        sitk.WriteImage(cerebellum_mask, str(cerebellum_mask_path))
        logging.info(f"Saved resampled cerebellum mask to {cerebellum_mask_path}")
    except Exception as e:
        logging.error(f"Error saving cerebellum mask: {e}")

    # Apply the cerebellum mask to remove cerebellar voxels.
    try:
        # We want to keep voxels where cerebellum_mask is 0.
        t1_cereb_removed = sitk.Mask(t1_img, cerebellum_mask)
        if flair_img is not None:
            flair_cereb_removed = sitk.Mask(flair_img, cerebellum_mask)
        else:
            flair_cereb_removed = None
        logging.info("Applied cerebellum mask to T1 (and FLAIR if available).")
    except Exception as e:
        logging.error(f"Error applying cerebellum mask: {e}")
        return

    # Save the cerebellum-stripped images.
    try:
        t1_out_path = output_folder / "t1_cerebellum_removed.nii.gz"
        sitk.WriteImage(t1_cereb_removed, str(t1_out_path))
        logging.info(f"Saved cerebellum-stripped T1 to {t1_out_path}")
        if flair_cereb_removed is not None:
            flair_out_path = output_folder / "flair_cerebellum_removed.nii.gz"
            sitk.WriteImage(flair_cereb_removed, str(flair_out_path))
            logging.info(f"Saved cerebellum-stripped FLAIR to {flair_out_path}")
    except Exception as e:
        logging.error(f"Error saving cerebellum-stripped images: {e}")
        return

    marker_file.touch()
    logging.info(f"Cerebellum stripping complete for study at {input_path}. Marker created at {marker_file}")

#--------------------------------------------------------------------------
# Per-Study Processing Function
#--------------------------------------------------------------------------

def process_study_cerebellum_stripping(subject, study, interim_directory, overwrite=False):
    """
    Processes cerebellum stripping for a single study.
    
    Constructs the study folder path using subject.id and study.date, then calls
    perform_cerebellum_stripping() on the "03_skullstripped" folder.
    
    Parameters:
      subject: Subject object with attribute 'id'.
      study: Study object with attribute 'date'.
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, reprocess even if marker file exists.
    """
    study_folder = interim_directory / subject.id / study.date.replace("/", ".")
    skullstrip_folder = study_folder / "03_skullstrip"
    if not skullstrip_folder.exists():
        logging.error(f"Skullstripped folder does not exist for subject {subject.id}, study {study.date}")
        return
    logging.info(f"Processing cerebellum stripping for subject {subject.id}, study {study.date}")
    perform_cerebellum_stripping(skullstrip_folder, overwrite=overwrite)

#--------------------------------------------------------------------------
# Batch Processing Function
#--------------------------------------------------------------------------

def batch_process_cerebellum_stripping(project, interim_directory, overwrite=False):
    """
    Batch processes cerebellum stripping for all subjects and studies in the project.
    
    Iterates over each subject and study, calling process_study_cerebellum_stripping().
    
    Parameters:
      project: Project object with method all_subject() returning a list of subjects.
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, reprocess even if marker files exist.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for cerebellum stripping.")
    for subject in all_subjects:
        logging.info(f"Processing subject: {subject.id}")
        studies = subject.all_study()
        for study in studies:
            try:
                process_study_cerebellum_stripping(subject, study, interim_directory, overwrite=overwrite)
            except Exception as e:
                logging.error(f"Error processing cerebellum stripping for subject {subject.id}, study {study.date}: {e}")

#--------------------------------------------------------------------------
# Visual Inspection Function: Cerebellum Stripping
#--------------------------------------------------------------------------

def inspect_cerebellum_stripping_for_study(subject, study, interim_directory, save_fig=False, overwrite=False):
    """
    Visually inspects cerebellum stripping for a single study by identifying the four axial slices
    with the highest cerebellum mask volume from the resampled cerebellum mask.
    
    Loads the cerebellum mask (saved as "cerebellum_mask_resampled.nii.gz") from the
    "03_cerebellum_removed" folder and the cerebellum-stripped T1 image ("t1_cerebellum_removed.nii.gz").
    It then computes the sum (volume) per axial slice, selects the four slices with highest volume,
    and creates a montage (e.g., 2x2 grid) showing the T1 image slice with the cerebellum mask overlay.
    
    If save_fig is True, the figure is saved with a timestamped filename to avoid overwriting.
    A marker file ("cerebellum_stripping_inspection_complete.txt") is created in the output folder.
    
    Parameters:
      subject: Subject object with attribute 'id'.
      study: Study object with attribute 'date'.
      interim_directory (Path): Base directory where subject folders reside.
      save_fig (bool): If True, saves the figure as a PNG; otherwise, displays interactively.
      overwrite (bool): If True, forces re-inspection even if a marker file exists.
    """
    study_folder = interim_directory / subject.id / study.date.replace("/", ".")
    output_folder = study_folder / "04_cerebellumstrip"
    marker_file = output_folder / "cerebellum_stripping_inspection_complete.txt"
    if not overwrite and marker_file.exists():
        logging.info(f"Skipping cerebellum stripping inspection for subject {subject.id}, study {study.date} (marker exists).")
        return

    try:
        # Load the cerebellum mask and the cerebellum-stripped T1 image
        cereb_mask = sitk.ReadImage(str(output_folder / "cerebellum_mask.nii.gz"))
        t1_cereb = sitk.ReadImage(str(output_folder / "t1_cerebellum_removed.nii.gz"))
        cereb_arr = sitk.GetArrayFromImage(cereb_mask)  # shape: [slices, height, width]
        t1_arr = sitk.GetArrayFromImage(t1_cereb)
        logging.info(f"Loaded images for cerebellum inspection for subject {subject.id}, study {study.date}")
        logging.info(f"T1 image shape: {t1_arr.shape}")
        logging.info(f"Cerebellum mask shape: {cereb_arr.shape}")
    except Exception as e:
        logging.error(f"Error loading images for cerebellum inspection for subject {subject.id}, study {study.date}: {e}")
        return

    # Instead of summing mask values, count the number of zeros in each slice.
    # This gives the number of voxels not in the cerebellum per slice.
    zero_counts = np.count_nonzero(cereb_arr == 0, axis=(1, 2))
    
    if np.count_nonzero(zero_counts) == 0:
        logging.warning(f"No zeros found in cerebellum mask for subject {subject.id}, study {study.date}")
        return

    # Find indices of the four slices with the highest zero counts.
    top_indices = np.argsort(-zero_counts)[:4]
    top_indices.sort()  # sort in ascending order for display

    # Create a 2x2 montage for the selected slices
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for idx, slice_idx in enumerate(top_indices):
        axs[idx].imshow(t1_arr[slice_idx, :, :], cmap='gray')
        axs[idx].set_title(f"Slice {slice_idx} (Vol: {zero_counts[slice_idx]:.0f})")
        axs[idx].axis('off')
    plt.suptitle(f"Cerebellum Stripping Inspection - Subject: {subject.id} | Study: {study.date}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_fig:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = output_folder / f"cerebellum_inspection_{timestamp}.png"
        plt.savefig(str(fig_path), bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved cerebellum inspection figure to {fig_path}")
    else:
        plt.show()
    plt.close(fig)
    
    marker_file.touch()
    logging.info(f"Cerebellum inspection complete for subject {subject.id}, study {study.date}. Marker created at {marker_file}")

#--------------------------------------------------------------------------
# Batch Processing Function for Visual Inspection
#--------------------------------------------------------------------------

def batch_inspect_cerebellum_stripping(project, interim_directory, save_fig=False, overwrite=False):
    """
    Batch visual inspection for cerebellum stripping for all subjects and studies.
    
    Iterates over each subject and study and calls inspect_cerebellum_stripping_for_study.
    
    Parameters:
      project: Project object with method all_subject() returning a list of subjects.
      interim_directory (Path): Base directory where subject folders reside.
      save_fig (bool): If True, saves each inspection figure; otherwise, displays interactively.
      overwrite (bool): If True, forces re-inspection even if marker files exist.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for cerebellum inspection.")
    for subject in all_subjects:
        for study in subject.all_study():
            try:
                inspect_cerebellum_stripping_for_study(subject, study, interim_directory, save_fig=save_fig, overwrite=overwrite)
            except Exception as e:
                logging.error(f"Error during cerebellum inspection for subject {subject.id}, study {study.date}: {e}")