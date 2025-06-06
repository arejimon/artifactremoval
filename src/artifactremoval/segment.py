import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from keras import backend as K
import logging
import matplotlib.pyplot as plt

import numpy as np
import SimpleITK as sitk
import shutil

from artifactremoval.imgproc import load_sitk_img, npy_to_sitk
from artifactremoval.registration import resample_image

##############################################

def flair_seg():
    tf.config.list_physical_devices()

    # Load loss function
    def tversky(y_true, y_pred, smooth=100):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

    def tversky_loss(y_true, y_pred):
        return 1 - tversky(y_true,y_pred)

    def tversky_and_bce_loss(y_true, y_pred):
        fl = tfa.losses.SigmoidFocalCrossEntropy()
        o = fl(y_true, y_pred) + tversky_loss(y_true, y_pred) 
        return o

    def dice_coef(y_true, y_pred, smooth=100):  
        y_true = y_true.astype('float32')
        y_pred = y_pred.astype('float32')

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice

    # Load 3D Unet model 
    model_path = r"X:\ArtifactRemovalProject\data\raw\SegmentationModels\flair_seg.h5"
    unet = tf.keras.models.load_model(model_path, custom_objects={"dice_coef":dice_coef, "tversky_and_bce_loss": tversky_and_bce_loss }, compile=False)

    return unet

def t1_seg():
    tf.config.list_physical_devices()

    # Load loss function
    def tversky(y_true, y_pred, smooth=100):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

    def tversky_loss(y_true, y_pred):
        return 1 - tversky(y_true,y_pred)

    def tversky_and_bce_loss(y_true, y_pred):
        fl = tfa.losses.SigmoidFocalCrossEntropy()
        o = fl(y_true, y_pred) + tversky_loss(y_true, y_pred) 
        return o

    def dice_coef(y_true, y_pred, smooth=100):  
        y_true = y_true.astype('float32')
        y_pred = y_pred.astype('float32')

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice

    # Load 3D Unet model 
    model_path = r"X:\ArtifactRemovalProject\data\raw\SegmentationModels\t1post_seg.h5"
    unet = tf.keras.models.load_model(model_path, custom_objects={"dice_coef":dice_coef, "tversky_and_bce_loss": tversky_and_bce_loss }, compile=False)

    return unet


# # -------------------------------------------------


def perform_segmentation(input_path, overwrite=False):
    """
    Perform segmentation for a single study.
    
    This function reads in preprocessed images from the input folder (input_path) and 
    performs segmentation of FLAIR hyperintensities using two U-Net models:
      - One model (flair_seg) predicts a probability map for FLAIR.
      - A second model (t1_seg) predicts a probability map for T1.
    
    The function then thresholds these probability maps to create binary segmentations,
    resizes the segmentations to the original image sizes, and saves several outputs
    (raw probability maps, binary maps, combined segmentation) to a folder named
    "skullstrip_seg" in the parent folder of input_path.
    
    A marker file ("segmentation_complete.txt") is created in the output folder once
    processing is complete. If the marker exists and overwrite is False, segmentation is skipped.
    
    Parameters:
      input_path (Path): Path to the pre-segmentation folder. Expected to contain:
                         - "skullstripped_preprocessed_t1.nii.gz"
                         - "skullstripped_preprocessed_flair.nii.gz"
                         - "unprocessed_t1.nii.gz"
                         - "unprocessed_flair.nii.gz"
      overwrite (bool): If True, re-run segmentation even if outputs already exist.
    
    Returns:
      None
    """
    # Define output folder (in the study folder, create "segmentation")
    segmentation_dir = input_path.parent / "05_segmentation"
    
    # If overwrite is True and the output folder exists, remove it entirely.
    if overwrite and segmentation_dir.exists():
        shutil.rmtree(segmentation_dir)
        logging.info(f"Overwrite True: Removed previous output folder {segmentation_dir}.")
    
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    
    marker_file = segmentation_dir / "segmentation_complete.txt"
    if not overwrite and marker_file.exists():
        logging.info(f"Segmentation already complete in {segmentation_dir}. Skipping.")
        return

    logging.info(f"Starting segmentation for folder: {input_path}")

    # Initialize segmentation models
    try:
        flair_unet = flair_seg()
        t1_unet = t1_seg()
    except Exception as e:
        logging.error(f"Error initializing segmentation models: {e}")
        return

    mask_cutoff = 0.4

    # Define important file paths
    t1input_path    = input_path / "t1_cerebellum_removed.nii.gz"
    flairinput_path = input_path / "flair_cerebellum_removed.nii.gz"

    # Read preprocessed (skullstripped) images
    try:
        t1_img = sitk.ReadImage(str(t1input_path))
        flair_img = sitk.ReadImage(str(flairinput_path))
    except Exception as e:
        logging.error(f"Error reading cerebellum and skull stripped images: {e}")
        return
    
    t1_spacing, t1_size = t1_img.GetSpacing(), t1_img.GetSize()
    flair_spacing, flair_size = flair_img.GetSpacing(), flair_img.GetSize()

    logging.info("Resampling images to [128, 128, 128] for segmentation input...")
    try:
        t1_img_res = resample_image(t1_img, mask=False, size=[128, 128, 128])
        flair_img_res = resample_image(flair_img, mask=False, size=[128, 128, 128])
    except Exception as e:
        logging.error(f"Error during resampling: {e}")
        return
    
    # Save preprocessed images
    try:
        filename = segmentation_dir / f"t1_seginput_128cubed.nii.gz"
        sitk.WriteImage(t1_img_res, str(filename))
        
        filename = segmentation_dir / f"flair_seginput_128cubed.nii.gz"        
        sitk.WriteImage(flair_img_res, str(filename))
    except Exception as e:
        logging.error(f"Error saving resampled images: {e}")
        return

    try:
        # Format images as numpy arrays for model input
        img1 = np.expand_dims(load_sitk_img(t1_img_res), axis=0)
        img2 = np.expand_dims(load_sitk_img(flair_img_res), axis=0)
        imgInput1 = np.expand_dims(np.concatenate([img1, img2], axis=0), axis=0)
        imgInput2 = np.expand_dims(np.concatenate([img1, img2], axis=0), axis=0)
    except Exception as e:
        logging.error(f"Error preparing images for segmentation: {e}")
        return

    # Run predictions
    try:
        preds_flair = flair_unet.predict(imgInput2)
        preds_t1 = t1_unet.predict(imgInput1)
        preds_flair_npy = preds_flair[0, 1, :, :, :]
        preds_t1_npy = preds_t1[0, 0, :, :, :]
    except Exception as e:
        logging.error(f"Error during segmentation model prediction: {e}")
        return

    # Save raw probability maps (before resizing)
    try:
        prob_seg_t1_resampled = npy_to_sitk(np.transpose(preds_t1_npy, (2, 0, 1)), t1_img_res)
        filename = segmentation_dir / f"prob_seg_t1_128cubed.nii.gz"
        sitk.WriteImage(prob_seg_t1_resampled, str(filename))

        prob_seg_flair_resampled = npy_to_sitk(np.transpose(preds_flair_npy, (2, 0, 1)), flair_img_res)
        filename = segmentation_dir / f"prob_seg_flair_128cubed.nii.gz"
        sitk.WriteImage(prob_seg_flair_resampled, str(filename))

    except Exception as e:
        logging.error(f"Error saving raw probability maps: {e}")
        return

    # Threshold to get binary segmentations
    binary_preds_t1_npy = (preds_t1_npy > mask_cutoff).astype(int)
    binary_preds_flair_npy = (preds_flair_npy > mask_cutoff).astype(int)

    # Create and save combined labels before resizing
    try:
        binary_preds_t1 = npy_to_sitk(np.transpose(binary_preds_t1_npy, (2, 0, 1)), t1_img_res)
        filename = segmentation_dir / f"binary_seg_t1_128cubed.nii.gz"
        sitk.WriteImage(binary_preds_t1, str(filename))

        binary_preds_flair = npy_to_sitk(np.transpose(binary_preds_flair_npy, (2, 0, 1)), t1_img_res)
        filename = segmentation_dir / f"binary_seg_flair_128cubed.nii.gz"
        sitk.WriteImage(binary_preds_flair, str(filename))

    except Exception as e:
        logging.error(f"Error saving binary segmentations before resizing: {e}")
        return

    # Resize segmentations to skull stripped image sizes
    try:
        resized_preds_t1 = resample_image(npy_to_sitk(np.transpose(binary_preds_t1_npy, (2, 0, 1)), t1_img_res), mask=True, size=t1_size)
        resized_preds_flair = resample_image(npy_to_sitk(np.transpose(binary_preds_flair_npy, (2, 0, 1)), flair_img_res), mask=True, size=flair_size)
                
        resized_prob_preds_t1 = resample_image(npy_to_sitk(np.transpose(preds_t1_npy, (2, 0, 1)), t1_img_res), mask=False, size=t1_size)
        resized_prob_preds_flair = resample_image(npy_to_sitk(np.transpose(preds_flair_npy, (2, 0, 1)), flair_img_res), mask=False, size=flair_size)
    except Exception as e:
        logging.error(f"Error resizing segmentations: {e}")
        return

    # Save resized segmentations and combined labels
    try:
        filename = segmentation_dir / f"binary_seg_t1_ssres.nii.gz"
        sitk.WriteImage(resized_preds_t1, str(filename))

        filename = segmentation_dir / f"binary_seg_flair_ssres.nii.gz"
        sitk.WriteImage(resized_preds_flair, str(filename))

        filename = segmentation_dir / f"prob_seg_t1_ssres.nii.gz"
        sitk.WriteImage(resized_prob_preds_t1, str(filename))

        filename = segmentation_dir / f"prob_seg_flair_ssres.nii.gz"
        sitk.WriteImage(resized_prob_preds_flair, str(filename))

    except Exception as e:
        logging.error(f"Error saving resized segmentations: {e}")
        return

    # Calculate and log segmentation volumes (in cc)
    try:
        voxel_vol = t1_spacing[0] * t1_spacing[1] * t1_spacing[2]
        t1_vol = np.sum(binary_preds_t1_npy) * voxel_vol * 0.001
        flair_vol = np.sum(binary_preds_flair_npy) * voxel_vol * 0.001
        logging.info(f"Segmentation volumes - T1: {t1_vol} cc, FLAIR: {flair_vol} cc")
    except Exception as e:
        logging.error(f"Error calculating segmentation volumes: {e}")

    # Create marker file to indicate completion
    marker_file.touch()
    logging.info(f"Segmentation complete. Marker created at {marker_file}")


def process_study_segmentation(subject, study, interim_directory, overwrite=False):
    """
    Process segmentation for a single study.
    
    Constructs the pre-segmentation folder path (assumed to be "04_cerebellumstrip" in the study folder)
    and then calls perform_segmentation().
    
    Parameters:
      subject: Subject object with attributes (e.g., id, subject_path).
      study: Study object with attribute date.
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, re-run segmentation even if a marker exists.
    """
    study_folder = study.date.replace("/", ".")
    skull_strip_dir = interim_directory / subject.id / study_folder / "04_cerebellumstrip"
    if not skull_strip_dir.exists():
        logging.error(f"Cerebellum Strip directory does not exist: {skull_strip_dir}")
        return
    logging.info(f"Processing segmentation for subject {subject.id}, study {study.date}")

    perform_segmentation(skull_strip_dir, overwrite=overwrite)


def batch_process_segmentation(project, interim_directory, overwrite=False):
    """
    Batch process segmentation for all subjects and studies.
    
    Iterates through each subject and study in the project, and calls process_study_segmentation.
    Uses a marker file to avoid reprocessing studies unless overwrite is specified.
    
    Parameters:
      project: Project object with a method all_subject() returning a list of subjects.
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, re-run segmentation even if a marker exists.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for segmentation.")
    for subject in all_subjects:
        logging.info(f"Processing segmentation for subject: {subject.subject_path}")
        studies = subject.all_study()
        for study in studies:
            try:
                process_study_segmentation(subject, study, interim_directory, overwrite=overwrite)
            except Exception as e:
                logging.error(f"Error processing segmentation for subject {subject.id}, study {study.date}: {e}")


def inspect_segmentation_for_study(subject, study, interim_directory, overwrite=False, save_fig=False):
    """
    Visually inspect segmentation results for flair hyperintensity for a given study.
    
    Finds the three axial slices with the largest segmentation volume from the binary flair segmentation,
    and displays them in a montage with two rows:
      - Top row: flair skull-stripped image with segmentation overlay.
      - Bottom row: flair skull-stripped image alone.
    
    A marker file ("segmentation_inspection_complete.txt") is created in the segmentation folder after a successful
    inspection to prevent re-inspection unless overwrite is True.
    
    Parameters:
      subject: A subject object with attribute 'id'.
      study: A study object with attribute 'date'.
      interim_directory: Base directory (Path) where subject folders reside.
      overwrite (bool): If True, re-run inspection even if marker exists.
      save_fig (bool): If True, save the figure as a PNG instead of displaying interactively.
    """
    # Construct the folder paths (assuming segmentation outputs are in "skullstrip_seg")
    study_folder = study.date.replace("/", ".")
    seg_dir = interim_directory / subject.id / study_folder / "05_segmentation"

    seg_dir.mkdir(parents=True, exist_ok=True)
    
    # Marker file for inspection completion
    marker_file = seg_dir / "segmentation_inspection_complete.txt"
    if not overwrite and marker_file.exists():
        logging.info(f"Skipping segmentation inspection for subject {subject.id}, study {study.date} (marker exists).")
        return
    
    # Define file paths (adjust filenames if needed)
    flair_seg_file = seg_dir / "binary_seg_flair_128cubed.nii.gz"
    flair_img_file = seg_dir / "flair_seginput_128cubed.nii.gz"
    
    try:
        flair_seg_img = sitk.ReadImage(str(flair_seg_file))
        flair_img = sitk.ReadImage(str(flair_img_file))
        logging.info(f"Loaded segmentation and flair image for subject {subject.id}, study {study.date}")
    except Exception as e:
        logging.error(f"Error loading images for subject {subject.id}, study {study.date}: {e}")
        return
    
    # Convert images to numpy arrays.
    seg_arr = sitk.GetArrayFromImage(flair_seg_img)  # shape: [slices, height, width]
    flair_arr = sitk.GetArrayFromImage(flair_img)
    
    # Compute segmentation volume per slice (sum of segmentation mask voxels)
    slice_volumes = np.array([np.sum(seg_arr[i]) for i in range(seg_arr.shape[0])])
    
    # Find indices of the three slices with the largest segmentation volume.
    # If there are fewer than three slices with nonzero segmentation, use available slices.
    if np.count_nonzero(slice_volumes) == 0:
        logging.warning(f"No segmentation found for subject {subject.id}, study {study.date}.")
        return
    top_indices = np.argsort(-slice_volumes)  # descending order
    num_slices = min(3, len(top_indices))
    selected_slices = top_indices[:num_slices]
    selected_slices.sort()  # sort ascending so they appear in order
    
    # Create a montage: 2 rows x num_slices columns.
    fig, axs = plt.subplots(2, num_slices, figsize=(5 * num_slices, 8))
    
    # If only one column, make axs 2D array for consistency.
    if num_slices == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    
    for idx, slice_idx in enumerate(selected_slices):
        # Top row: flair image with segmentation overlay.
        axs[0, idx].imshow(flair_arr[slice_idx, :, :], cmap="gray")
        # Overlay segmentation in red where segmentation mask is nonzero.
        axs[0, idx].imshow(seg_arr[slice_idx, :, :], cmap="Reds", alpha=0.5)
        axs[0, idx].set_title(f"Slice {slice_idx})")
        axs[0, idx].axis("off")
        
        # Bottom row: flair skull-stripped image (without overlay).
        axs[1, idx].imshow(flair_arr[slice_idx, :, :], cmap="gray")
        axs[1, idx].set_title(f"Flair (Slice {slice_idx})")
        axs[1, idx].axis("off")
    
    plt.suptitle(f"Segmentation QC - Subject: {subject.id} | Study: {study.date}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_fig:
        fig_path = seg_dir / "segmentation_inspection.png"
        plt.savefig(str(fig_path))
        logging.info(f"Saved segmentation inspection figure to {fig_path}")
    else:
        plt.show()
    plt.close(fig)
    
    # Create marker file upon successful inspection.
    marker_file.touch()
    logging.info(f"Segmentation inspection complete for subject {subject.id}, study {study.date}. Marker created at {marker_file}")

    
def batch_inspect_segmentation(project, interim_directory, overwrite=False, save_fig=False):
    """
    Batch process segmentation visual inspection for all subjects and studies.
    
    Iterates through each subject and each study, calling inspect_segmentation_for_study.
    A marker file in the study's segmentation folder prevents re-inspection unless overwrite=True.
    
    Parameters:
      project: A project object with method all_subject() returning a list of subjects.
      interim_directory: Base directory (Path) where subject folders reside.
      overwrite (bool): If True, re-run inspection even if marker file exists.
      save_fig (bool): If True, save the inspection figures; if False, display interactively.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for segmentation inspection.")
    
    for subject in all_subjects:
        studies = subject.all_study()
        for study in studies:
            try:
                inspect_segmentation_for_study(subject, study, interim_directory, overwrite=overwrite, save_fig=save_fig)
            except Exception as e:
                logging.error(f"Error during segmentation inspection for subject {subject.id}, study {study.date}: {e}")
