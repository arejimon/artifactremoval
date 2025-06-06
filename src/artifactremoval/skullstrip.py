import itk
import SimpleITK as sitk
import logging
import matplotlib.pyplot as plt

from artifactremoval.imgproc import itk_to_sitk, sitk_to_itk

def skull_strip(image):
    #Initialize ITK image and pixel types
    PixelType = itk.SS
    ImageType = itk.Image[PixelType,3]
    AtlasImageType = itk.Image[itk.SS,3]
    AtlasLabelType = itk.Image[itk.UC,3]

    atlasImageName = r"X:\ArtifactRemovalProject\data\raw\Atlases\atlasImage.mha"
    atlasMaskName = r"X:\ArtifactRemovalProject\data\raw\Atlases\atlasMask.mha"

    atlasReader = itk.ImageFileReader[AtlasImageType].New()
    atlasReader.SetFileName(atlasImageName)
    atlasReader.Update()

    labelReader = itk.ImageFileReader[AtlasLabelType].New()
    labelReader.SetFileName(atlasMaskName)
    labelReader.Update()

    skullStrip = itk.StripTsImageFilter.New()
    skullStrip.SetInput(image)
    skullStrip.SetAtlasImage(atlasReader.GetOutput())
    skullStrip.SetAtlasBrainMask(labelReader.GetOutput())
    skullStrip.Update()
    skullMask = skullStrip.GetOutput()

    maskFilter = itk.MaskImageFilter[ImageType, AtlasLabelType, ImageType].New()
    maskFilter.SetInput1(image)
    maskFilter.SetInput2(skullMask)
    maskFilter.Update()
    maskImage = maskFilter.GetOutput()

    return skullMask, itk_to_sitk(maskImage)

def apply_mask(skullMask, sitkImg):
    itkImg = sitk_to_itk(sitk.Cast(sitkImg, sitk.sitkInt16))

    #Initialize ITK image and pixel types
    PixelType = itk.SS
    ImageType = itk.Image[PixelType,3]
    AtlasImageType = itk.Image[itk.SS,3]
    AtlasLabelType = itk.Image[itk.UC,3]

    maskFilter = itk.MaskImageFilter[ImageType, AtlasLabelType, ImageType].New()
    maskFilter.SetInput1(itkImg)
    maskFilter.SetInput2(skullMask)
    maskFilter.SetCoordinateTolerance(1e-3)
    maskFilter.Update()

    mask_stripped = maskFilter.GetOutput()

    return itk_to_sitk(mask_stripped)

def process_study_skullstrip(subject, study, interim_directory, overwrite=False):
    """
    Skull-strip registered T1 and FLAIR images for one study.
    
    - Loads registered images from the "02_registered" folder.
    - Skull-strips the FLAIR image to create a skull mask.
    - Applies that mask to the T1 image.
    - Saves the skull-stripped images (and optionally the skull mask) in a "03_skullstrip" folder.
    - Uses a marker file to avoid reprocessing unless overwrite=True.
    
    Parameters:
      subject: Subject object with attributes like id and subject_path.
      study: Study object with attribute date.
      interim_directory: Base directory (Path) where subject folders are stored.
      overwrite: If True, reprocess even if a marker file exists.
    """
    # Define folder names based on study date (replace "/" with ".")
    study_folder = study.date.replace("/", ".")
    registered_path = interim_directory / subject.id / study_folder / "02_registration"
    skullstrip_path = interim_directory / subject.id / study_folder / "03_skullstrip"
    skullstrip_path.mkdir(parents=True, exist_ok=True)
    
    # Marker file for skull stripping completion
    marker_file = skullstrip_path / "skullstrip_complete.txt"
    if not overwrite and marker_file.exists():
        logging.info(f"Skipping skull stripping for subject {subject.id}, study {study.date}: marker exists.")
        return

    # Define file paths for registered images (assumed file names)
    t1_file = registered_path / "t1.nii.gz"
    flair_file = registered_path / "flair_reg.nii.gz"
    
    # Read the registered images using SimpleITK
    try:
        t1_sitk = sitk.ReadImage(str(t1_file))
        logging.info(f"Loaded T1 image from {t1_file}")
    except Exception as e:
        logging.error(f"Failed to load T1 image from {t1_file}: {e}")
        return

    try:
        flair_sitk = sitk.ReadImage(str(flair_file))
        logging.info(f"Loaded FLAIR image from {flair_file}")
    except Exception as e:
        logging.error(f"Failed to load FLAIR image from {flair_file}: {e}")
        return

    # Convert the FLAIR image from SimpleITK to ITK format for skull stripping.
    try:
        flair_itk = sitk_to_itk(sitk.Cast(flair_sitk, sitk.sitkInt16))
    except Exception as e:
        logging.error(f"Error converting FLAIR image to ITK format: {e}")
        return

    # Perform skull stripping on the FLAIR image.
    try:
        # skull_strip() returns an ITK skull mask and a SimpleITK version of the masked image.
        skull_mask_itk, flair_masked_sitk = skull_strip(flair_itk)
        logging.info("Skull stripping of FLAIR completed.")
    except Exception as e:
        logging.error(f"Error during skull stripping of FLAIR: {e}")
        return

    # Apply the skull mask (obtained from FLAIR) to the T1 image.
    try:
        t1_skullstrip_sitk = apply_mask(skull_mask_itk, t1_sitk)
        logging.info("Applied skull mask to T1 image.")
    except Exception as e:
        logging.error(f"Error applying skull mask to T1 image: {e}")
        return

    # Optionally, convert the skull mask to SimpleITK format for saving.
    try:
        skull_mask_sitk = itk_to_sitk(skull_mask_itk)
    except Exception as e:
        logging.error(f"Error converting skull mask to SimpleITK format: {e}")
        skull_mask_sitk = None

    # Save the skull-stripped images (and skull mask) to the skullstrip_path.
    try:
        t1_out_file = skullstrip_path / "t1_skullstrip.nii.gz"
        flair_out_file = skullstrip_path / "flair_skullstrip.nii.gz"
        sitk.WriteImage(t1_skullstrip_sitk, str(t1_out_file))
        sitk.WriteImage(flair_masked_sitk, str(flair_out_file))
        logging.info(f"Saved skull-stripped T1 to {t1_out_file} and FLAIR to {flair_out_file}")
        if skull_mask_sitk is not None:
            mask_out_file = skullstrip_path / "skull_mask.nii.gz"
            sitk.WriteImage(skull_mask_sitk, str(mask_out_file))
            logging.info(f"Saved skull mask to {mask_out_file}")
    except Exception as e:
        logging.error(f"Error saving skull-stripped images: {e}")
        return

    # Create the marker file to indicate that skull stripping is complete.
    marker_file.touch()
    logging.info(f"Skull stripping complete for subject {subject.id}, study {study.date}. Marker created at {marker_file}")


def batch_process_skullstrip(project, interim_directory, overwrite=False):
    """
    Batch process skull stripping for all subjects and studies.
    
    For each subject, iterates through all studies and processes the registered images
    in the "02_registered" folder by:
      - Skull stripping the FLAIR image to get a skull mask.
      - Applying that skull mask to the T1 image.
      - Saving the skull-stripped outputs in a "03_skullstrip" folder.
    
    Uses marker files to avoid reprocessing unless overwrite is specified.
    
    Parameters:
      project: A project object with a method all_subject() returning a list of subjects.
      interim_directory: Base directory (Path) where subject folders reside.
      overwrite: If True, reprocess all studies even if a marker file exists.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for skull stripping.")

    for subject in all_subjects:
        logging.info(f"Processing skull stripping for subject: {subject.subject_path}")
        studies = subject.all_study()
        for study in studies:
            try:
                process_study_skullstrip(subject, study, interim_directory, overwrite=overwrite)
            except Exception as e:
                logging.error(f"Error in skull stripping for subject {subject.id}, study {study.date}: {e}")


def inspect_skullstrip_for_study(subject, study, interim_directory, slice_idx=None, save_fig=False):
    """
    Visually inspect skull stripping results for a given study.
    
    Loads the skull-stripped T1, skull-stripped FLAIR, and skull mask from the "03_skullstrip" folder,
    then displays a montage:
      - Skull-stripped T1 image.
      - Skull-stripped FLAIR image.
      - T1 image with the skull mask overlaid.
    
    Parameters:
      subject: A subject object with an attribute 'id'.
      study: A study object with an attribute 'date'.
      interim_directory: Base directory (Path) where subject folders reside.
      slice_idx: Optional slice index to display; if None, the middle slice is used.
      save_fig: If True, the figure is saved as a PNG in the same folder.
    """
    study_folder = study.date.replace("/", ".")
    registered_path = interim_directory / subject.id / study_folder / "02_registration"
    skullstrip_path = interim_directory / subject.id / study_folder / "03_skullstrip"
    
    # Define file paths (adjust file names if needed)

    flair_file = registered_path / "flair_reg.nii.gz"
    flair_ss_file = skullstrip_path / "flair_skullstrip.nii.gz"
    mask_file  = skullstrip_path / "skull_mask.nii.gz"
    
    try:
        flair_img    = sitk.ReadImage(str(flair_file))
        flair_ss_img = sitk.ReadImage(str(flair_ss_file))
        mask_img  = sitk.ReadImage(str(mask_file))
        logging.info(f"Loaded skullstrip images for subject {subject.id}, study {study.date}")
    except Exception as e:
        logging.error(f"Error loading skullstrip images for subject {subject.id}, study {study.date}: {e}")
        return
    
    # Convert images to numpy arrays (shape: [slices, height, width])
    flair_arr    = sitk.GetArrayFromImage(flair_img)
    flair_ss_arr = sitk.GetArrayFromImage(flair_ss_img)
    mask_arr  = sitk.GetArrayFromImage(mask_img)
    
    # Use the middle slice if not provided
    if slice_idx is None:
        slice_idx = flair_arr.shape[0] // 2
    
    # Create a montage of three views:
    #  1. Skull-stripped T1.
    #  2. Skull-stripped FLAIR.
    #  3. T1 overlaid with skull mask.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(flair_arr[slice_idx, :, :], cmap="gray")
    axs[0].set_title("Original FLAIR")
    axs[0].axis("off")
    
    axs[1].imshow(flair_ss_arr[slice_idx, :, :], cmap="gray")
    axs[1].set_title("Skull-stripped FLAIR")
    axs[1].axis("off")
    
    axs[2].imshow(flair_arr[slice_idx, :, :], cmap="gray")
    axs[2].imshow(mask_arr[slice_idx, :, :], cmap="Reds", alpha=0.5)
    axs[2].set_title("Overlay: FLAIR + Skull Mask")
    axs[2].axis("off")
    
    plt.suptitle(f"Subject: {subject.id} | Study: {study.date} | Slice: {slice_idx}")
    plt.tight_layout()
    
    if save_fig:
        fig_path = skullstrip_path / f"inspection_slice_{slice_idx}.png"
        plt.savefig(str(fig_path))
        logging.info(f"Saved inspection figure to {fig_path}")
    else:
        plt.show()
    plt.close(fig)


def batch_inspect_skullstrip(project, interim_directory, overwrite=False, save_fig=False):
    """
    Batch process visual inspection for skull stripping outputs.
    
    For each subject and each study, this function calls `inspect_skullstrip_for_study`
    to display (or save) a montage of the skull stripping results.
    
    A marker file ("visual_inspection_complete.txt") is used in the "03_skullstrip" folder
    to prevent re-inspection unless `overwrite=True`.
    
    Parameters:
      project: A project object with a method all_subject() returning a list of subjects.
      interim_directory: Base directory (Path) where subject folders reside.
      overwrite: If True, re-run inspection even if marker file exists.
      save_fig: If True, save the inspection figure instead of displaying it.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for visual inspection.")
    
    for subject in all_subjects:
        studies = subject.all_study()
        for study in studies:
            study_folder = study.date.replace("/", ".")
            skullstrip_path = interim_directory / subject.id / study_folder / "03_skullstrip"
            skullstrip_path.mkdir(parents=True, exist_ok=True)
            marker_file = skullstrip_path / "visual_inspection_complete.txt"
            
            if not overwrite and marker_file.exists():
                logging.info(f"Skipping visual inspection for subject {subject.id}, study {study.date} (already inspected).")
                continue
            
            try:
                inspect_skullstrip_for_study(subject, study, interim_directory, save_fig=save_fig)
                # Create marker file after successful inspection
                marker_file.touch()
                logging.info(f"Visual inspection completed for subject {subject.id}, study {study.date}")
            except Exception as e:
                logging.error(f"Error during visual inspection for subject {subject.id}, study {study.date}: {e}")







