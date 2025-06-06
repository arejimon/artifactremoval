import logging
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path


#--------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------

def parse_unique_id(unique_id, subject, study):
    """
    Parses a unique ID using the provided subject and study information.
    
    The unique ID is expected to be in the format:
      '{subject.id}_{study.date.replace("/", ".")}_{i}_{j}_{k}'
      
    This function splits the unique ID from the right into four parts. The first part should be
    the combination of the subject ID and the study date (with '/' replaced by '.'). It then
    verifies that these values match the provided subject and study, and extracts the three indices.
    
    Parameters:
      unique_id (str): The unique ID string.
      subject: The subject object (with attribute 'id').
      study: The study object (with attribute 'date').
      
    Returns:
      tuple: (subject_id, study_date, (i, j, k))
      
    Raises:
      ValueError: If the unique ID format is incorrect or the extracted subject and date do not match.
    """
    # Split the unique ID from the right into 4 parts.
    parts = unique_id.rsplit('_', 3)
    if len(parts) != 4:
        raise ValueError(f"Unique ID '{unique_id}' does not have the expected format.")
    
    extracted_prefix = parts[0]  # This should be "subject.id_studyDate"
    expected_prefix = f"{subject.id}_{study.date.replace('/', '.')}"
    
    if extracted_prefix != expected_prefix:
        raise ValueError(f"Mismatch in unique ID: expected prefix '{expected_prefix}', got '{extracted_prefix}'")
    
    try:
        i, j, k = map(int, parts[1:])
    except Exception as e:
        raise ValueError(f"Error parsing indices from unique ID '{unique_id}': {e}")
    
    return subject.id, study.date, (i, j, k)

def scale_coordinates(coords, source_shape, target_shape):
    """
    Scale coordinates from a lower-resolution array (source_shape) to a higher-resolution array (target_shape).
    """
    logging.info("SCALING COORDINATES!! CHECK THESE COORDINATES")
    logging.info(f"SIREF Shape: {source_shape}")
    logging.info(f"Target T1 Shape: {target_shape}")

    scale_factors = [target / source for source, target in zip(source_shape, target_shape)]
    scaled_coords = tuple(int(coord * scale) for coord, scale in zip(coords, scale_factors))
    return scaled_coords

def plot_slice(ax, image, coord, orientation, full_coords, title=None, xlabel=None, ylabel=None):
    """
    Plots a single slice of a 3D image with crosshairs marking the voxel location.
    
    Parameters:
      ax: Matplotlib Axes to plot on.
      image: 3D NumPy array with shape (Z, Y, X).
      coord: The slice index along the specified orientation.
      orientation: One of 'axial', 'sagittal', or 'coronal'.
      full_coords: A tuple (i, j, k) representing the voxel location in the full image.
      title, xlabel, ylabel: Optional strings for labeling the plot.
    
    Returns:
      None.
    """
    Z, Y, X = image.shape
    i, j, k = full_coords

    if orientation == 'axial':
        # Axial: slice along Z (i.e. image[coord, :, :])
        # Flip the image horizontally (mirror it)
        slice_data = np.fliplr(image[coord, :, :])
        # Adjust the crosshair: the horizontal coordinate becomes X - k - 1.
        ch_row, ch_col = j, X - k - 1
    elif orientation == 'sagittal':
        # Sagittal: slice along X (i.e. image[:, :, coord]), where coord = k.
        # Using a 180° rotation.
        slice_data = np.rot90(image[:, :, coord], -2)
        # After 180° rotation, original (i, j) becomes (Z - i - 1, Y - j - 1)
        ch_row, ch_col = Z - i - 1, Y - j - 1
    elif orientation == 'coronal':
        # Coronal: slice along Y (i.e. image[:, coord, :]), where coord = j.
        # Using a 180° rotation.
        slice_data = np.rot90(image[:, coord, :], -2)
        # After 180° rotation, original (i, k) becomes (Z - i - 1, X - k - 1)
        ch_row, ch_col = Z - i - 1, X - k - 1
    else:
        raise ValueError("Invalid orientation. Use 'axial', 'sagittal', or 'coronal'.")

    ax.imshow(slice_data, cmap='gray', aspect='equal')
    ax.axhline(ch_row, color='r', linestyle='--')
    ax.axvline(ch_col, color='r', linestyle='--')
    if title:
        ax.set_title(title, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


#--------------------------------------------------------------------------
# Single-Input Function: Save Orthogonal Slices for a Unique ID
#--------------------------------------------------------------------------

def save_orthogonal_slices_for_unique_id(unique_id, subject, study, t1_image, output_dir):
    """
    For a given unique ID (formatted as "subjectID_studyDate_i_j_k"), extract the voxel indices,
    scale them from a source resolution (assumed to be (32, 64, 64)) to the full T1 image resolution,
    and save orthogonal slices (axial, sagittal, coronal) with a crosshair at the voxel location.
    
    The slices are saved as PNG files in the output_dir (organized by subject/date).
    """
    try:
        subject, date, indices = parse_unique_id(unique_id, subject, study)
        if len(indices) < 3:
            raise ValueError("Incomplete indices in unique ID: " + unique_id)
        i, j, k = indices
        
        # Our downsampled coordinates come from the spectral extraction step.
        # Assume they are in (depth, height, width) order from a downsampled image with shape (32, 64, 64).
        source_shape = (32, 64, 64)
        # Get the full-resolution image shape from the T1 image.
        # Note: SimpleITK's GetSize() returns (width, height, depth). For consistency, we use the numpy array shape,
        # which is (depth, height, width).
        t1_arr = sitk.GetArrayFromImage(t1_image)  # shape: (depth, height, width)
        target_shape = t1_arr.shape  # (depth, height, width)
        
        # Scale the coordinates from source resolution to target resolution.
        scaled_coords = scale_coordinates((i, j, k), source_shape, target_shape)
        i_full, j_full, k_full = scaled_coords  # i_full = slice index (depth), j_full = row, k_full = column
        
        
        # Save axial slice (slice along depth): use i_full; crosshair at (j_full, k_full)
        axial_path = output_dir / f"{unique_id}_axial.png"
        fig, ax = plt.subplots(figsize=(5,5))
        plot_slice(ax, t1_arr, i_full, 'axial', scaled_coords)
        ax.axis('off')
        fig.savefig(str(axial_path), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Save sagittal slice (slice along width): use k_full; crosshair at (j_full, i_full)
        sagittal_path = output_dir / f"{unique_id}_sagittal.png"
        fig, ax = plt.subplots(figsize=(5,5))
        plot_slice(ax, t1_arr, k_full, 'sagittal', scaled_coords)
        ax.axis('off')
        fig.savefig(str(sagittal_path), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Save coronal slice (slice along height): use j_full; crosshair at (k_full, i_full)
        coronal_path = output_dir / f"{unique_id}_coronal.png"
        fig, ax = plt.subplots(figsize=(5,5))
        plot_slice(ax, t1_arr, j_full, 'coronal', scaled_coords)
        ax.axis('off')
        fig.savefig(str(coronal_path), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        logging.info(f"Saved slices for unique ID {unique_id} in {output_dir}")
    except Exception as e:
        logging.error(f"Error saving slices for {unique_id}: {e}")

#--------------------------------------------------------------------------
# Per-Study Processing Function
#--------------------------------------------------------------------------

def process_study_slice_extraction(subject, study, interim_directory, overwrite=False):
    """
    For a given subject and study, this function:
      - Constructs the study folder path (using subject.id and study.date).
      - Loads the registered T1 image from the "02_registration/t1.nii.gz" file.
      - Loads unique spectral data from the "08_uniqueid/unique_spectral_data_4thspectra.pkl" file to obtain unique IDs.
      - Creates an output folder (e.g., "08_slices") in the study folder.
      - Iterates over each unique ID and calls save_orthogonal_slices_for_unique_id().
      - Writes a marker file ("slice_extraction_complete.txt") to prevent reprocessing.
    
    If overwrite is True, any existing output folder ("08_slices") is deleted first.
    """
    study_folder = interim_directory / subject.id / study.date.replace("/", ".")
    
    # Load the registered T1 image from "02_registration/t1.nii.gz"
    registration_folder = study_folder / "02_registration"
    t1_path = registration_folder / "t1.nii.gz"
    try:
        t1_image = sitk.ReadImage(str(t1_path))
    except Exception as e:
        logging.error(f"Error loading T1 image for subject {subject.id}, study {study.date}: {e}")
        return
    
    # Load unique spectral data from "07_uniqueid/unique_spectral_data_4thspectra.pkl"
    uniqueid_folder = study_folder / "07_uniqueid"
    spectral_data_file = uniqueid_folder / "unique_spectral_data_4thspectra.pkl"
    if not spectral_data_file.exists():
        logging.error(f"Spectral data file does not exist for subject {subject.id}, study {study.date}")
        return
    try:
        with open(spectral_data_file, "rb") as f:
            spectral_data = pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading spectral data for subject {subject.id}, study {study.date}: {e}")
        return
    
    # Create output folder for slices ("08_slices")
    slices_folder = study_folder / "08_slices"
    if overwrite and slices_folder.exists():
        shutil.rmtree(slices_folder)
        logging.info(f"Overwrite True: Removed previous slices folder {slices_folder}")
    slices_folder.mkdir(parents=True, exist_ok=True)
    
    marker_file = slices_folder / "slice_extraction_complete.txt"
    if not overwrite and marker_file.exists():
        logging.info(f"Skipping slice extraction for subject {subject.id}, study {study.date} (marker exists).")
        return
    
    # Process each spectral data entry (each has a unique_id)
    for entry in spectral_data:
        unique_id = entry.get("unique_id")
        if unique_id is None:
            logging.warning("Spectral entry missing unique_id, skipping.")
            continue
        save_orthogonal_slices_for_unique_id(unique_id, subject, study, t1_image, slices_folder)
    
    marker_file.touch()
    logging.info(f"Slice extraction complete for subject {subject.id}, study {study.date}. Marker created at {marker_file}")

#--------------------------------------------------------------------------
# Batch Processing Function
#--------------------------------------------------------------------------

def batch_process_slice_extraction(project, interim_directory, overwrite=False):
    """
    Iterates over all subjects and studies in the project and calls process_study_slice_extraction()
    to extract and save orthogonal T1 slices for each unique spectral data entry.
    
    Parameters:
      project: A project object that provides a method all_subject() returning a list of subjects.
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, reprocess even if marker files exist.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for slice extraction.")
    for subject in all_subjects:
        studies = subject.all_study()
        for study in studies:
            try:
                process_study_slice_extraction(subject, study, interim_directory, overwrite=overwrite)
            except Exception as e:
                logging.error(f"Error processing slice extraction for subject {subject.id}, study {study.date}: {e}")