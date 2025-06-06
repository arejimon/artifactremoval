import logging
import pickle
import numpy as np
import SimpleITK as sitk
import shutil
import matplotlib.pyplot as plt
import random
from datetime import datetime

from artifactremoval.midas import NNFitDataset

#------------------------------------------------------------------------------
# Helper: Create Unique IDs from Final Mask
#------------------------------------------------------------------------------

def create_unique_ids_from_final_mask(subject, study, interim_directory):
    """
    Loads the final mask (from the "04_downsample" folder) for a given study,
    finds all voxel indices where the mask is positive, and returns a list of unique IDs.
    
    The unique ID format is:
        '{subject.id}_{study.date.replace("/", ".")}_{i}_{j}_{k}'
    
    Parameters:
      subject: Subject object (with attribute id).
      study: Study object (with attribute date).
      interim_directory (Path): Base directory where subject folders reside.
    
    Returns:
      unique_ids (list of str): List of unique IDs.
    """
    study_folder = interim_directory / subject.id / study.date.replace("/", ".")
    downsample_folder = study_folder / "06_downsample"
    try:
        final_mask_path = downsample_folder / "final_mask.nii.gz"
        final_mask_img = sitk.ReadImage(str(final_mask_path))
        final_mask_arr = sitk.GetArrayFromImage(final_mask_img)  # shape: [slices, height, width]
        indices = np.argwhere(final_mask_arr > 0)
        unique_ids = [f"{subject.id}_{study.date.replace('/', '.')}_{i}_{j}_{k}" for i, j, k in indices]
        logging.info(f"Created {len(unique_ids)} unique IDs for subject {subject.id}, study {study.date}")
        return unique_ids
    except Exception as e:
        logging.error(f"Error creating unique IDs for subject {subject.id}, study {study.date}: {e}")
        return []

#------------------------------------------------------------------------------
# Single-Study Spectral Extraction
#------------------------------------------------------------------------------

def perform_spectral_extraction(study, unique_ids, output_folder, overwrite=False):
    """
    For a single study, load spectral data (using NNFitDataset or fallback defaults)
    and extract spectral information at the voxel indices encoded in unique_ids.
    
    The function saves the output (a list of dictionaries with keys:
      "unique_id", "spectral_sampling", "raw_spectrum", "midas_fit", "nnfit")
    to a file named "unique_spectral_data.pkl" in the output_folder.
    
    Parameters:
      study: The study object (provides methods like si_sampling(), fitt(), etc.)
      unique_ids (list of str): List of unique IDs encoding voxel indices.
      output_folder (Path): Folder (e.g., "07_uniqueid") to save output.
      overwrite (bool): If True, reprocess even if the output file exists.
    
    Returns:
      spectral_data (list of dict)
    """
    output_file = output_folder / "unique_spectral_data.pkl"
    if not overwrite and output_file.exists():
        logging.info(f"Spectral data already exists at {output_file}. Skipping extraction.")
        with open(output_file, "rb") as f:
            spectral_data = pickle.load(f)
        return spectral_data

    spectral_data = []
    try:
        # Attempt to load spectral data from NNFitDataset
        nnfit_data = NNFitDataset(study)
        si_sample = study.si_sampling()
        siref_sample = study.siref_sampling()
        siref = study.siref()
        spec = nnfit_data.load_spectra()
        nnfit = nnfit_data.load_baseline() + nnfit_data.load_peaks()
        midasfit = study.fitt()
        logging.info("Loaded spectral data from NNFitDataset.")
    except Exception as e:
        logging.error(f"Error loading NNFitDataset: {e}. Falling back to defaults.")
        si_sample = study.si_sampling()
        siref_sample = study.siref_sampling()
        siref = study.siref()
        default_shape = (64, 64, 32, 512)
        try:
            spec = study.si()
            #spec = np.transpose(spec, (2, 1, 0, 3))
        except Exception as ex:
            logging.error(f"Error loading study.si(): {ex}. Using zeros for spectra.")
            spec = np.zeros(default_shape, dtype=np.float32)
        nnfit = np.zeros(default_shape, dtype=np.float32)
        try:
            midasfit = study.fitt()
            #midasfit = np.transpose(midasfit, (2, 1, 0, 3))
        except Exception as midas_error:
            logging.error(f"Error loading MIDAS fit: {midas_error}. Using zeros for MIDAS fit.")
            midasfit = np.zeros(default_shape, dtype=np.float32)
    
    skipped_zero_count = 0  # Counter for spectra skipped because they are all zeros

    # Process each unique ID to extract the spectral data
    for unique_id in unique_ids:
        try:
            # Assume unique_id is formatted as: "subjectid_studydate_i_j_k"
            parts = unique_id.split('_')
            if len(parts) < 5:
                logging.warning(f"Skipping incomplete ID: {unique_id}")
                continue
            i, j, k = int(parts[-3]), int(parts[-2]), int(parts[-1])
            if i < spec.shape[0] and j < spec.shape[1] and k < spec.shape[2]:
                water_ijk    = np.real(siref[i, j, k, :])
                spectrum_ijk = np.real(spec[i, j, k, :])
                nnfit_ijk    = np.real(nnfit[i, j, k, :])
                midasfit_ijk = np.real(midasfit[i, j, k, :])
                # Skip if the raw spectrum is all zeros
                if np.all(spectrum_ijk == 0):
                    skipped_zero_count += 1
                    continue
                spectral_data.append({
                    "unique_id": unique_id,
                    "spectral_sampling": si_sample,
                    "spectral_siref_sampling": siref_sample,
                    "raw_spectrum": spectrum_ijk,
                    "water_siref": water_ijk,
                    "midas_fit": midasfit_ijk,
                    "nnfit": nnfit_ijk
                })
            else:
                logging.warning(f"Indices out of bounds for ID: {unique_id}")
        except Exception as ex:
            logging.error(f"Error processing unique ID {unique_id}: {ex}")
            continue

    logging.info(f"Skipped spectra (all zeros): {skipped_zero_count}")

    logging.info(f"Total number of collected spectra: {len(spectral_data)}")

    try:
        with open(output_file, "wb") as f:
            pickle.dump(spectral_data, f)
        logging.info(f"Saved spectral data to {output_file}")
    except Exception as e:
        logging.error(f"Error saving spectral data to {output_file}: {e}")

    spectral_data_4th = spectral_data[::4]
    logging.info(f"Total number of collected spectra after taking every 4th: {len(spectral_data_4th)}")
    
    output_file_4th = output_folder / "unique_spectral_data_4thspectra.pkl"
    try:
        with open(output_file_4th, "wb") as f:
            pickle.dump(spectral_data_4th, f)
        logging.info(f"Saved spectral data with every 4th spectrum to {output_file_4th}")
    except Exception as e:
        logging.error(f"Error saving 4th spectral data to {output_file_4th}: {e}")
    
    return spectral_data

#------------------------------------------------------------------------------
# Per-Study Processing for Spectral Extraction
#------------------------------------------------------------------------------

def process_study_spectral_extraction(subject, study, interim_directory, overwrite=False):
    """
    Processes spectral extraction for a single study.
    
    This function:
      - Constructs the study folder path.
      - If overwrite is True and the "07_uniqueid" folder exists, deletes it.
      - Creates an output folder ("07_uniqueid") in the study folder.
      - Calls create_unique_ids_from_final_mask() to generate unique IDs from the final mask.
      - Calls perform_spectral_extraction() to extract the spectral data.
      - Creates a marker file ("spectral_extraction_complete.txt") to avoid reprocessing.
    
    Parameters:
      subject: Subject object (with attribute id).
      study: Study object (with attribute date).
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, reprocess even if a marker file exists.
    
    Returns:
      spectral_data (list of dict)
    """
    study_folder = interim_directory / subject.id / study.date.replace("/", ".")
    uniqueid_folder = study_folder / "07_uniqueid"
    
    if overwrite and uniqueid_folder.exists():
        shutil.rmtree(uniqueid_folder)
        logging.info(f"Overwrite True: Removed previous '07_uniqueid' folder for subject {subject.id}, study {study.date}.")
    
    uniqueid_folder.mkdir(parents=True, exist_ok=True)
    marker_file = uniqueid_folder / "spectral_extraction_complete.txt"
    
    if not overwrite and marker_file.exists():
        logging.info(f"Skipping spectral extraction for subject {subject.id}, study {study.date} (marker exists).")
        return
    
    unique_ids = create_unique_ids_from_final_mask(subject, study, interim_directory)
    if not unique_ids:
        logging.warning(f"No unique IDs generated for subject {subject.id}, study {study.date}.")
        return
    
    spectral_data = perform_spectral_extraction(study, unique_ids, uniqueid_folder, overwrite=overwrite)
    marker_file.touch()
    logging.info(f"Spectral extraction complete for subject {subject.id}, study {study.date}. Marker created at {marker_file}")
    return spectral_data

#------------------------------------------------------------------------------
# Batch Processing for Spectral Extraction
#------------------------------------------------------------------------------

def batch_process_spectral_extraction(project, interim_directory, overwrite=False):
    """
    Batch processes spectral extraction for all subjects and studies.
    
    Iterates through each subject and study in the project and calls process_study_spectral_extraction().
    
    Parameters:
      project: Project object with method all_subject() returning a list of subjects.
      interim_directory (Path): Base directory where subject folders reside.
      overwrite (bool): If True, reprocess even if marker files exist.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for spectral extraction.")
    for subject in all_subjects:
        logging.info(f"Processing spectral extraction for subject: {subject.id}")
        studies = subject.all_study()
        for study in studies:
            try:
                process_study_spectral_extraction(subject, study, interim_directory, overwrite=overwrite)
            except Exception as e:
                logging.error(f"Error processing spectral extraction for subject {subject.id}, study {study.date}: {e}")

def verify_spectral_extraction_for_study(subject, study, interim_directory, save_fig=False, overwrite=False):
    """
    Verifies spectral extraction for a single study by randomly selecting three unique IDs
    from the saved spectral data (in "07_uniqueid") and plotting:
      - Top row: the downsampled flair image slice with a marker at the voxel location.
      - Bottom row: the extracted spectrum, with x-axis constructed from spec_sample.
      
    The x-axis is generated as follows:
        spectra_xaxis = np.linspace(spec_sample['left_edge_ppm'],
                                    spec_sample['left_edge_ppm'] - spec_sample['ppm_range'],
                                    spec_sample['spec_pts'])
    The x-axis is inverted, and axes are labeled "PPM" (x) and "Amplitude" (y).
    
    Parameters:
      subject: Subject object with attribute 'id'.
      study: Study object with attribute 'date'.
      interim_directory (Path): Base directory where subject folders reside.
      save_fig (bool): If True, saves the figure as a PNG instead of displaying it.
      overwrite (bool): (Optional) Forces re-verification even if markers exist.
    """
    study_folder = interim_directory / subject.id / study.date.replace("/", ".")
    uniqueid_folder = study_folder / "07_uniqueid"
    spectral_data_file = uniqueid_folder / "unique_spectral_data.pkl"
    
    if not spectral_data_file.exists():
        logging.error(f"Spectral data file does not exist for subject {subject.id}, study {study.date}")
        return

    # Load the saved spectral data
    with open(spectral_data_file, "rb") as f:
        spectral_data = pickle.load(f)
    
    if len(spectral_data) == 0:
        logging.warning(f"No spectral data available for subject {subject.id}, study {study.date}")
        return

    # Randomly sample three unique entries (or all if fewer than 3)
    sample_entries = random.sample(spectral_data, min(3, len(spectral_data)))

    # Load the downsampled flair image from the "06_downsample" folder
    downsample_folder = study_folder / "06_downsample"
    flair_downsampled_path = downsample_folder / "flair_vol_downsampled.nii.gz"
    try:
        flair_img = sitk.ReadImage(str(flair_downsampled_path))
        flair_arr = sitk.GetArrayFromImage(flair_img)  # assumed shape: [slices, height, width]
    except Exception as e:
        logging.error(f"Error loading downsampled flair image for subject {subject.id}, study {study.date}: {e}")
        return

    # Create a figure with 2 rows and N columns (N = number of sample entries)
    n = len(sample_entries)
    fig, axs = plt.subplots(2, n, figsize=(6*n, 10))
    if n == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    
    for idx, entry in enumerate(sample_entries):
        unique_id = entry["unique_id"]
        parts = unique_id.split('_')
        if len(parts) < 5:
            logging.warning(f"Skipping incomplete unique ID: {unique_id}")
            continue
        try:
            i, j, k = int(parts[-3]), int(parts[-2]), int(parts[-1])
        except Exception as e:
            logging.error(f"Error parsing indices from unique ID {unique_id}: {e}")
            continue
        
        # Verify that the slice index is within bounds
        if i < 0 or i >= flair_arr.shape[0]:
            logging.warning(f"Slice index {i} out of bounds for subject {subject.id}, study {study.date}")
            continue
        
        # Top row: Display the flair image slice with a red marker at (j,k)
        axs[0, idx].imshow(flair_arr[i, :, :], cmap="gray")
        axs[0, idx].plot(k, j, 'ro', markersize=8)
        axs[0, idx].set_title(f"Location: ({i},{j},{k})")
        axs[0, idx].axis("off")
        
        # Bottom row: Plot the extracted spectrum using spectral sampling info.
        spectrum = entry["raw_spectrum"]
        nnfit = entry["nnfit"]
        spec_sample = entry["spectral_sampling"]
        # Create the x-axis using the stored spectral sampling info:
        spectra_xaxis = np.linspace(spec_sample['left_edge_ppm'],
                                    spec_sample['left_edge_ppm'] - spec_sample['ppm_range'],
                                    spec_sample['spec_pts'])
        axs[1, idx].plot(spectra_xaxis, spectrum, linestyle='-', linewidth = 3)
        axs[1, idx].plot(spectra_xaxis, nnfit, linestyle = '--', linewidth = 3)
        axs[1, idx].set_xlabel("PPM")
        axs[1, idx].set_ylabel("Amplitude")
        axs[1, idx].grid(True)
        axs[1, idx].invert_xaxis()
        axs[1, idx].set_title(f"Unique ID: {unique_id}")
    
    plt.suptitle(f"Spectral Verification - Subject: {subject.id} | Study: {study.date}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_fig:
        # Create a unique filename using the current date and time.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = uniqueid_folder / f"spectral_verification_{timestamp}.png"
        plt.savefig(str(fig_path))
        logging.info(f"Saved spectral verification figure to {fig_path}")
    else:
        plt.show()
        plt.close(fig)

def batch_verify_spectral_extraction(project, interim_directory, save_fig=False, overwrite=False):
    """
    Batch verifies spectral extraction for all subjects and studies by calling
    verify_spectral_extraction_for_study for each study.
    
    Parameters:
      project: Project object with method all_subject() returning a list of subjects.
      interim_directory (Path): Base directory where subject folders reside.
      save_fig (bool): If True, saves each inspection figure; otherwise, displays interactively.
      overwrite (bool): If True, forces re-verification even if markers exist.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects for spectral verification.")
    for subject in all_subjects:
        studies = subject.all_study()
        for study in studies:
            try:
                verify_spectral_extraction_for_study(subject, study, interim_directory, save_fig=save_fig, overwrite=overwrite)
            except Exception as e:
                logging.error(f"Error during spectral verification for subject {subject.id}, study {study.date}: {e}")
