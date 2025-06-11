import logging
import SimpleITK as sitk
from pathlib import Path
import shutil

from artifactremoval.imgproc import itk_to_sitk

def convert_midas2nifti(study, output_path):
    """
    Converts raw MIDAS study data to NIfTI files for T1, FLAIR, SIREF, and Brain Mask.
    If one of the images is missing, logs a warning and returns None for that image.
    
    Parameters:
      study: A study object with methods t1(), flair(), ref(), and brain_mask().
      output_path: A Path object where the NIfTI files will be saved.
      
    Returns:
      A tuple of (t1_sitk, flair_sitk, siref_sitk, brainmask_sitk)
    """
    # Ensure output_path exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize image variables
    t1_sitk = flair_sitk = siref_sitk = brainmask_sitk = None

    # Convert T1
    try:
        t1_itk = study.t1()[1]
        t1_sitk = itk_to_sitk(t1_itk)
        save_t1_path = output_path / "unprocessed_t1.nii.gz"
        sitk.WriteImage(t1_sitk, str(save_t1_path))
        logging.info(f"Saved T1 at {save_t1_path}")
    except Exception as e:
        if "does not exist" in str(e):
            logging.warning(f"T1 Warning: {e}. Problem with T1.")
        else:
            raise

    # Convert FLAIR
    try:
        flair_itk = study.flair()[1]
        flair_sitk = itk_to_sitk(flair_itk)
        save_flair_path = output_path / "unprocessed_flair.nii.gz"
        sitk.WriteImage(flair_sitk, str(save_flair_path))
        logging.info(f"Saved FLAIR at {save_flair_path}")
    except Exception as e:
        if "does not exist" in str(e):
            logging.warning(f"FLAIR Warning: {e}. Problem with FLAIR.")
        else:
            raise

    # Convert SIREF
    try:
        siref_itk = study.ref()[1]
        siref_sitk = itk_to_sitk(siref_itk)
        save_siref_path = output_path / "unprocessed_siref.nii.gz"
        sitk.WriteImage(siref_sitk, str(save_siref_path))
        logging.info(f"Saved SIREF at {save_siref_path}")
    except Exception as e:
        if "does not exist" in str(e):
            logging.warning(f"SIREF Warning: {e}. Problem with SIREF.")
        else:
            raise

    # Convert Brain Mask
    try:
        brainmask_itk = study.brain_mask()[1]
        brainmask_sitk = itk_to_sitk(brainmask_itk)
        save_brainmask_path = output_path / "unprocessed_brainmask.nii.gz"
        sitk.WriteImage(brainmask_sitk, str(save_brainmask_path))
        logging.info(f"Saved Brain Mask at {save_brainmask_path}")
    except Exception as e:
        if "does not exist" in str(e):
            logging.warning(f"Brain Mask Warning: {e}. Problem with Brain Mask.")
        else:
            raise
        
        # Convert Brain Mask
    try:
        qmap_itk = study.qmap()[1]
        qmap_sitk = itk_to_sitk(qmap_itk)
        save_qmap_path = output_path / "unprocessed_qmap.nii.gz"
        sitk.WriteImage(qmap_sitk, str(save_qmap_path))
        logging.info(f"Saved QMAP at {save_qmap_path}")
    except Exception as e:
        if "does not exist" in str(e):
            logging.warning(f"QMAP Warning: {e}. Problem with QMAP.")
        else:
            raise

    return t1_sitk, flair_sitk, siref_sitk, brainmask_sitk, qmap_sitk

# ---------------------------------------------------------------------
# Process a Single Study Conversion
# ---------------------------------------------------------------------
def process_study_conversion(subject, study, interim_directory):
    """
    For a given subject and study, convert MIDAS data to NIfTI files.
    
    The output is saved to:
      interim_directory / subject.id / study.date.replace("/", ".") / "01_unprocessed"
    
    Parameters:
      subject: An object representing the subject. It should have attributes like `id` and `subject_path`.
      study: An object representing the study. It should have a `date` attribute.
      interim_directory: Base directory (Path) where the output folders are created.
    """
    # Build the output directory path
    output_path = interim_directory / subject.id / study.date.replace("/", ".") / "01_unprocessed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Processing subject {subject.id} study {study.date} (output: {output_path})")
    
    try:
        # Convert MIDAS study data to NIfTI using the conversion function
        t1_sitk, flair_sitk, siref_sitk, brainmask_sitk, qmap_sitk = convert_midas2nifti(study, output_path)
        logging.info(f"Conversion completed for study {study.date} in subject {subject.subject_path}")
    except Exception as e:
        logging.error(f"Error processing subject {subject.subject_path} study {study.date}: {e}")

# ---------------------------------------------------------------------
# Batch Conversion Function
# ---------------------------------------------------------------------

def has_been_converted(output_path):
    """
    Check if the study has already been converted by looking for a marker file.
    """
    marker_file = output_path / "conversion_complete.txt"
    return marker_file.exists()

def mark_as_converted(output_path):
    """
    Create a marker file in the output folder to indicate conversion is complete.
    """
    marker_file = output_path / "conversion_complete.txt"
    marker_file.touch()

def batch_convert_midas2nifti(project, interim_directory, overwrite=False):
    """
    Batch process conversion of MIDAS data to NIfTI for all subjects and their studies.
    
    Parameters:
      project: A project object that provides a method all_subject() returning a list of subjects.
      interim_directory: Base directory (Path) where the output folders reside.
      overwrite: If True, process all studies even if they have been converted before.
                 If False, skip studies that have already been converted.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects to process.")
    
    for subject in all_subjects:
        logging.info(f"Processing subject: {subject.subject_path}")
        studies = subject.all_study()
        for study in studies:
            # Build the expected output directory for this study conversion.
            # (Assuming your conversion output is stored under "01_unprocessed")
            output_path = interim_directory / subject.id / study.date.replace("/", ".") / "01_unprocessed"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Skip this study if it has already been converted and we are not overwriting.
            if not overwrite and has_been_converted(output_path):
                logging.info(f"Skipping subject {subject.subject_path} study {study.date}: already converted.")
                continue

            try:
                process_study_conversion(subject, study, interim_directory)
                # Mark the study as converted once processing is complete.
                mark_as_converted(output_path)
                logging.info(f"Completed conversion for subject {subject.subject_path} study {study.date}")
            except Exception as e:
                logging.error(f"Error converting subject {subject.subject_path} study {study.date}: {e}")



# --------------------------------------------------------- STEP 2 of PREPROCESSING ---------------------------------------------------------------



def check_and_copy_missing_subject(subject, interim_directory):
    """
    For each study of the subject, check if the four unprocessed files exist.
    If one file is missing, attempt to copy it from another study folder
    within the same subject that has the file.
    
    Parameters:
      subject: A subject object (must have an 'id' attribute and a method all_study())
      interim_directory: Base directory (Path) where the subject folders are stored.
    """
    # Define the expected file names
    file_names = {
        "t1": "unprocessed_t1.nii.gz",
        "siref": "unprocessed_siref.nii.gz",
        "flair": "unprocessed_flair.nii.gz",
        "brainmask": "unprocessed_brainmask.nii.gz",
        "qmap": "unprocess_qmap.nii.gz"
    }
    
    # Build a dictionary mapping study date to the unprocessed folder path
    study_folders = {}
    for study in subject.all_study():
        study_folder = interim_directory / subject.id / study.date.replace("/", ".") / "01_unprocessed"
        study_folders[study.date] = study_folder

    # Now, iterate over each study folder for this subject
    for study_date, folder in study_folders.items():
        # Ensure the folder exists (if not, log and skip)
        if not folder.exists():
            logging.error(f"Folder {folder} for study {study_date} in subject {subject.id} does not exist.")
            continue

        for key, file_name in file_names.items():
            target_file = folder / file_name
            if not target_file.exists():
                logging.warning(f"File {file_name} is missing in study {study_date} for subject {subject.id}.")
                
                # Search in other study folders for this subject
                source_file = None
                source_study_date = None
                for other_date, other_folder in study_folders.items():
                    if other_date == study_date:
                        continue
                    candidate = other_folder / file_name
                    if candidate.exists():
                        source_file = candidate
                        source_study_date = other_date
                        break
                
                if source_file:
                    try:
                        shutil.copy2(source_file, target_file)
                        logging.info(f"Copied {file_name} from study {source_study_date} to study {study_date} for subject {subject.id}.")
                    except Exception as e:
                        logging.error(f"Failed to copy {file_name} for subject {subject.id} from study {source_study_date}: {e}")
                else:
                    logging.error(f"Could not find {file_name} in any other study for subject {subject.id}.")

def has_been_checked_missing(output_path):
    """
    Check if the missing files have already been checked for this study.
    """
    marker_file = output_path / "missing_files_checked.txt"
    return marker_file.exists()

def mark_as_checked_missing(output_path):
    """
    Create a marker file to indicate that missing files have been checked for this study.
    """
    marker_file = output_path / "missing_files_checked.txt"
    marker_file.touch()

def batch_check_and_copy_missing(project, interim_directory, overwrite=False):
    """
    For each subject in the project, check their study folders for missing files,
    and copy the missing files from another study (if available).
    
    Parameters:
      project: A project object that provides a method all_subject() returning a list of subjects.
      interim_directory: Base directory (Path) where the subject folders are stored.
      overwrite: If True, re-check even if missing files have already been handled.
    """
    all_subjects = project.all_subject()
    logging.info(f"Checking missing files for {len(all_subjects)} subjects.")
    
    for subject in all_subjects:
        logging.info(f"Processing subject: {subject.id}")
        studies = subject.all_study()
        for study in studies:
            # Define the output folder for this study (assumed to be the "01_unprocessed" folder)
            output_path = interim_directory / subject.id / study.date.replace("/", ".") / "01_unprocessed"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Skip this study if missing files have already been checked and we are not overwriting.
            if not overwrite and has_been_checked_missing(output_path):
                logging.info(f"Skipping subject {subject.id} study {study.date}: missing files already checked.")
                continue

            try:
                # Call your helper function to check and copy missing files.
                # Here we assume you have defined this function to accept (subject, interim_directory, study).
                check_and_copy_missing_subject(subject, interim_directory)
                
                # Mark this study as processed for missing file checks.
                mark_as_checked_missing(output_path)
                logging.info(f"Missing files checked for subject {subject.id} study {study.date}")
            except Exception as e:
                logging.error(f"Error processing missing files for subject {subject.id} study {study.date}: {e}")