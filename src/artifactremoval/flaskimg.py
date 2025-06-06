import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

from artifactremoval.imgproc import sitk_to_npy, itk_to_sitk

def parse_unique_id(unique_id):
    try:
        parts = unique_id.split('_')  # Split the string by '_'
        i, j, k = parts[-3:]
        date = parts[-4]
        project_id = "_".join(parts[:-4])  
        return str(project_id), str(date), int(i), int(j), int(k)  # Convert to integers
    except ValueError as e:
        print(f"Error parsing unique_id '{unique_id}': {e}")
        return None  # Handle invalid IDs gracefully

def find_subject_and_date(subjects, target_subject_id, target_date):
    """
    Searches for a specific subject and verifies if a target date exists in their studies.

    Args:
        subjects (list): List of subject objects.
        target_subject_id (str): The id of the target subject.
        target_date (str): The target date to search for.

    Returns:
        dict: A dictionary containing the study object and subject id if found,
              or None if the subject or date is not found.
    """
    # Use list comprehension and next to find the study with the target date
    result = next(
        (
            {"study": study, "subject_id": subject.id, "study_date": study.date}
            for subject in subjects
            if hasattr(subject, "id") and subject.id == target_subject_id
            and hasattr(subject, "all_study") and isinstance(subject.all_study(), list)
            for study in subject.all_study()
            if hasattr(study, "date") and study.date == target_date
        ),
        None  # Default if no match is found
    )
    
    # If no match, check if the subject exists but the date does not
    if result is None:
        subject_exists = any(
            subject.id == target_subject_id
            for subject in subjects
            if hasattr(subject, "id")
        )
        if subject_exists:
            return {"study": None, "subject_id": target_subject_id}  # Subject exists, but no date
        else:
            return None  # Subject not found

    return result

def getT1(unique_id, allsubj):
    """
    Get the T1-weighted brain image for a specific subject and date.

    Args:
        unique_id (str): A unique ID containing subject and date information.
        allsubj (list): List of all subject objects.

    Returns:
        numpy.ndarray: The T1 brain image as a NumPy array.
    """
    # Parse subject and date from the unique ID
    subject, date, *_ = parse_unique_id(unique_id)

    # Replace dots in the date with slashes if needed
    target_date = date.replace('.', '/')

    # Find the subject and date
    result = find_subject_and_date(allsubj, subject, target_date)

    # Check the result and retrieve the T1 image
    if result:
        if result["study"]:
            print(f"Subject '{result['subject_id']}' exists, and a matching study with the date {result['study_date']} was found.")
        else:
            print(f"Subject '{result['subject_id']}' exists, but no study with the date {target_date} was found.")
            raise ValueError(f"No study found for subject {subject} on date {target_date}")
    else:
        print(f"Subject '{subject}' was not found.")
        raise ValueError(f"Subject {subject} not found in the dataset.")

    # Extract and preprocess the T1 image
    t1_itk = result['study'].t1()[1]
    t1_sitk = itk_to_sitk(t1_itk)
    t1_npy = sitk_to_npy(t1_sitk)

    return t1_npy

def plot_slice(ax, image, coord, orientation, crosshair_coords, title = None, xlabel = None, ylabel = None):
    """
    Utility function to plot a single slice with crosshairs.
    """
    if orientation == 'sagittal':
        slice_data = np.rot90(image[coord, :, :])
        crosshair_h =  abs(image.shape[2] - crosshair_coords[0])
        crosshair_v = crosshair_coords[1]
    elif orientation == 'axial':
        slice_data = image[:, :, coord]
        crosshair_h, crosshair_v = crosshair_coords
    elif orientation == 'coronal':
        slice_data = np.rot90(image[:, coord, :])
        crosshair_v = crosshair_coords[0]
        crosshair_h = abs(image.shape[2] - crosshair_coords[1])
    else:
        raise ValueError("Invalid orientation. Use 'sagittal', 'axial', or 'coronal'.")
    
    ax.imshow(slice_data, cmap='gray', aspect='equal')
    ax.axhline(crosshair_h, color='r', linestyle='--')  # Horizontal crosshair
    ax.axvline(crosshair_v, color='r', linestyle='--')  # Vertical crosshair
    ax.set_title(title, fontsize = 12)
    ax.set_xlabel(xlabel, fontsize = 10)
    ax.set_ylabel(ylabel, fontsize = 10)

def scale_coordinates(coords, source_shape, target_shape):
    """
    Scale coordinates from a lower-resolution array to a higher-resolution array.

    Args:
        coords (tuple or list): The coordinates in the lower-resolution array (e.g., (x, y, z)).
        source_shape (tuple): Shape of the lower-resolution array (e.g., (64, 64, 32)).
        target_shape (tuple): Shape of the higher-resolution array (e.g., (256, 256, 160)).

    Returns:
        tuple: Scaled coordinates in the higher-resolution array.
    """
    # Calculate scaling factors for each dimension
    scale_factors = [target / source for source, target in zip(source_shape, target_shape)]
    
    # Scale the coordinates
    scaled_coords = tuple(int(coord * scale) for coord, scale in zip(coords, scale_factors))

    return scaled_coords

def sort_unique_ids(unique_ids):
    """
    Sort the unique IDs by subject, date, and other indices.

    Args:
        unique_ids (list): List of unique IDs to be sorted.

    Returns:
        list: Sorted list of unique IDs.
    """
    try:
        # Parse each unique ID into components and sort by subject, date, and indices
        sorted_ids = sorted(unique_ids, key=lambda uid: parse_unique_id(uid))
        return sorted_ids
    except Exception as e:
        print(f"Error while sorting unique IDs: {str(e)}")
        return unique_ids

def save_orthogonal_slices(unique_id, brain_image, output_dir):
    try:
        subject, date, *indices = parse_unique_id(unique_id)
        if len(indices) < 3:
            raise ValueError("Incomplete indices in selected ID.")

        i, j, k = indices
        x, y, z = scale_coordinates(indices, (64, 64, 32), brain_image.shape)

        # Create subject-date subdirectory
        subject_date_dir = output_dir / f"{subject}/{date.replace('/', '-')}"
        subject_date_dir.mkdir(parents=True, exist_ok=True)

        # Save coronal slice
        coronal_path = subject_date_dir / f"{unique_id}_coronal.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_slice(ax, brain_image, x, 'sagittal', (z, y))
        ax.axis('off')
        fig.savefig(coronal_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Save sagittal slice
        sagittal_path = subject_date_dir / f"{unique_id}_sagittal.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_slice(ax, brain_image, y, 'coronal', (x, z))
        ax.axis('off')
        fig.savefig(sagittal_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Save axial slice
        axial_path = subject_date_dir / f"{unique_id}_axial.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_slice(ax, brain_image, z, 'axial', (x, y))
        ax.axis('off')
        fig.savefig(axial_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"Saved slices for {unique_id} in {subject_date_dir}")
    except Exception as e:
        print(f"Error saving slices for {unique_id}: {str(e)}")

def group_and_process_unique_ids(unique_ids, allsubj, output_dir):
    """
    Group unique IDs by subject and date, load the T1 brain image for each group, 
    and process them to save orthogonal slices.

    Args:
        unique_ids (list): List of unique IDs to group and process.
        allsubj (list): List of all subject objects.
        output_dir (Path): Path to the directory where slices will be saved.

    Returns:
        None
    """
    try:
        # Parse and sort the unique IDs by subject and date
        sorted_unique_ids = sorted(unique_ids, key=lambda uid: (parse_unique_id(uid)[0], parse_unique_id(uid)[1]))

        # Group IDs by (subject, date)
        grouped_ids = groupby(sorted_unique_ids, key=lambda uid: (parse_unique_id(uid)[0], parse_unique_id(uid)[1]))

        # Process each group
        for (subject, date), ids in grouped_ids:
            print(f"Processing subject: {subject}, date: {date}")

            # Call getT1 to get the brain image for the group
            brain_image = getT1(f"{subject}_{date}_1_1_1", allsubj)  # Pass the subject and date to get brain image
            
            # Process all unique IDs in the group
            for unique_id in ids:
                save_orthogonal_slices(unique_id, brain_image, output_dir)
        
        print("All groups processed successfully.")
    except Exception as e:
        print(f"Error during grouping and processing: {str(e)}")