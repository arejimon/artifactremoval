import SimpleITK as sitk
import logging
from pathlib import Path

def rigid_registration_euler_geometry(fixed_image, moving_image):
    # Initial alignment of the two volumes
    transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=200)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.7,
        numberOfIterations=150,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=5,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(transform)

    final_transform = registration_method.Execute(fixed_image, moving_image)

    return final_transform

def rigid_registration_euler_moments(fixed_image, moving_image):
    # Initial alignment of the two volumes
    transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    # Multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=200)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.7,
        numberOfIterations=150,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=5,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(transform)

    final_transform = registration_method.Execute(fixed_image, moving_image)

    return final_transform

def rigid_registration_versor_moments(fixed_image, moving_image):
    # Initial alignment of the two volumes
    transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.VersorRigid3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    # Multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=80)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.9,
        numberOfIterations=150,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(transform)

    final_transform = registration_method.Execute(fixed_image, moving_image)

    return final_transform

def resample_image(itkimg, mask=False, size=[256,256,160]):
    
    # Create reference image with parameters and sizes that I want
    reference_image = sitk.Image(size, itkimg.GetPixelIDValue())
    reference_image.SetOrigin(itkimg.GetOrigin())
    reference_image.SetSpacing([(sz*spc)/new_sz for sz,spc,new_sz in zip(itkimg.GetSize(), itkimg.GetSpacing(), size)])
    reference_image.SetDirection(itkimg.GetDirection())

    # Resample original image to the reference
    initial_transform = sitk.CenteredTransformInitializer(reference_image, 
                                                      itkimg, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

    if mask == True:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear

    img_resampled = sitk.Resample(itkimg, reference_image, initial_transform, interpolator, 0.0, itkimg.GetPixelID())

    return img_resampled

class BestTransformTracker:
    def __init__(self, registration_method, initial_transform):
        self.best_metric = float("inf")  # Initialize with a large value
        self.best_parameters = None  # Store best transform parameters
        self.registration_method = registration_method
        self.initial_transform = initial_transform

    def __call__(self):
        current_metric = self.registration_method.GetMetricValue()
        current_parameters = self.registration_method.GetOptimizerPosition()  # Get current transform parameters

        #print(f"Iteration: {self.registration_method.GetOptimizerIteration()}, Metric: {current_metric}")

        if current_metric < self.best_metric:  # Save the best parameters found so far
            self.best_metric = current_metric
            self.best_parameters = current_parameters

    def get_best_transform(self):
        """ Returns the best transform with saved parameters """
        if self.best_parameters is not None:
            best_transform = sitk.Transform(self.initial_transform)
            best_transform.SetParameters(self.best_parameters)
            return best_transform
        else:
            return self.initial_transform  # Return initial transform if no improvement was found

            
def rigid_registration_best_transform(fixed_image, moving_image):
    # Initial alignment of the two volumes
    transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.VersorRigid3DTransform(), 
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    
    registration_method = sitk.ImageRegistrationMethod()

    # Initialize Best Transform Tracker
    tracker = BestTransformTracker(registration_method, transform)
    registration_method.AddCommand(sitk.sitkIterationEvent, tracker)
    
    # Multi-resolution rigid registration using Mutual Information
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.2)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=150,
        convergenceMinimumValue=1e-5,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetOptimizerScalesFromIndexShift()
    registration_method.SetInitialTransform(transform)
    
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Return the best transform instead of the last one
    best_transform = tracker.get_best_transform()

    return best_transform


# ---------------------------------------------------------------------
# Registration function 
# ---------------------------------------------------------------------
def register_unprocessed2t1(input_path):
    """
    Attempts to read T1, FLAIR, SIREF, and brainmask from the specified directory.
    If T1 is missing, the function will return early (cannot do registration).
    If FLAIR, SIREF, or brainmask are missing, those steps will be skipped.
    
    Also saves the computed registration transforms as .tfm files.
    """
    
    # Make sure 'input_path' is either a string or a Path
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    # Helper to read image if it exists
    def try_read_image(path):
        if path.exists():
            try:
                img = sitk.ReadImage(str(path))
                logging.info(f"\tSuccessfully loaded {path.name}")
                return img
            except Exception as e:
                logging.error(f"\tError reading {path.name}: {e}")
                return None
        else:
            logging.warning(f"\t{path.name} not found at {path}")
            return None

    # Filenames
    t1_path    = input_path / "unprocessed_t1.nii.gz"
    flair_path = input_path / "unprocessed_flair.nii.gz"
    siref_path = input_path / "unprocessed_siref.nii.gz"
    bm_path    = input_path / "unprocessed_brainmask.nii.gz"

    # Attempt to load each
    t1_sitk    = try_read_image(t1_path)
    flair_sitk = try_read_image(flair_path)
    siref_sitk = try_read_image(siref_path)
    bm_sitk    = try_read_image(bm_path)

    # If T1 is missing, bail early:
    if t1_sitk is None:
        logging.error("\tT1 is missing. Cannot perform registration.")
        return None, None, None, None

    # We'll collect outputs in a list for clarity
    # [0: T1, 1: SIREF_registered, 2: BM_registered, 3: FLAIR_registered, 4: Transform SIREF, 5: Transform FLAIR]
    results = [t1_sitk, None, None, None, None, None]
    logging.info(f"\tSize of T1: {t1_sitk.GetSize()}")

    # For slicing in the z-dimension
    fixed_image = t1_sitk
    mid_t1 = fixed_image.GetSize()[2] // 2

    num_mid_slices = 10
    half_window = num_mid_slices // 2

    # --------------------
    # Handle SIREF + BM
    # --------------------
    if siref_sitk is not None:
        mid_siref = siref_sitk.GetSize()[2] // 2

        # How many T1 slices for the same physical thickness as num_mid_slices in siref
        mid_t1siref_10 = int(num_mid_slices * (siref_sitk.GetSpacing()[2] / fixed_image.GetSpacing()[2])) // 2

        # Extract mid slices for registration
        moving_siref_mid = siref_sitk[:, :, (mid_siref - half_window):(mid_siref + half_window)]
        fixed_siref_mid  = fixed_image[:, :, (mid_t1 - mid_t1siref_10):(mid_t1 + mid_t1siref_10)]

        # Rigid registration (Assuming you have defined this function elsewhere)
        transform_siref = rigid_registration_euler_moments(fixed_siref_mid, moving_siref_mid)

        # Resample SIREF
        siref_reg_sitk = sitk.Resample(
            siref_sitk,
            fixed_image,
            transform_siref,
            sitk.sitkLinear,
            0.0,
            siref_sitk.GetPixelID()
        )
        logging.info(f"\tNew SIREF Size: {siref_reg_sitk.GetSize()}")
        results[1] = siref_reg_sitk
        results[4] = transform_siref

        # If BM is present, register using the same transform
        if bm_sitk is not None:
            bm_reg_sitk = sitk.Resample(
                bm_sitk,
                fixed_image,
                transform_siref,
                sitk.sitkLinear,
                0.0,
                bm_sitk.GetPixelID()
            )
            logging.info(f"\tNew Brain Mask Size: {bm_reg_sitk.GetSize()}")
            results[2] = bm_reg_sitk
    else:
        logging.warning("\tSIREF missing -> skipping SIREF & BM registration.")
    
    # --------------------
    # Handle FLAIR
    # --------------------
    if flair_sitk is not None:
        mid_flair = flair_sitk.GetSize()[2] // 2

        # Window around the mid-slice
        mid_t1flair_10 = int(num_mid_slices * (flair_sitk.GetSpacing()[2] / fixed_image.GetSpacing()[2])) // 2

        # Extract mid slices
        moving_flair_mid = flair_sitk[:, :, (mid_flair - half_window):(mid_flair + half_window)]
        fixed_flair_mid  = fixed_image[:, :, (mid_t1 - mid_t1flair_10):(mid_t1 + mid_t1flair_10)]

        # Rigid registration (Assuming you have defined this function elsewhere)
        transform_flair = rigid_registration_euler_moments(fixed_flair_mid, moving_flair_mid)

        # Resample FLAIR
        flair_reg_sitk = sitk.Resample(
            flair_sitk,
            fixed_image,
            transform_flair,
            sitk.sitkLinear,
            0.0,
            flair_sitk.GetPixelID()
        )
        logging.info(f"\tNew FLAIR Size: {flair_reg_sitk.GetSize()}")
        results[3] = flair_reg_sitk
        results[5] = transform_flair
    else:
        logging.warning("\tFLAIR missing -> skipping FLAIR registration.")

    # Return the images (some may be None if missing)
    return results[0], results[1], results[2], results[3], results[4], results[5]

# ---------------------------------------------------------------------
# Process a Single Study
# ---------------------------------------------------------------------
def process_study(subject, study, interim_directory):
    """
    Process a single study for a given subject.
    
    Parameters:
      subject: An object representing the subject. (Should have an 'id' attribute.)
      study: An object representing the study. (Should have a 'date' attribute.)
      interim_directory: Base directory (Path) where the input and output folders reside.
    """
    study_folder = study.date.replace("/", ".")
    input_path = interim_directory / subject.id / study_folder / "01_unprocessed"
    output_path = interim_directory / subject.id / study_folder / "02_registration"
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Processing Subject {subject.id} - Study {study.date}")
    
    # Get the registered images
    t1_image, siref_reg, bm_reg, flair_reg, transform_siref, transform_flair = register_unprocessed2t1(input_path)

    # If T1 is missing, skip this study
    if t1_image is None:
        logging.error(f"Skipping study {study.date} for subject {subject.id} due to missing T1.")
        return

    # Define paths to save the images
    save_t1_path    = output_path / "t1.nii.gz"
    save_siref_path = output_path / "siref_reg.nii.gz"
    save_bm_path    = output_path / "brainmask_reg.nii.gz"
    save_flair_path = output_path / "flair_reg.nii.gz"
    save_siref_transform_path = output_path / "transform_siref.tfm"
    save_flair_transform_path = output_path / "transform_flair.tfm"

    # Write images with error handling
    try:
        sitk.WriteImage(t1_image, str(save_t1_path))
        logging.info("\tSaved T1")
    except Exception as e:
        logging.error(f"\tError saving T1: {e}")

    if siref_reg is not None:
        try:
            sitk.WriteImage(siref_reg, str(save_siref_path))
            logging.info("\tSaved SIREF")
        except Exception as e:
            logging.error(f"\tError saving SIREF: {e}")
    else:
        logging.warning("\tNo SIREF image to write.")

    if bm_reg is not None:
        try:
            sitk.WriteImage(bm_reg, str(save_bm_path))
            logging.info("\tSaved Brain Mask")
        except Exception as e:
            logging.error(f"\tError saving Brain Mask: {e}")
    else:
        logging.warning("\tNo Brain Mask image to write.")

    if flair_reg is not None:
        try:
            sitk.WriteImage(flair_reg, str(save_flair_path))
            logging.info("\tSaved FLAIR")
        except Exception as e:
            logging.error(f"\tError saving FLAIR: {e}")
    else:
        logging.warning("\tNo FLAIR image to write.")

    if transform_siref is not None:
        try:
            sitk.WriteTransform(transform_siref, str(save_siref_transform_path))
            logging.info(f"\tSaved SIREF transform at: {save_siref_transform_path}")
        except Exception as e:
            logging.error(f"\tError saving SIREF Transform: {e}")
    else:
        logging.warning("\tNo SIREF Transform to write.")    

    if transform_flair is not None:
        try:
            sitk.WriteTransform(transform_flair, str(save_flair_transform_path))
            logging.info(f"\tSaved FLAIR transform at: {save_flair_transform_path}")
        except Exception as e:
            logging.error(f"\tError saving FLAIR Transform: {e}")
    else:
        logging.warning("\tNo FLAIR Transform to write.")    

def has_been_registered(output_path):
    """
    Determine if processing has already been completed by checking for a marker file.
    """
    marker_file = output_path / "registration_complete.txt"
    return marker_file.exists()

def mark_as_registered(output_path):
    """
    Create a marker file to indicate that processing is complete.
    """
    marker_file = output_path / "registration_complete.txt"
    marker_file.touch()  # Create an empty file

# ---------------------------------------------------------------------
# Batch Processing Function
# ---------------------------------------------------------------------
def batch_process_registration(project, interim_directory, overwrite=False):
    """
    Process all subjects (and their studies) in a project.
    
    Parameters:
      project: A project object that provides a method all_subject() returning a list of subjects.
      interim_directory: Base directory (Path) where the input and output folders reside.
      overwrite: If True, process all subjects even if they've been processed before.
                 If False, skip subjects (or studies) that already have a processing marker.
    """
    all_subjects = project.all_subject()
    logging.info(f"Found {len(all_subjects)} subjects to process.")

    for subject in all_subjects:
        logging.info(f"\n=== Processing Subject: {subject.id} ===")
        studies = subject.all_study()
        for study in studies:
            # Build the expected output path for this study.
            study_output = interim_directory / subject.id / study.date.replace("/", ".") / "02_registration"
            study_output.mkdir(parents=True, exist_ok=True)
            
            if not overwrite and has_been_registered(study_output):
                logging.info(f"Skipping subject {subject.id}, study {study.date}: already processed.")
                continue

            try:
                process_study(subject, study, interim_directory)
                # If processing succeeds, mark this study as processed.
                mark_as_registered(study_output)
            except Exception as e:
                logging.error(f"Error processing subject {subject.id}, study {study.date}: {e}")
