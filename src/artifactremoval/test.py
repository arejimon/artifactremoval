
import SimpleITK as sitk

from artifactremoval.registration import resample_image, rigid_registration_versor_moments
from artifactremoval.imgproc import itk_to_sitk, sitk_to_itk
from artifactremoval.skullstrip import skull_strip, apply_mask

#--------------------------------------------------------------------------------------------------------------------------------#

def test_convert_midas2nifti(study):
    print("TEST FUNCTION. NOTHING IS SAVED.")
    
    try: 
        # Attempt to get T1 image from the study        
        t1_itk = study.t1()[1]
        t1_sitk = itk_to_sitk(t1_itk)
    except Exception as e: 
        if "does not exist" in str(e):
            print(f"Warning: {e}. Problem with T1.")
            t1_itk = None
        else:
            raise


    try: 
        # FLAIR  
        flair_itk = study.flair()[1]
        flair_sitk = itk_to_sitk(flair_itk)
    except Exception as e: 
        if "does not exist" in str(e):
            print(f"Warning: {e}. Problem with FLAIR.")
            flair_itk = None
            flair_sitk = None
        else:
            raise

    try: 
        # SIREF      
        siref_itk = study.ref()[1]
        siref_sitk = itk_to_sitk(siref_itk)
    except Exception as e: 
        if "does not exist" in str(e):
            print(f"Warning: {e}. Problem with SIREF.")
            siref_itk = None
        else:
            raise

    try: 
        # BRAIN MASK      
        brainmask_itk = study.brain_mask()[1]
        brainmask_sitk = itk_to_sitk(brainmask_itk)
    except Exception as e: 
        if "does not exist" in str(e):
            print(f"Warning: {e}. Problem with Brain Mask.")
            brainmask_itk = None
        else:
            raise        

    return t1_sitk, flair_sitk, siref_sitk, brainmask_sitk

def skullstrip_resample_study_test(interim_preseg_directory):

    print("RUNNING TEST VERSION. NOTHING WILL BE SAVED.")

    print("Will output t1_img, flair_img, full registered flair, mid 10 slices registered flair, and masked mid 10 slices reg flair")

    print("Processing: ", str(interim_preseg_directory))

    t1_path = interim_preseg_directory / "unprocessed_t1.nii.gz"
    flair_path = interim_preseg_directory / "unprocessed_flair.nii.gz"

    # Read Image
    t1_img = sitk.ReadImage(str(t1_path))
    flair_img = sitk.ReadImage(str(flair_path))

    print("RESAMPLING")
    # Resample images to resample size
    t1_img_res = resample_image(t1_img, mask=False, size=[128, 128, 128])
    flair_img_res = resample_image(flair_img, mask=False, size=[128, 128, 128])

    fixed_image = t1_img_res
    moving_image = flair_img_res

    print("FULL REGISTRATION")
    # Register images with skull still on save this transform
    full_reg_transform = rigid_registration_versor_moments(fixed_image, moving_image)
    flair_img_reg_full = sitk.Resample(moving_image, fixed_image, full_reg_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    print("Mid 10 Slices Registration")
    # get mid 10 slices of flair and t1
    num_mid_slices = 10
    mid_flair_10 = num_mid_slices//2
    mid_t1_10_2 = int( num_mid_slices * (fixed_image.GetSpacing()[2] / moving_image.GetSpacing()[2]) ) // 2
    mid_flair = moving_image.GetSize()[2] // 2
    mid_t1_2 = fixed_image.GetSize()[2] // 2

    # Set the middle slice of t1 sitk as the fixed volume
    fixed_middle10_2 = fixed_image[:, :, mid_t1_2-mid_t1_10_2:mid_t1_2+mid_t1_10_2]
    moving_flair_middle10 = moving_image[:, :, mid_flair-mid_flair_10:mid_flair+mid_flair_10]
    fixed_middle10_2.GetSize()
    moving_flair_middle10.GetSize()

    # do full registration between t1_sitk and flair_sitk (no need to do 10 middle slices here)
    mid10_reg_transform = rigid_registration_versor_moments(fixed_middle10_2, moving_flair_middle10)
    flair_img_reg_mid10 = sitk.Resample(moving_image, fixed_image, mid10_reg_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Skullstrip T1, then apply skull mask to T1 and FLAIR
    print("SKULL STRIPPING Version 1")
    skullMask_v1, _ = skull_strip(sitk_to_itk(sitk.Cast(flair_img_reg_mid10, sitk.sitkInt16)))
    t1_img_masked = apply_mask(skullMask_v1, t1_img_res)
    flair_img_masked_reg = apply_mask(skullMask_v1, flair_img_reg_mid10)

    return fixed_image, moving_image, flair_img_reg_full, flair_img_reg_mid10, flair_img_masked_reg 
