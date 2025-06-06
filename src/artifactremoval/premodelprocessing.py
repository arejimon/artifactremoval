import pickle
import pandas as pd
import random
import logging
from statistics import mode, StatisticsError

def aggregate_ratings(group):
    """
    Given a group of ratings (from the same spectrum), aggregates them
    into three separate rating columns, a list of rater names, and a consensus vote.
    """
    # Convert the ratings to a list.
    ratings = group['rating'].tolist()
    raters = group['name'].tolist()

    # Compute the consensus vote using the mode.
    try:
        consensus = mode(ratings)
        logging.debug(f"Mode for ratings {ratings} computed as: {consensus}")
    except StatisticsError:
        consensus = "No Consensus"
        logging.warning(f"Ratings {ratings} did not have a unique mode; setting consensus to 'No Consensus'")
    
    # Ensure three ratings (fill with None if missing).
    rating_1 = ratings[0] if len(ratings) > 0 else None
    rating_2 = ratings[1] if len(ratings) > 1 else None
    rating_3 = ratings[2] if len(ratings) > 2 else None

    return pd.Series({
        'rating_1': rating_1,
        'rating_2': rating_2,
        'rating_3': rating_3,
        'consensus': consensus,
        'raters': raters
    })

def verify_aggregates(aggregated_df, all_ratings_df):
    """
    For each unique_id in the aggregated DataFrame, extracts the corresponding
    rows from the original concatenated CSV data and checks whether the aggregated
    ratings (including consensus) match the expected values.
    Returns a list of inconsistencies.
    """
    inconsistencies = []
    logging.info("Starting verification of aggregated ratings.")

    for _, agg_row in aggregated_df.iterrows():
        unique_id = agg_row['unique_id']
        group = all_ratings_df[all_ratings_df['unique_id'] == unique_id]
        ratings_list = group['rating'].tolist()
        
        expected_rating_1 = ratings_list[0] if len(ratings_list) > 0 else None
        expected_rating_2 = ratings_list[1] if len(ratings_list) > 1 else None
        expected_rating_3 = ratings_list[2] if len(ratings_list) > 2 else None
        
        try:
            expected_consensus = mode(ratings_list)
        except StatisticsError:
            expected_consensus = "No Consensus"
        
        # Compare expected values with the aggregated values.
        if (agg_row['rating_1'] != expected_rating_1 or 
            agg_row['rating_2'] != expected_rating_2 or 
            agg_row['rating_3'] != expected_rating_3 or 
            agg_row['consensus'] != expected_consensus):
            
            inconsistency = {
                'unique_id': unique_id,
                'expected': {
                    'rating_1': expected_rating_1,
                    'rating_2': expected_rating_2,
                    'rating_3': expected_rating_3,
                    'consensus': expected_consensus
                },
                'actual': {
                    'rating_1': agg_row['rating_1'],
                    'rating_2': agg_row['rating_2'],
                    'rating_3': agg_row['rating_3'],
                    'consensus': agg_row['consensus']
                }
            }
            inconsistencies.append(inconsistency)
            logging.error(f"Inconsistency found for unique_id {unique_id}: {inconsistency}")
    
    if not inconsistencies:
        logging.info("Verification succeeded: All aggregated ratings match the original CSV data.")
    return inconsistencies

def load_spectral_data(spectral_file):
    """Loads the spectral data from a pickle file."""
    logging.info(f"Loading spectral data from pickle file: {spectral_file}")
    with open(spectral_file, 'rb') as f:
        spectral_data = pickle.load(f)
    logging.info("Spectral data loaded successfully.")
    return spectral_data

def update_spectral_data_with_consensus(spectral_data, consensus_mapping):
    """
    For each spectral entry, adds a 'consensus_rating' field from the consensus mapping.
    """
    logging.info("Updating spectral data with consensus ratings.")
    for entry in spectral_data:
        uid = entry.get("unique_id")
        if uid in consensus_mapping:
            entry["consensus_rating"] = consensus_mapping[uid]
            logging.debug(f"Unique ID {uid} updated with consensus rating: {consensus_mapping[uid]}")
        else:
            entry["consensus_rating"] = None
            logging.warning(f"No consensus rating found for Unique ID {uid}; set to None.")
    logging.info("Spectral data updated with consensus ratings.")
    return spectral_data

def display_random_consensus_entries(spectral_data, aggregated_df, sample_size=10):
    """
    Displays a random sample of entries (with consensus ratings) from spectral data alongside
    the corresponding rows from the aggregated CSV for manual verification.
    """
    spectral_data_with_consensus = [entry for entry in spectral_data if entry.get("consensus_rating") is not None]
    if len(spectral_data_with_consensus) < sample_size:
        sample_entries = spectral_data_with_consensus
    else:
        sample_entries = random.sample(spectral_data_with_consensus, sample_size)
    
    logging.info(f"Displaying {len(sample_entries)} random spectral entries for verification.")
    
    for entry in sample_entries:
        uid = entry.get("unique_id")
        consensus_rating = entry.get("consensus_rating")
        
        # Extract the corresponding row from the aggregated DataFrame.
        agg_row = aggregated_df[aggregated_df['unique_id'] == uid]
        logging.info(f"Unique ID: {uid}")
        logging.info(f"Spectral Data Consensus Rating: {consensus_rating}")
        logging.info("Aggregated CSV Row:")
        logging.info(agg_row.to_string(index=False))
        logging.info("-" * 50)
        
        

def extract_subject_id(uid: str) -> str:
    """
    Extract the full patient / project ID from a unique_id that follows
    the pattern:  patientid_date_x_y_z

    The last four underscore-separated tokens are   date, x, y, z.
    Everything to the **left** of those belongs to patientid, even if it
    contains underscores.
    """
    # Split from the RIGHT, keeping at most 4 splits
    # e.g.  'DOSEESC_UM03_20240101_12_30_40'
    #   -> ['DOSEESC_UM03', '20240101', '12', '30', '40']
    parts = uid.rsplit('_', 4)

    if len(parts) != 5:
        raise ValueError(f"unique_id not in expected format: {uid}")

    patientid = parts[0]          # everything before the final 4 fields
    return patientid


def extract_subject_id(uid: str) -> str:
    """
    Extract the full patient / project ID from a unique_id that follows
    the pattern:  patientid_date_x_y_z

    The last four underscore-separated tokens are   date, x, y, z.
    Everything to the **left** of those belongs to patientid, even if it
    contains underscores.
    """
    # Split from the RIGHT, keeping at most 4 splits
    # e.g.  'DOSEESC_UM03_20240101_12_30_40'
    #   -> ['DOSEESC_UM03', '20240101', '12', '30', '40']
    parts = uid.rsplit('_', 4)

    if len(parts) != 5:
        raise ValueError(f"unique_id not in expected format: {uid}")

    patientid = parts[0]          # everything before the final 4 fields
    return patientid

def drop_missing(records, required=("consensus", "raw_spectrum")):
    """
    Return a new list that keeps only entries where every `required`
    key exists and is not None.
    """
    clean  = [r for r in records
              if all(k in r and r[k] is not None for k in required)]
    removed = len(records) - len(clean)
    if removed:
        logging.info(f"Filtered out {removed} / {len(records)} records "
                     f"missing any of {required}.")
    return clean