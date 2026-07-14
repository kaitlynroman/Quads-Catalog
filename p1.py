"""
Part 1:
    1. Checks to see if there is an existing query data file for the requested cone center
        and radius.
    2. If there is a .parquet file (download and DataLab query completed), it returns
        that file for use.
    3. If there is no .parquet file:
        4. Checks to see if there is a .csv file of the same name (indicating that a previous
            query completed but was not downloaded), and downloads that file without requerying.
        5. If there is not, it queries DES DR2 Y6 Gold for coadd_object_id, ra, dec,
            psf_mag_aper_8_g_corrected, psf_mag_aper_8_r_corrected,
            psf_mag_aper_8_i_corrected, psf_mag_aper_8_z_corrected, psf_mag_aper_8_y_corrected,
            ext_mash, bdf_t, bdf_g_1, bdf_g_2, psf_t, psf_g_1, and psf_g_2 and downloads the
            data, so it does not need to be queried again.

Creates .csv files (later deleted) and a .parquet file to be converted to
a pandas DataFrame for P2, which organizes and scores detected foursomes.
"""

# =============================================================================
# LIBRARY IMPORTING, DEFAULT VALUES, AND HELPER FUNCTIONS
# =============================================================================

import os, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from dl import authClient as ac
from dl import storeClient as sc
from dl import queryClient as qc

MAG_COLS = [f"psf_mag_aper_8_{b}_corrected" for b in "grizy"] # Magntiude column names as list
VALID = {"QUEUED", "EXECUTING", "COMPLETED", "PENDING", "HELD", "SUSPENDED"} # Valid statuses for jobid

def j2000_to_degrees(ra_lens_str, dec_lens_str):
    """
    Converts the central lens position from J2000 to degrees.
    Parameters:
        ra_lens_str (str): The central lens right of ascension in HMS
        dec_lens_str (str): The central lens declination in DMS
    Returns the coordinates of the central lens in degrees.
    """
    c = SkyCoord(ra_lens_str, dec_lens_str, unit=(u.hourangle, u.deg), frame='icrs')
    return c.ra.deg, c.dec.deg

def tile_pixels(ra_region, dec_region, max_distance, tile_nside):
    """
    Finds the nside=32 HEALPix cells (NESTED) covering the search cone.
    Parameters:
        ra_region (float): The cone center RA in degrees
        dec_region (float): The cone center Dec in degrees
        max_distance (float): The search radius of the cone in degrees
        tile_nside (int): The nside at which to tile
    Returns a sorted list of unique hpix_32 pixel numbers to query.
    """
    # Cells whose centers/areas fall in the cone (inclusive catches edge cells)
    vec = hp.ang2vec(ra_region, dec_region, lonlat=True)
    core = hp.query_disc(tile_nside, vec, np.radians(max_distance),
                         inclusive=True, nest=True)
    return sorted(set(core.tolist()))

def vospace_has(name):
    """
    Determines whether a file is listed in the user's VOSpace root.
    Parameters:
        name (str): The name of the filepath to locate
    Returns a bool based on whether it finds the filepath in VOSpace.
    """
    # Checks whether there is a file in VOSpace
    try:
        return name in [x.strip() for x in sc.ls().split(",")]
    except Exception as e:
        print(f"  Could not list VOSpace ({e}). Assuming absent...", flush=True)
        return False

def submit_and_wait(sql, table_name):
    """
    Submits the async query and blocks errors until it completes.
    Parameters:
        sql (str): The SQL or ADQL query to submit to DataLab
        table_name (str): The name of the table to store data in
    Puts the data in VOSpace and returns when completed.
    """
    # Submits the query and prints the jobid for identification
    print("Submitting query...", flush=True)
    jobid = qc.query(sql=sql, out=f"vos://{table_name}.csv",
                    async_=True, drop=True)
    print("jobid:", jobid, flush=True)

    # Checks status until the job is complete
    fails = 0
    notfound = 0
    while True:
        # Gets the status of the job
        try:
            status = qc.status(jobid)
            fails = 0
        # Catches exception and retries checking the status again
        except Exception as e:
            fails += 1
            if fails > 40:
                raise RuntimeError(f"status unreachable {fails}x; last: {e}")
            print(f"  Trying to regain connection x{fails}: {e}", flush=True)
            time.sleep(15)
            continue

        # Strips the status for analysis
        s = str(status).strip()

        # If the job is complete alert the user and end the function
        if s == "COMPLETED":
            print("COMPLETED!")
            return
        # If there is an error or the job is aborted raise an error
        if s in ("ERROR", "ABORTED"):
            raise RuntimeError(qc.error(jobid))

        # qc.status can return an error as a normal string and it is caught here
        if s not in VALID:
            if "not found" in s.lower():
                notfound += 1
                print(f"  Job not found ({notfound}/3)", flush=True)
                if notfound >= 3:
                    print("VOSpace contents:", sc.ls(), flush=True)
                    raise RuntimeError(
                        f"The job {jobid} no longer exists. If the output file is "
                        f"listed above, re-run and the VOSpace check will skip the query."
                    )
            else:
                fails += 1
                print(f"  Bad status text ({fails}): {s[:120]}", flush=True)
                if fails > 40:
                    raise RuntimeError(f"Status unusable {fails}x")
            time.sleep(10)
            continue

        # Good status clears the notfound value and is printed
        notfound = 0
        print(f"  STATUS: {s} AT {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(10)

def download_with_retry(table_name, local_csv, attempts=10, wait=30):
    """
    Retries downloading the file if a failure occurs and verifies a non-empty file landed.
    Parameters:
        table_name (str): The name of the table to download
        local_csv (str): The name of the .csv to convert the table to and check contents
        attempts (int): The amount of times to attempt downloading (default=10)
        wait (int): The amount of time in between attempts to wait (default=30 seconds)
    Downloads the .csv file to VOSpace (kept if query completes but .parquet download fails).
    """
    # Tries storing the .csv file to VOSpace
    for attempt in range(1, attempts + 1):
        try:
            sc.get(fr=f"vos://{table_name}.csv", to=local_csv)
            # If it suceeds, the function returns
            if os.path.exists(local_csv) and os.path.getsize(local_csv) > 0:
                return
            raise RuntimeError("sc.get produced no file")
        # If the attempt fails, it tries again unless it hits the max allowed failures
        except Exception as e:
            print(f"  download attempt {attempt}/{attempts} failed: "
                  f"{type(e).__name__}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(wait)

def clean_magnitudes(df, min_mag, max_mag):
    """
    Marks out-of-range magnitude readings (<min_mag or >max_mag) as NaN.
    Parameters:
        df (DataFrame): The raw queried data
        min_mag (int/float): The minimum value for an object magnitude to not be marked as NaN
        max_mag (int/float): The minimum value for an object magnitude to not be marked as NaN
    Returns the DataFrame with bad magnitudes replaced by NaN.
    """
    # Checks to make sure file is complete with all magnitude columns
    missing = [c for c in MAG_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"missing {missing}; got {list(df.columns)}")

    # Removes bad magnitude readings (<min_mag and >max_mag)
    for c in MAG_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[(df[c] <= min_mag) | (df[c] >= max_mag), c] = np.nan
    return df

# =============================================================================
# RUNNING QUERIES
# =============================================================================

def query_batch(pixels, table_name, tile_nside, min_mag, max_mag):
    """
    Queries one batch of hpix_32 cells, downloads it, and returns a cleaned DataFrame.
    Parameters:
        pixels (list): The hpix_32 (NESTED) pixel numbers in this batch
        table_name (str): The name of the table/.csv for this batch
        tile_nside (int): The nside at which to tile (for SQL Query)
        min_mag (int/float): The minimum value for an object magnitude to not be marked as NaN
        max_mag (int/float): The minimum value for an object magnitude to not be marked as NaN
    Returns a pandas DataFrame of the batch, or a cached one if it already exists
    """
    parquet_path = f"{table_name}.parquet"
    local_csv = f"{table_name}.csv"

    # If already downloaded and cleaned, it loads the parquet data as a pandas DataFrame
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df):,} objects from {parquet_path} (cached).")
        return df

    # SQL Query for DES DR2 Y6 Gold, selecting whole hpix_32 cells
    pix_list = ", ".join(str(p) for p in pixels)
    sql = f"""
        SELECT coadd_object_id, ra, dec,
               psf_mag_aper_8_g_corrected, psf_mag_aper_8_r_corrected,
               psf_mag_aper_8_i_corrected, psf_mag_aper_8_z_corrected,
               psf_mag_aper_8_y_corrected,
               ext_mash, bdf_t, bdf_g_1, bdf_g_2, psf_t, psf_g_1, psf_g_2
        FROM des_dr2.y6_gold
        WHERE hpix_{str(tile_nside)} IN ({pix_list})
    """

    # If the query already ran but the download failed, it does not requery
    if vospace_has(f"{table_name}.csv"):
        print(f"{table_name}.csv already in VOSpace - skipping query.")

    # If the query did not run, submit it
    else:
        submit_and_wait(sql, table_name)

    # Downloads the file from VOSpace
    print("Downloading from VOSpace...", flush=True)
    download_with_retry(table_name, local_csv)
    print(f"Downloaded {os.path.getsize(local_csv)/1e9:.2f} GB", flush=True)

    # Reads the csv for magnitude alterations
    df = pd.read_csv(local_csv)
    print(f"Got {len(df):,} objects.")

    # Removes bad magnitude readings (<MIN_MAG and >MAX_MAG)
    print("Removing bad magnitude readings...", flush=True)
    df = clean_magnitudes(df, min_mag, max_mag)

    # Writes the .parquet file and stores it
    df.to_parquet(parquet_path)
    print(f"Cached to {parquet_path}.")

    # Deletes the local_csv (not needed anymore) and returns the pandas DataFrame
    if os.path.exists(local_csv):
        os.remove(local_csv)
    return df

def query_region(ra_region, dec_region, max_distance, table_name, min_mag, max_mag, tile_nside, batch_size):
    """
    Returns a pandas DataFrame of the region by querying hpix_32 tiles in batches
    of BATCH_SIZE, skipping any batch whose parquet or VOSpace csv already exists.
    Parameters:
        ra_region (float): The cone center RA in degrees
        dec_region (float): The cone center Dec in degrees
        max_distance (float): The search radius of the cone
        table_name (str): The base name for the region's .parquet/.csv files
        min_mag (int/float): The minimum value for an object magnitude to not be marked as NaN
        max_mag (int/float): The minimum value for an object magnitude to not be marked as NaN
        tile_nside (int): The nside at which to tile
        batch_size (int): The amount of tile_nside pixels to query per batch
    Returns a pandas DataFrame of the region data for P2
    """
    # Names the table and identifies what the parquet_path would be
    parquet_path = f"{table_name}.parquet"

    # If the whole region was already assembled and cleaned, load it and return
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df):,} objects from {parquet_path} (cached).")
        return df

    # Finds the hpix_32 tiles (plus neighbor buffer) covering the search cone
    pixels = tile_pixels(ra_region, dec_region, max_distance, tile_nside)
    print(f"{len(pixels)} hpix_32 tiles to query "
          f"in {int(np.ceil(len(pixels)/batch_size))} batches of {batch_size}.", flush=True)

    # Queries the tiles BATCH_SIZE at a time, caching each batch for resumability
    parts = []
    batch_names = []
    for i in range(0, len(pixels), batch_size):
        batch = pixels[i:i + batch_size]
        # For the last batch, prevent IndexError
        if i + batch_size < len(pixels):
            batch_name = f"{table_name}_{pixels[i]}_{pixels[i + batch_size]}"
            batch_names.append(batch_name)
        # For the all other batches, label with pixel numbers
        else:
            batch_name = f"{table_name}_{batch[0]}_{batch[-1]}"
            batch_names.append(batch_name)
        try:
            parts.append(query_batch(batch, batch_name, tile_nside, min_mag, max_mag))
        except Exception as e:
            print(f"  Batch {i//batch_size} failed: {e} so we will skip and rerun later", flush=True)
            continue

    # Concatenates the batches and writes the assembled region .parquet
    df = pd.concat(parts, ignore_index=True)
    print(f"\nCombined {len(df):,} objects across {len(parts)} batches.", flush=True)
    df.to_parquet(parquet_path)

    # Deletes unmerged parquet files to save memory
    for name in batch_names:
        if os.path.exists(f"{name}.parquet"):
            os.remove(f"{name}.parquet")

    # Returns the dataframe for use
    print(f"Cached region to {parquet_path}.")
    return df

# =============================================================================
# MAIN
# =============================================================================

def main(params):
    # Check to make sure the user is logged in (necessary for async queries)
    print(f"Logged into the NOIRLab as: {ac.whoAmI()}\n", flush=True)

    # Parameter values from final.py
    ra = params["ra_center"]
    dec = params["dec_center"]
    max_distance = params["radius_deg"]
    data_file_name = params["data_file"]
    min_mag = params["min_mag"]
    max_mag = params["max_mag"]
    tile_nside = params["tile_nside"]
    batch_size = params["batch_size"]

    # Queries the desired region for data
    ra_deg, dec_deg = j2000_to_degrees(ra, dec)
    print(f"CONE WITH CENTER ({ra_deg:.3f}, {dec_deg:.3f}) AND RADIUS {max_distance}\n", flush=True)
    query_region(ra_deg, dec_deg, max_distance, data_file_name, min_mag, max_mag, tile_nside, batch_size)

if __name__ == "__main__":
    main()
