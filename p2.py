"""
Part 2:
    1. Flag dense coarse pixels (>= DENSE_MIN objects at DENSE_NSIDE) and their
        neighbors (likely star clusters/rich groups) and return a veto set.
    2. Pixelate/group objects at (nside, origin_step), keeping cells with 4 to 7
        objects and dropping any cell whose center falls in a vetoed coarse pixel,
        then write the FORTRAN input cell file.
    3. CALLS FORTRAN: form all foursomes per cell and score them (positional rms
        and magnitude frms) for the desired quad model (4img or 3img_lens).
    4. Parse the FoM output, cutting foursomes with fom_rms > FOM_RMS_CUT or
        fom_mag > FOM_MAG_CUT.
    5. Re-emit the survivors and sort by positional rms.

Creates .txt files for FORTRAN_input and FORTRAN_output for analysis in P3.
"""

# =============================================================================
# LIBRARY IMPORTING, DEFAULT VALUES, AND HELPER FUNCTIONS
# =============================================================================

import os
import shutil
import subprocess
import pathlib
import numpy as np
import pandas as pd
import healpy as hp

def radec_to_healpix(ra, dec, nside):
    """
    Converts the ra and dec values to healpix pixel numbers.
    Parameters:
        ra (float): The right of ascension in degrees of an image
        dec (float): The declination in degrees of an image
        nside (int): The resolution (the number of divisions along the side of a pixel)
    Returns the healpix pixel number of a coordinate.
    """
    return hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)

def fmt(x):
    """
    Format values as strings for the FORTRAN input file.
    Parameters:
        x (array-like): The pandas DataFrame column to be formatted
    Returns the column with string values.
    """
    return f"nan" if pd.isna(x) else f"{float(x):.10f}"

# =============================================================================
# PYTHON: PIXELATING/GROUPING AND FORTRAN INPUT FILE CREATION
# =============================================================================

def find_dense_pixels(df, dense_nside, dense_min):
    """
    Finds cells at dense_nside holding an unusually large number of objects (liekly a cluster),
    then expands each by its 8 immediate neighbors so the whole cluster is vetoed.
    Parameters:
        df (pandas DataFrame): The raw queried data from p1.py
        dense_nside (int): The nside dense pixels are classified at
        dense_min (int): The minimum object count for a cell to be dense
    Returns a set of nested dense_nside pixel numbers to veto.
    """
    # Converts ra and dec columns to array-like objects
    print(f"Finding dense pixels (>= {dense_min} objects at nside={dense_nside})...")
    ra = pd.to_numeric(df["ra"],  errors="coerce").to_numpy(np.float64)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(np.float64)

    # Count objects per coarse pixel
    pix = hp.ang2pix(dense_nside, ra, dec, lonlat=True, nest=True)
    vals, counts = np.unique(pix, return_counts=True)
    dense = vals[counts >= dense_min]

    # Expand each dense pixel by its 8 neighbors so the cluster's full extent is vetoed
    veto = set(int(p) for p in dense)
    for p in dense:
        for nb in hp.get_all_neighbours(dense_nside, int(p), nest=True):
            if nb >= 0:
                veto.add(int(nb))

    print(f"Found {len(dense):,} dense pixels and vetoed {len(veto):,} total pixels "
          f"(with neighbors).\n")
    return veto

def pixelate_and_group(df, nside, origin_step, low_cut, high_cut, dense_nside,
                       veto_pixels):
    """
    Pixelates and groups objects with the same HEALPix pixel number and cuts groups with
    < low_cut or > high_cut objects.
    Parameters:
        df (pandas DataFrame): The raw queried data to be grouped/rejected
        nside (int): The resolution (the number of divisions along the side of a pixel)
        origin_step (tuple): The step factor (d_ra_frac, d_dec_frac) at which the pixel origin is shifted
        low_cut (int): The minimum number of objects contained in the step for the
        system to move on to testing
        high_cut (int): The maximum number of objects contained in the step for the system
        to move on to testing
        dense_nside (int): The nside dense pixels are classified at
        veto_pixels (set): dense pixels to reject from find_dense_pixels
    Returns a dataframe with the group_ids, HEALPix number, and corresponding object_ids.
    """
    print("Pixelating and grouping...")
    d_ra_frac, d_dec_frac = origin_step

    # Pixel angular size at this nside in degrees, for the origin shift
    pix_deg = np.degrees(hp.nside2resol(nside))

    # Converts ra and dec columns to array-like objects
    ra = pd.to_numeric(df["ra"],  errors="coerce").to_numpy(np.float64)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(np.float64)

    # Shift the grid by shifting the objects the opposite way, with cos(dec) on RA
    dec_shifted = dec + d_dec_frac * pix_deg
    ra_shifted  = ra  + d_ra_frac  * pix_deg / np.cos(np.radians(dec))

    # Assign each object its nested HEALPix pixel under the given nside and origin step
    df = df.copy()
    df["pix"] = hp.ang2pix(nside, ra_shifted, dec_shifted, lonlat=True, nest=True).astype(np.int64)

    # Get cell centers, computed once per object from its pixel
    cra, cdec = hp.pix2ang(nside, df["pix"].to_numpy(), nest=True, lonlat=True)
    df["racell"] = cra
    df["deccell"] = cdec

    # Drop objects whose cell center falls in a vetoed coarse pixel
    if veto_pixels:
        cell_coarse = hp.ang2pix(dense_nside, df["racell"].to_numpy(),
                                 df["deccell"].to_numpy(), lonlat=True, nest=True)
        in_veto = np.isin(cell_coarse, np.fromiter(veto_pixels, dtype=np.int64))
        n_before = len(df)
        df = df[~in_veto].copy()
        print(f"Dropped {n_before - len(df):,} objects in "
              f"{len(veto_pixels):,} vetoed pixels.")

    # Keep only objects in bare cells with low_cut to high_cut members for FORTRAN combinatorics
    counts = df.groupby("pix")["pix"].transform("size")
    grouped = df[(counts >= low_cut) & (counts <= high_cut)].copy()

    print(f"For nside={nside}, origin_step={origin_step}, kept {grouped['pix'].nunique():,} cells with {len(grouped):,} objects.\n")
    return grouped

# =============================================================================
# FORTRAN: COMBINATIONS AND SCORING
# =============================================================================

def to_cellfile(combined, input_path, folder):
    """
    Converts the pandas DataFrame from pixelate_and_group() to a txt file in a format for the FORTRAN code to parse.
    Parameters:
        combined (pandas DataFrame): The DataFrame returned by pixelate_and_group()
        input_path (str): The FORTRAN input file name
        folder (str): The name of the folder the FORTRAN files are located in
    Creates the input file for FORTRAN use.
    """
    # Opens the input file to be created
    target = pathlib.Path.cwd() / folder / input_path
    with open(target, "w") as f:
        for pix, group in combined.groupby("pix"):
           # For each distinct cell, get the number of objects, HEALPix number, and HEALPix center in degrees
            n = len(group)
            pix = int(group["pix"].iloc[0])
            racell = float(group["racell"].iloc[0])
            deccell = float(group["deccell"].iloc[0])

            f.write("\n")
            f.write(f"cell   {pix:d} {n:d} ra {racell:.10f} dec {deccell:.10f}\n")

            for _, r in group.iterrows():
                # Writes the data for each line
                f.write(" ".join([
                    f"{int(r['coadd_object_id'])}",
                    f"{r['ra']:.10f}", f"{r['dec']:.10f}",
                    fmt(r['psf_mag_aper_8_g_corrected']), fmt(r['psf_mag_aper_8_r_corrected']),
                    fmt(r['psf_mag_aper_8_i_corrected']), fmt(r['psf_mag_aper_8_z_corrected']),
                    fmt(r['psf_mag_aper_8_y_corrected']),
                    fmt(int(r['ext_mash'])),
                    fmt(r['bdf_t']), fmt(r['bdf_g_1']), fmt(r['bdf_g_2']),
                    fmt(r['psf_t']), fmt(r['psf_g_1']), fmt(r['psf_g_2']),
                    f"{int(pix)}",
                ]) + "\n")

    print(f"{input_path} successfully created!\n")
    return input_path

def run_siepmod(input_file,
                folder,
                fom_file,
                param_file,
                loser_file,
                resid_file,
                exe):
    """
    Runs the compiled FORTRAN scorer by feeding the 5 prompt answers via the input cell file
    and four output filenames.
    Parameters:
        input_file (str): The input cell file from to_cellfile()
        folder (str): The name of the folder the FORTRAN files are located in
        fom_file (str): The figure of merit output file
        param_file (str): The parameter output file
        loser_file (str): The loser output file
        resid_file (str): The residual output file
        exe (str): The executable file
    Returns the path to the FoM output file.
    """
    # Gets the folder the FORTRAN files are located in
    target_folder = pathlib.Path.cwd() / folder

    # Input files must live in the same folder too.
    answers = "\n".join([input_file, fom_file, param_file,
                         loser_file, resid_file]) + "\n"

    # Runs SIEPmod
    result = subprocess.run(
        [str(target_folder / exe)],
        input=answers,
        capture_output=True,
        text=True,
        cwd=str(target_folder),
    )

    # Checks the FORTRAN output to see if there was an error
    if result.returncode != 0:
        print("FORTRAN stderr:", result.stderr)
        raise RuntimeError(f"siepmod exited with code {result.returncode}")
    print(f"{exe} ran successfully!\n")

    return fom_file

def parse_fom(fom_file, folder, fom_rms_cut, fom_mag_cut):
    """
    Parses run_siepmod's fom_file and keeps rms_pos <= FOM_RMS_CUT and rms_mag <= FOM_MAG_CUT.
    Parameters:
        fom_file (str): The FoM output filename from run_siepmod
        folder (str): The name of the folder the FORTRAN files are located in
        fom_rms_cut (float): The minimum fom_rms for a foursome to be kept as a candidate
        fom_mag_cut (float): The minimum fom_mag for a foursome to be kept as a candidate
    Returns a pandas DataFrame with one row per scored quartet.
    """
    # Gets the path to the fom_file for analysis
    cwd = pathlib.Path.cwd()
    target_folder = cwd / folder
    fom_path = target_folder / fom_file

    # Reads the fom_file
    rows = []
    with open(fom_path) as f:
        for line in f:
            t = line.split()

            # A real record has exactly 25 fields
            if len(t) != 25:
                continue

            # Gets the necessary info from the record
            try:
                pixel   = int(t[0])
                racell  = float(t[1])
                deccell = float(t[2])
                ncat    = int(t[3])
                inc     = tuple(int(x) for x in t[4:8])
                catids  = [int(float(x)) for x in t[8:12]]
                p6, p7  = float(t[12]), float(t[13])
                semi    = float(t[14])
                angle   = float(t[15])
                qp      = float(t[16])
                p1, p2  = float(t[17]), float(t[18])
                hyp     = [float(t[j]) for j in range(19, 23)]
                rms_pos = float(t[23])
                rms_mag = float(t[24])
            except (ValueError, IndexError):
                continue

            # FoM RMS and MAG cuts for a foursome to be considered a good candidate
            if not ((0.0 < rms_pos <= fom_rms_cut) and (0.0 < rms_mag <= fom_mag_cut)):
                continue

            # Collects candidate data into a list of dicts
            rows.append({
                "pixel": pixel,
                "racell": racell,
                "deccell": deccell,
                "ncat": ncat,
                "inc": inc,
                "member_ids": frozenset(catids),
                "p6": p6, "p7": p7, "semi": semi, "angle": angle, "qp": qp,
                "p1": p1, "p2": p2,
                "hyp1": hyp[0], "hyp2": hyp[1], "hyp3": hyp[2], "hyp4": hyp[3],
                "rms_pos": rms_pos,
                "rms_mag": rms_mag,
            })

    # Converts data to pandas and sorts candidates by best rms_pos ascending
    df = pd.DataFrame(rows)
    print("Successfully parsed FoM output!\n")
    return df.sort_values("rms_pos").reset_index(drop=True) if len(df) else df

def test_foursomes(combined, input_path_name, folder, fom_file, param_file, loser_file,
                  resid_file, exe, fom_rms_cut, fom_mag_cut):
    """
    Integrates to_cellfile, run_siepmod, and parse_fom into one function.
    Parameters:
        combined (pandas DataFrame): the grouped output of pixelate_and_group()
        input_path_name (str): The name of the FORTRAN input file
        folder (str): The name of the folder the FORTRAN files are located in
        fom_file (str): The name of the FORTRAN fom file
        param_file (str): The name of the FORTRAN param file
        loser_file (str): The name of the FORTRAN loser file
        resid_file (str): The name of the FORTRAN resid file
        exe (str): The name of the SIEPmod executable
        fom_rms_cut (float): The minimum fom_rms for a foursome to be kept as a candidate
        fom_mag_cut (float): The minimum fom_mag for a foursome to be kept as a candidate
    Returns (fom_df, cell_path) so results can be prepared to be written to the FORTRAN output file.
    """
    print(f"Creating {input_path_name}...")
    input_path = to_cellfile(combined, input_path_name, folder)

    print("Running SIEPmod...")
    fom = run_siepmod(input_path, folder, fom_file, param_file, loser_file,
                         resid_file, exe)

    print("Parsing FoM output...")
    return parse_fom(fom, folder, fom_rms_cut, fom_mag_cut), input_path

# =============================================================================
# PYTHON: CLUSTER VETO AND OUTPUT FILE CREATION
# =============================================================================

def build_cell_index(input_file):
    """
    From the cell file, map each object id to its (pixel, racell, deccell) and record
    the ordered member-id list per pixel, so a foursome's pixel/center/inc can be
    rebuilt from its member ids.
    Parameters:
        input_file (str): The name of the FORTRAN input file
    Returns a tuple of dicts mapping object ID to pix, racell, deccell and pix to object ID
    """
    id_info = {}
    pix_members = {}
    pix, racell, deccell = None, None, None

    # Opens the input file to build the cell index
    with open(input_file) as f:
        for line in f:
            t = line.split()

            # Skips empty space/invalid components
            if not t:
                continue

            # Gets the "cell" line information
            if t[0] == "cell":
                pix     = int(t[1])
                racell  = float(t[4])
                deccell = float(t[6])
                pix_members[pix] = []
                continue

            # Gets the info for id_info and pix_members for each object ID
            try:
                oid = int(float(t[0]))
            except (IndexError, ValueError):
                continue
            id_info[oid] = (pix, racell, deccell)
            pix_members[pix].append(oid)

    return id_info, pix_members

def write_output(fom_df, id_info, pix_members, out_path):
    """
    Re-emits surviving candidtaes in the layout P3 takes, rebuilding data from
    the FORTRAN input file and output file.
    Parameters:
        fom_df (pandas DataFrame): The DataFrame returned by test_foursomes()
        id_info (dict): The dict mapping object_id to (pix, racell, deccell) from build_cell_index()
        pix_members (dict): The dict mapping pix to ordered member ids from build_cell_index()
        out_path (str): The name of the FORTRAN output filen
    Returns the path to the FORTRAN output file.
    """
    # The column names from the fom_file to be added to the FORTRAN output file
    pcols = ["p6","p7","semi","angle","qp","p1","p2","hyp1","hyp2","hyp3","hyp4"]
    n = 0

    # Writes the output file using data from the FORTRAN input and output files
    with open(out_path, "w") as f:
        for _, row in fom_df.iterrows():

            # For each row, finds the first available value in the dict for each id
            ids = sorted(row["member_ids"])
            rec = next((id_info[i] for i in ids if i in id_info), None)

            # If no data is found, skip it
            if rec is None:
                continue

            # Gets the pix, racell, deccell, ncat, and member data for each object
            pix, racell, deccell = rec
            members = pix_members.get(pix, [])
            ncat = len(members)

            # Gets the index ordering (inc) for each object
            try:
                inc = [members.index(i) + 1 for i in ids]
            except ValueError:
                continue

            # Writes the data as strings for the file and increments amount of objects written to the file
            idstr = " ".join(f"{i}.0".ljust(19) for i in ids)
            incstr = " ".join(str(x) for x in inc)
            pstr = " ".join(
                f"{row[c]:.3E}" if (row[c] != 0 and (abs(row[c]) >= 1e3 or abs(row[c]) < 1e-2))
                else f"{row[c]:7.3f}" for c in pcols
            )
            f.write(f"{pix:11d} {racell:14.10f} {deccell:15.10f} {ncat:d}  "
                    f"{incstr} {idstr} {pstr} {row['rms_pos']:7.3f} {row['rms_mag']:7.3f}\n")
            n += 1

    print(f"Wrote {n:,} foursomes to {out_path}!")
    return out_path

def sort_and_write(fom_df, input_path, output_path_name, folder):
    """
    Integrates build_cell_index() and write_output() and moves files into the correct directory
    for p3.py.
    Parameters:
        fom_df (pandas DataFrame): The DataFrame returned by test_foursomes()
        input_path (str): The name of the FORTRAN input file
        output_path_name (str): The name of the FORTRAN output file
        folder (str): The name of the folder the FORTRAN output file is moved out from
    Writes the FORTRAN output file after it is sorted by fom_rms.
    """
    # Rebuilds the id (pix, racell, deccell) and pix to members maps from the cell file
    id_info, pix_members = build_cell_index(pathlib.Path.cwd() / folder / input_path)

    # Moves the FORTRAN input file out of the FORTRAN folder for p3.py
    src = pathlib.Path.cwd() / folder / input_path
    dst = pathlib.Path.cwd() / input_path
    shutil.move(src, dst)

    # If no foursomes survived the cuts, lets the user know
    if len(fom_df) == 0:
        print("No foursomes survived the validity cuts.")
        return

    # Sort by positional FoM and re-emit in the correct layout
    print("Sorting by Positional Figure of Merit...")
    fom_df = fom_df.sort_values("rms_pos").reset_index(drop=True)
    write_output(fom_df, id_info, pix_members, output_path_name)

# =============================================================================
# MAIN
# =============================================================================

def main(params):
    # Parameter values from final.py
    nside = params["nside"]
    origin_step = params["origin_step"]
    dense_nside = params["dense_nside"]
    dense_min = params["dense_min"]
    low_cut = params["low_cut"]
    high_cut = params["high_cut"]
    data_file = params["data_file"]
    folder = params["fortran_folder"]
    input_path_name = params["input_path"]
    output_path_name = params["output_path"]
    fom_file = params["fom_file"]
    param_file = params["param_file"]
    loser_file = params["loser_file"]
    resid_file = params["resid_file"]
    exe = params["exe"]
    fom_rms_cut = params["fom_rms_cut"]
    fom_mag_cut = params["fom_mag_cut"]

    # Reads the desired data file
    df = pd.read_parquet(f"{data_file}.parquet")

    # Finds all the dense pixels
    veto_pixels = find_dense_pixels(df, dense_nside, dense_min)

    # Pixelates and groups objects based on nside and origin step
    combined = pixelate_and_group(df, nside, origin_step, low_cut, high_cut, dense_nside, veto_pixels)

    # Score the foursomes and gets the cell file back for reconstruction
    fom_df, input_path = test_foursomes(combined, input_path_name, folder, fom_file, param_file, loser_file,
                  resid_file, exe, fom_rms_cut, fom_mag_cut)

    # Sorts and writes the FORTRAN output file
    sort_and_write(fom_df, input_path, output_path_name, folder)

if __name__ == "__main__":
    main()
