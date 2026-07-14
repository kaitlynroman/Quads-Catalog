"""
Quadruply-Lensed System Search (DES DR2 Y6 GOLD)

Pipeline:
User inputs search radius, sets desired HEALPix nside, sets origin step, sets whether to score
for 4 image or 3 image with lens quad systems, and adjusts base parameters if need be.

PART 1: QUERY FOR DATA IN SQL/PYTHON
    1. Checks to see if there is an existing query data file for the requested cone center
        and radius
    2. If there is a .parquet file (download and DataLab query completed), it returns
        that file for use
    3. If there is no .parquet file:
        4. Checks to see if there is a .csv file of the same name (indicating that a previous
            query completed but was not downloaded), and downloads that file without requerying
        5. If there is not, it queries DES DR2 Y6 Gold for coadd_object_id, ra, dec,
            psf_mag_aper_8_g_corrected, psf_mag_aper_8_r_corrected,
            psf_mag_aper_8_i_corrected, psf_mag_aper_8_z_corrected, psf_mag_aper_8_y_corrected,
            ext_mash, bdf_t, bdf_g_1, bdf_g_2, psf_t, psf_g_1, and psf_g_2 and downloads the
            data, so it does not need to be queried again.

PART 2: GROUPING/POSITIONAL SCORING IN PYTHON/FORTRAN
    6. Flag dense coarse pixels (>= DENSE_MIN objects at DENSE_NSIDE) and their
       neighbors (likely star clusters/rich groups) and return a veto set.
    7. Pixelate/group objects at (nside, origin_step), keeping cells with 4 to 7
       objects and dropping any cell whose center falls in a vetoed coarse pixel,
       then write the FORTRAN input cell file.
    8. CALLS FORTRAN: form all foursomes per cell and score them (positional rms
       and magnitude frms) for the desired quad model (4img or 3img_lens).
    9. Parse the FoM output, cutting foursomes with fom_rms > FOM_RMS_CUT or
        fom_mag > FOM_MAG_CUT.
    10. Re-emit the survivors and sort by positional rms.

PART 3: COLOR SCORING AND DATA ANALYSIS IN PYTHON
    11. Fetches FORTRAN input data and FORTRAN output data.
    12. Calculates the color FoM, cutting systems with fom_color < FOM_COLOR_CUT
        and sorts by best color FoM to highest
    13. Creates a .csv table with the top systems with columns:
        Mugshot, Graph Output, DES Object IDs, RA, Dec, NCat,
        Figure of Merit (Position), Figure of Merit (Magnitude),
        Figure of Merit (Color), Point-Like/Extended (0 = star, 4 = galaxy),
        Lens Center X, Lens Center Y, Semi-Major Deflection, Position Angle (deg),
        Axis Ratio, Source Position X, Source Position Y,
        Hyperbola Center X, Hyperbola Center Y, Hyperbola Semi-Squared (signed),
        Asymptote Angle (deg).
    14. Names and graphs each top system's Witt-Wynne diagram and object coloring for analysis.

To Run:
    - Ensure the desired data file, final.py, p1.py, p2.py, p3.py and the folder
        containing all the files for FORTRAN to run are in the same directory
    - Update the MakeFile to use the specific FCC and CC used by the computer
    - Ensure the following libraries are installed and Python is up to date:
        - astro-datalab
        - pyarrow.parquet
        - pandas
        - numpy
        - astropy
        - healpy
        - matplotlib
    - To run on Anaconda Prompt, type:
        conda activate datalab
        cd "{INSERT DIRECTORY}"
        python final.py
    - Input desired half-angle search radius, HEALPix nside, origin step, and whether to score
        for 4img or 3img_lens (option to adjust base parameters given as well)
"""

# =============================================================================
# FILE IMPORTING, DEFAULT VALUES, AND HELPER FUNCTIONS
# =============================================================================

import p1
import p2
import p3

CODED_DEFAULTS = {
    "ra_center": "0h51m26s",        # SGP RAcone center (J2000)
    "dec_center": "-27d07m42s",     # SGP Dec cone center (J2000)
    "min_mag": 10.0,                # min valid magnitude (else marked NaN)
    "max_mag": 35.0,                # max valid magnitude (else marked NaN)
    "tile_nside": 32,               # DES hpix_32 tile nside
    "batch_size": 10,               # hpix_32 cells per query
    "dense_nside": 16384,           # overdensity counting nside
    "dense_min": 10,                # min objects for a dense pixel
    "low_cut": 4,                   # min objects FORTRAN can score
    "high_cut": 7,                  # max objects FORTRAN can score
    "fom_rms_cut": 0.2,             # cut positional rms above this
    "fom_mag_cut": 0.2,             # cut magnitude rms above this
    "fom_color_cut": 0.2,           # cut color rms above this
}

def _ask(prompt, cast):
    """
    Prompts the user to get desired pipeline parameters.
    Parameters:
        prompt (str): The prompt given to the user
        cast (type): The object type to cast the input string to
    Returns the casted input by the user.
    """
    raw = input(f"  {prompt}: ").strip()
    # Guards aganist invalid inputs
    try:
        return cast(raw)
    except (ValueError, TypeError):
        print(f"    Could not parse {raw!r}")

def _parse_origin(raw):
    """
    Parses the origin tuple input (given as either '0.0,0.0', '0.25 0.0',
    or '(0.0, 0.0)').
    Parameters:
        raw (str): The user input for the origin step
    Returns a tuple of floats for the parameter list.
    """
    # Strips parenthesises, turns commas into spaces, and then splits on whitespace
    cleaned = raw.replace("(", " ").replace(")", " ").replace(",", " ")
    parts = cleaned.split()
    # Guards aganist invalid inputs
    if len(parts) != 2:
        raise ValueError("need two numbers (dRA, dDec)")
    return (float(parts[0]), float(parts[1]))

def _origin_tag(origin_step):
    """
    Generates a compact filename tag for an origin step (e.g. (0.25, 0.0) becomes 0p25-0p0).
    Parameters:
        origin_step (tuple): The origin parsed from _parse_origin()
    Returns a string for the origin step such that the filename can be considered valid.
    """
    # Replaces decimals and minus signs to be letters
    a, b = origin_step
    fmt = lambda x: str(x).replace(".", "p").replace("-", "m")
    return f"{fmt(a)}-{fmt(b)}"

def _radius_tag(r):
    """
    Generates a compact radius tag for the half-angle radius (e.g. 6.0 becomes 6 and 6.5
    becomes 6p5).
    Parameters:
        r (float): The casted radius input from the user
    Returns a string for the radius such that the filename can be considered valid.
    """
    # Returns a whole number casted as a string or a str with the decimal replaced by the letter "p"
    return str(int(r)) if float(r).is_integer() else str(r).replace(".", "p")

def _coerce_like(raw, old):
    """
    Casts a raw string to the type of the "old" value.
    Parameters:
        raw (str): The user input to be casted
        old (any): The input type for raw to be converted to
    Returns the casted version of raw.
    """
    raw = raw.strip()
    if isinstance(old, tuple):
        return _parse_origin(raw)
    if isinstance(old, bool):
        return raw.lower() in ("1", "true", "yes", "y", "t")
    if isinstance(old, int):
        return int(raw)
    if isinstance(old, float):
        return float(raw)
    return raw

def _split_keys(raw):
    """
    Split a comma/space-separated key list into clean names.
    Parameters:
        raw (list): The separated str list of keys to be converted into clean names
    Returns a list of the cleaned keys.
    """
    return [p.strip() for p in raw.replace(",", " ").split() if p.strip()]

def _split_values(raw, olds):
    """
    Splits a comma-separated value list, keeping parenthesised tuples intact so
    a tuple value like '(0.25, 0.0)' is not broken on its internal comma.
    Parameters:
        raw (list): The separated str list of values to be split
        olds (list): The old values that are unused for the split but document the
            alignment with keys
    Returns a list of the cleaned values.
    """
    out, buf, depth = [], "", 0
    # Keeps parenthesis and commas intact during the split so origin_steps can be adjusted
    for ch in raw:
        if ch == "(":
            depth += 1; buf += ch
        elif ch == ")":
            depth -= 1; buf += ch
        elif ch == "," and depth == 0:
            out.append(buf.strip()); buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf.strip())
    return out

# =============================================================================
# SETTING SEARCH PARAMETERS
# =============================================================================

def collect_search_settings(params):
    """
    Prompts the user for the desired search settings and collects them into a dict.
    Parameters:
        params (dict): A dict (typically empty) mapping terms used in p1, p2, and p3 to their values
    Returns the updated dict for configuration.
    """
    # Prints the catalog being used for display
    print("=" * 60)
    print(f"SEARCHING THE DES DR2 Y6 GOLD DATABASE")
    print("=" * 60)
    print("\n=== Search Settings ===")

    # Gets user input for the radius, nside, and origin step
    params["radius_deg"]  = _ask("Half-Angle Search Radius (deg)", float)
    params["nside"]       = _ask("HEALPix NSIDE", int)
    params["origin_step"] = _ask("Origin Step (dRA, dDec)", _parse_origin)

    # Gets the desired scoring model (3img_lens not available yet)
    raw = input("  Scoring Model (4 for 4img, 3 for 3img_lens): ").strip()
    if raw == "3":
        print("    3img_lens scoring cannot be done yet (using 4img instead for now)")
        params["model"] = "4img"
    else:
        params["model"] = "4img"

    return params

def derive_config(params):
    """
    Adds all the file names (from the search settings) and coded defaults to params.
    Parameters:
        params (dict): The dict to be updated
    Returns the updated dict mapping terms to their values.
    """
    # Gets info for conventional file naming
    r  = _radius_tag(params["radius_deg"])
    ns = params["nside"]
    tag = _origin_tag(params["origin_step"])

    # Adds the default values coded in p1, p2, and p3 (can be adjusted)
    for k, v in CODED_DEFAULTS.items():
        params.setdefault(k, v)

    # Sets default file names based on conventional system
    params["data_file"]      = f"sgp_{r}_data"
    params["fortran_folder"] = "FORTRAN"
    params["input_path"]     = f"FORTRAN_{r}_{ns}_{tag}_input.txt"
    params["output_path"]    = f"FORTRAN_{r}_{ns}_{tag}_output.txt"
    params["fom_file"]       = "fom.txt"
    params["param_file"]     = "param.txt"
    params["loser_file"]     = "loser.txt"
    params["resid_file"]     = "resid.txt"
    params["exe"]            = "siepmod.exe"
    params["csv_name"]       = f"potential_quads_{r}_{ns}_{tag}.csv"
    params["plot_folder"]    = f"candidate_plots_{r}_{ns}_{tag}"

    return params

def show_config(params):
    """
    Prints the current configuration for the user to analyze.
    Parameters:
        params (dict): The updated dict of values from derive_config()
    Prints the current configuration to the screen.
    """
    print("\n========== Current Configuration ==========")
    for k, v in params.items():
        print(f"  {k:16s} = {v!r}")

def confirm_and_edit(params):
    """
    Shows the config and asks the user if the program should run with these values.
    Y -> return it. N -> take parallel lists of keys and new values, apply, then re-show and re-ask.
    Parameters:
        params (dict): The updated/edited dict of values from derive_config()
    Returns the updated params dict when the user approves the values.
    """
    # Runs until receiving a "Y" or "Yes"
    while True:
        show_config(params)
        ans = input("\nRun with this configuration? (Y/N): ").strip().lower()
        if ans in ("y", "yes"):
            return params

        # If not approved, prompts the user so they can update the values
        keys = _split_keys(input("  Keys to change (comma-separated): "))
        if not keys:
            print("    No keys given so nothing changed")
            continue
        vals = _split_values(input("  New values (comma-separated in same order): "),
                             [params.get(k) for k in keys])

        # If there is a mismatch in parallel list length, it change nothing and has the user try again
        if len(keys) != len(vals):
            print(f"    Got {len(keys)} keys but {len(vals)} values so nothing changed")
            continue

        # Iterates through changing the values in the dict
        for k, v in zip(keys, vals):
            if k not in params:
                print(f"    '{k}' is not a parameter so it is skipped")
                continue
            # Ensures correct datatype is given for each value
            try:
                params[k] = _coerce_like(v, params[k])
                print(f"    {k} -> {params[k]!r}")
            except (ValueError, TypeError) as e:
                print(f"    Could not set {k} to {v!r}: {e}")

        # If names depend on radius/nside/origin_step and those changed, they are re-derived
        if any(k in ("radius_deg", "nside", "origin_step") for k in keys):
            derive_config(params)
            print("    (file names re-derived from updated search settings)")

def configure():
    """
    Runs the full interactive configuration.
    Returns the confirmed params dict.
    """
    # Collects user input settings
    params = collect_search_settings({})

    # Derives filenames
    params = derive_config(params)

    # User confirms and/or edits
    params = confirm_and_edit(params)

    print("\nConfiguration confirmed! Running pipeline...")
    return params

# =============================================================================
# RUNNING PIPELINE
# =============================================================================

def main():
    # User configures the settings
    params = configure()

    # Runs p1.py
    print("\n========== PART 1: QUERY FOR DATA IN SQL/PYTHON ==========\n")
    p1.main(params)

    # Runs p2.py
    print("\n========== PART 2: GROUPING/POSITIONAL SCORING IN PYTHON/FORTRAN ==========\n")
    p2.main(params)

    # Runs p3.py
    print("\n========== PART 3: COLOR SCORING AND DATA ANALYSIS IN PYTHON ==========\n")
    p3.main(params)

    # Alerts user when the pipeline is completed
    print("\n========== PIPELINE COMPLETE ==========")

if __name__ == "__main__":
    main()
