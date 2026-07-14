"""
Part 3:
    1. Fetches FORTRAN input data and FORTRAN output data.
    2. Calculates the color FoM, cutting systems with fom_color < FOM_COLOR_CUT
        and sorts by best color FoM to highest
    3. Creates a .csv table with the top systems with columns:
        Mugshot, Graph Output, System Name, DES Object IDs, J2000, RA, Dec, NCat,
        Figure of Merit (Position), Figure of Merit (Magnitude),
        Figure of Merit (Color), Point-Like/Extended (0 = star, 4 = galaxy),
        Lens Center X, Lens Center Y, Semi-Major Deflection, Position Angle (deg),
        Axis Ratio, Source Position X, Source Position Y,
        Hyperbola Center X, Hyperbola Center Y, Hyperbola Semi-Squared (signed),
        Asymptote Angle (deg).
    4. Names and graphs each top system's Witt-Wynne diagram and object coloring for analysis.

Creates a .csv data file sorted by best color FoM and draws illustrations of each candidate
written to the candidate plots folder.
"""

# =============================================================================
# LIBRARY IMPORTING, DEFAULT VALUES, AND HELPER FUNCTIONS
# =============================================================================

import os
import csv
import math
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GR_VMIN, GR_VMAX = -0.5, 2.5 # Fixed g-r colormap range for graphing
MIN_VALID = 2 # The minimum valid objects in a foursome to be meaningful for the color FoM

def absolute_to_offsets(ra_c, dec_c, ras, decs):
    """
    Converts absolute RA/Dec (deg) to arcsec offsets from a reference point.
    Parameters:
        ra_c (float): The reference point RA in degrees
        dec_c (float): The reference point Dec in degrees
        ras (list): The object RA positions in degrees
        dec (list): The object Dec positions in degrees
    Returns a tuple of the image positions in arcsec offsets.
    """
    ras, decs = np.asarray(ras, float), np.asarray(decs, float)
    return ((ras - ra_c) * np.cos(np.radians(dec_c)) * 3600.0,
            (decs - dec_c) * 3600.0)

def offsets_to_absolute(racell, deccell, dx_arcsec, dy_arcsec):
    """
    Converts RA/Dec arcsec offsets to absolute degrees from a reference point.
    Parameters:
        racell (float): The reference point RA in degrees
        deccell (float): The reference point Dec in degrees
        dx_arcsec (list): The object x positions in arcsec offsets from the reference point
        dy_arcsec (list): The object y positions in arcsec offsets from the reference point
    Returns a tuple of the image positions in absolute degrees.
    """
    dec = deccell + dy_arcsec / 3600.0
    ra  = racell  + dx_arcsec / (3600.0 * math.cos(math.radians(deccell)))
    return ra, dec

def iau_name(ra_deg, dec_deg, short=None, prefix="DES"):
    """
    Makes an IAU-style designation from the J2000 RA/Dec of the lens center.
    Parameters:
        ra_deg (float): The RA of the lens center in degrees
        dec_deg (float): The Dec of the lens center in degrees
        short (bool): Whether to make the system name or give J2000 coords (default=None)
        prefix (str): The survey the object was found in (default="DES")
    Returns a string name for the object based on the IAU designation system.
    """
    # Initialize coordinate in the ICRS (J2000) frame
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    name = coords.to_string('hmsdms')

    # If short=False, it returns the full coordinates in J2000
    if not short:
        return name

    # If short=False, it returns the system name
    decimal_name = name.split()
    new_name = decimal_name[0][:2] + decimal_name[0][3:5] + decimal_name[1][:3] + decimal_name[1][4:6]
    return (f"{prefix}J{new_name}")

def invert_shape(semi_f, angle_f, theta, qp_f):
    """
    Since siepbb writes the derived semi, angle, and qp values and not the raw p(3),p(4),p(5) that the
    conic equations need, we must find these raw values for drawing.
    Parameters:
        semi_f (float): the semi value as written in the FoM file
        angle_f (float): the angle value as written in the FoM file
        qp_f (float): the qp value as written in the FoM file
        theta (float): the asymptote angle h(4) that equals the raw p(4)
    Returns a tuple of the raw p(3) and p(5) values.
    """
    # Finds the delta between angle_f and theta
    d = (angle_f - theta) % 180.0

    # If the angles are basically perpendicular, square and divide by the axis ratio, then half and square the semi-major axis
    branch_A = min(abs(d - 90.0), abs(d + 90.0), abs(d - 270.0)) < 1.0
    if branch_A:
        qd2   = qp_f ** 2
        semi2 = (semi_f / qp_f) ** 2 if qp_f else np.nan

    # If the angles are not perpendicular, take the reciprocal of the square of the axis ratio and square the semi-major axis
    else:
        qd2   = 1.0 / qp_f ** 2 if qp_f else np.nan
        semi2 = semi_f ** 2
    return semi2, qd2

# =============================================================================
# TABLE CREATION
# =============================================================================

def build_cell_lookup(input_path):
    """
    Creates a dictionary mapping object ids to their sky position, grizy magnitudes,
    and the morphology columns used to test whether an image is resolved.
    Parameters:
        input_path (str): The name of the FORTRAN input file (has magnitude values)
    Returns {object_id: (ra, dec, g, r, i, z, y, ext_mash)} with bad magnitudes as NaN.
    """
    lookup = {}
    # Opens the FORTRAN_input file to read
    with open(input_path) as f:
        for line in f:
            t = line.split()
            # Skips the header lines (which start with "cell") and empty lines
            if not t or t[0] == "cell":
                continue
            # Gets the data from the file for the mapping
            try:
                oid = int(float(t[0]))
                ra, dec = float(t[1]), float(t[2])
                mags = []
                for v in t[3:8]:
                    x = float(v)   # "nan" in the file parses to nan here
                    mags.append(x)
                ext_mash = float(t[8])
                # Adds the object ID and its ra, dec, magnitudes, and morphology to the dict
                lookup[oid] = (ra, dec, *mags, ext_mash)
            except (IndexError, ValueError):
                continue
    return lookup

def color_scatter(ids, lookup):
    """
    Calculates the color Figure of Merit (the amount of deviation among the foursome objects' g-r and i-z bands).
    Parameters:
        ids (list): The list of object_ids in the foursome
        lookup (dict): The dict from build_cell_lookup() for getting grizy values for each object
    Returns the color Figure of Merit (Euclidean distance of standard deviations for g-r and i-z).
    """
    # Returns the row of (ra, dec, grizy) for each object in the system
    found = [lookup[i] for i in ids if i in lookup]

    # Converts the rows to an array of floats
    if not found:
        return np.nan
    arr = np.array(found, dtype=float)

    # Helper function to calculate the standard deviation of a color across the objects
    def _scatter(color):
        valid = color[np.isfinite(color)]
        if len(valid) < MIN_VALID:
            return np.nan
        return float(np.std(valid))

    # Finds the g-r and i-z for the objects and returns the color Figure of Merit
    gr_s = _scatter(arr[:, 2] - arr[:, 3])
    iz_s = _scatter(arr[:, 4] - arr[:, 5])
    if np.isnan(gr_s) or np.isnan(iz_s):
        return np.nan
    return float(np.hypot(gr_s, iz_s))

def build_table(output_path, lookup, fom_color_cut, out_csv):
    """
    Builds the .csv table of the systems sorted by best color FoM.
    Parameters:
        output_path (str): The FORTRAN output file to pull positional data from
        lookup (dict): The dict from build_cell_lookup() for getting grizy values for each object
        fom_color_cut (float): The minimum fom_color for a foursome to be kept as a candidate
        out_csv (str): The file name for the table
    Writes the .csv table and returns a list of params dicts (one per kept system) for graphing.
    """
    # Table Column Names
    header = [
    "Mugshot", "Graph Output", "System Name", "DES Object IDs", "J2000 (for IPAC NED lookup)",
    "RA", "Dec", "NCat",
    "Figure of Merit (Position)",
    "Figure of Merit (Magnitude)",
    "Figure of Merit (Color)",
    "Point-Like/Extended (0 = star, 4 = galaxy)",
    "Lens Center X", "Lens Center Y",
    "Semi-Major Deflection",
    "Position Angle (deg)",
    "Axis Ratio",
    "Source Position X", "Source Position Y",
    "Hyperbola Center X", "Hyperbola Center Y",
    "Hyperbola Semi-Squared (signed)",
    "Asymptote Angle (deg)",
    ]

    # Reads the FORTRAN_output file to get needed values for the table
    candidates = []
    with open(output_path) as f:
        for line in f:
            t = line.split()
            # Skip short/blank lines before indexing into them
            if len(t) < 25:
                continue
            # Prepares data for the .csv file
            try:
                ids     = [int(float(x)) for x in t[8:12]]
                racell  = float(t[1])
                deccell = float(t[2])
                ncat    = int(t[3])
                fom_rms  = float(t[23])
                fom_frms = float(t[24])
                p6, p7   = float(t[12]), float(t[13])              # lens center, arcsec offsets
                semi     = float(t[14])                            # DERIVED (see invert_shape)
                angle    = float(t[15])                            # DERIVED position angle
                qp       = float(t[16])                            # DERIVED axis ratio
                p1, p2   = float(t[17]), float(t[18])              # source in arcsec offsets
                hyp      = [float(t[j]) for j in range(19, 23)]    # h(1..4)
            except (IndexError, ValueError):
                continue

            # Calculates the color FoM for the foursome, skipping ones greater than FOM_COLOR_CUT
            cscat = color_scatter(ids, lookup)
            if np.isnan(cscat) or cscat > fom_color_cut:
                continue

            # Get the ext_mash values for each object in the foursome
            ext_vals, ext_list = [], []
            for oid in ids:
                if oid not in lookup:
                    ext_list.append("na")
                    continue
                ext_mash = lookup[oid][7]
                if np.isnan(ext_mash):
                    ext_list.append("nan")
                    continue
                ext_vals.append(ext_mash)
                ext_list.append(str(int(ext_mash)))

            # Recover the raw conic parameters the curve equations need.
            theta = hyp[3]
            semi2, qd2 = invert_shape(semi, angle, theta, qp)

            # Name the system from its fitted lens center (p6,p7 are arcsec
            ra_l, dec_l = offsets_to_absolute(racell, deccell, p6, p7)
            name = iau_name(ra_l, dec_l, short=True)
            j2000 = iau_name(ra_l, dec_l, short=False)

            # Inputs data for the candidate row for the .csv file
            row = [
                "", "", f"{name}",                               # Mugshot, Graph Output (stamped below)
                " ".join(map(str, ids)), f"{iau_name(ra_l, dec_l, short=False)}", f"{racell:.10f}", f"{deccell:.10f}", ncat,
                f"{fom_rms:.4f}", f"{fom_frms:.4f}", f"{cscat:.4f}",
                " ".join(map(str, ext_list)),
                f"{p6:.6g}", f"{p7:.6g}", f"{semi:.6g}",
                f"{angle:.6g}", f"{qp:.6g}", f"{p1:.6g}", f"{p2:.6g}",
                f"{hyp[0]:.6g}", f"{hyp[1]:.6g}", f"{hyp[2]:.6g}", f"{hyp[3]:.6g}",
            ]
            # Creates a params dict for the graphing portion (raw values for the conics)
            params = {
                "racell": racell, "deccell": deccell,
                "source": (p1, p2), "lens": (p6, p7),
                "semi2": semi2, "qd2": qd2, "theta_deg": theta,
                "hyp_center": (hyp[0], hyp[1]), "h3": hyp[2],
                "ids": ids, "name": name, "j2000": j2000,
            }
            candidates.append((cscat, row, params))

    # Sorts the candidates by their color FoM
    candidates.sort(key=lambda c: c[0])

    # Two systems can share an IAU name so we disambiguate with a letter suffix
    used = {}
    for _, row, params in candidates:
        base = params["name"]
        n = used.get(base, 0)
        used[base] = n + 1
        graph = f"{base}" if n == 0 else f"{base}{chr(ord('a') + n)}"
        params["graph"] = graph
        row[1] = graph

    # Writes the .csv file to be output
    with open(out_csv, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(header)
        for _, row, _ in candidates:
            w.writerow(row)

    # Returns the params dicts for graphing
    print(f"{len(candidates):,} candidates passed and had a "
          f"color score.\nWrote the {len(candidates)} candidates (sorted by color FoM) to {out_csv}.")
    return [params for _, _, params in candidates]

# =============================================================================
# GRAPH CREATION
# =============================================================================

def wynne_ellipse(p, n=400):
    """
    Uses the raw model parameters to graph the Wynne Ellipse.
    In the model-aligned frame the ellipse is  x^2 + y^2/qd2 = semi2,
    centered on the source, with the model axis at angle theta = p(4).
    Semi-axis sqrt(semi2) lies along the model axis; sqrt(semi2*qd2) is
    perpendicular (if qd2 > 1 the model-axis semi is the minor axis)
    Parameters:
        p (dict): The params dict returned by build_table()
        n (int): The number of data points to generate (default=400)
    Returns an (n,2) array of x,y points in the original frame
    """
    # Gets the source (ellipse center) and the model axis angle in radians
    sx, sy = p["source"]
    th = np.radians(p["theta_deg"])
    ct, st = np.cos(th), np.sin(th)

    # Parametrize the ellipse in the aligned frame then translate back
    t = np.linspace(0, 2 * np.pi, n)
    ux = np.sqrt(p["semi2"]) * np.cos(t)
    uy = np.sqrt(p["semi2"] * p["qd2"]) * np.sin(t)

    # Returns the ellipse as a 2D array for graphing
    return np.column_stack([sx + ct * ux - st * uy, sy + st * ux + ct * uy])

def witt_hyperbola(p, n=400):
    """
    Uses the raw model parameters to graph the Witt Hyperbola.
    In the aligned, hyperbola-centered frame it is  u*v = h3/2, a rectangular
    hyperbola whose asymptotes lie along the model axes (angle theta = p(4)).
    h3 may be negative, which flips which pair of quadrants is occupied
    Parameters:
        p (dict): The params dict returned by build_table()
        n (int): The number of data points per branch (default=400)
    Returns a list of two (n,2) arrays (one per branch) in the original frame
    """
    # Gets the hyperbola center and the constant c in u*v = c
    hx, hy = p["hyp_center"]
    c = p["h3"] / 2.0

    # Gets the model axis angle in radians and k sets the parametrization scale
    th = np.radians(p["theta_deg"])
    ct, st = np.cos(th), np.sin(th)
    k = np.sqrt(abs(c)) if c else 1e-6

    # Sweep u exponentially over both branches (u = +/- k e^t), with v = c/u
    t = np.linspace(-3.0, 3.0, n)
    out = []
    for sign in (+1, -1):
        u = sign * k * np.exp(t)
        v = c / u
        out.append(np.column_stack([hx + ct * u - st * v, hy + st * u + ct * v]))
    # Returns both branches for graphing
    return out

def plot_row(p, lookup, outdir):
    """
    Plots the graph using FORTRAN_input, wynne_ellipse(), and witt_hyperbola()
    Parameters:
        p (dict): The params dict returned by build_table()
        lookup (dict): The dict from build_cell_lookup() for getting grizy values for each object
        outdir (str): The name of the folder to put images in
    Saves the plots in PNG form to outdir
    """
    # Gets the sky position and g-r of each image for graphing
    ra, dec, gr = [], [], []
    for oid in p["ids"]:
        if oid not in lookup:
            continue
        rec = lookup[oid]
        ra.append(rec[0]); dec.append(rec[1]); gr.append(rec[2] - rec[3])
    if len(ra) < 4:
        print(f"  {p['graph']}: only {len(ra)}/4 images in cell file; skipped")
        return

    # Converts the image positions to arcsec offsets from the cell center, the same frame the model parameters live in
    ox, oy = absolute_to_offsets(p["racell"], p["deccell"], ra, dec)

    # Creates a 7x7 inch figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Determines which objects have and do not have a measurable g-r for coloring
    gr = np.asarray(gr, float)
    good = np.isfinite(gr)

    # Objects with a measurable g-r are colored on the diverging scale
    if good.any():
        sc = ax.scatter(ox[good], oy[good], c=gr[good], cmap="RdBu_r",
                        vmin=GR_VMIN, vmax=GR_VMAX, s=200,
                        edgecolor="black", linewidth=1.0, zorder=5)
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label("g-r: (blue = quasar-like,  red = galaxy-like)")

    # Objects with NaN g-r (missing/sentinel photometry) are colored gray
    if (~good).any():
        ax.scatter(ox[~good], oy[~good], color="0.65", s=200,
                   edgecolor="black", linewidth=1.0, zorder=5,
                   label="no g-r (missing photometry)")

    # Labels points numbers 1-4
    for i, lab in enumerate("1234"[:len(ox)]):
        ax.text(ox[i] + 0.1, oy[i] + 0.1, lab, fontsize=12, weight="bold")

    # Plots the Wynne Ellipse and Witt Hyperbola
    ell = wynne_ellipse(p)
    ax.plot(ell[:, 0], ell[:, 1], color="tab:red", lw=1.3, label="Wynne Ellipse")
    for k, br in enumerate(witt_hyperbola(p)):
        ax.plot(br[:, 0], br[:, 1], color="tab:orange", lw=1.3,
                label="Witt Hyperbola" if k == 0 else None)

    # Plots the source position and the lens center
    ax.scatter(*p["source"], marker="D", color="red", s=200, zorder=6, label="source")
    ax.scatter(*p["lens"], marker="x", color="black", s=130, zorder=6, label="lens center")

    # Creates the graph axes, labels, and titles
    pad = 1.0
    ax.set_xlim(ox.min() - pad, ox.max() + pad)
    ax.set_ylim(oy.min() - pad, oy.max() + pad)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\Delta$RA (arcsec)  [East $\leftarrow$]")
    ax.set_ylabel(r"$\Delta$Dec (arcsec)  [$\uparrow$ North]")
    ax.set_title(f"{p['graph']} ({p['racell']:.4f}, {p['deccell']:.4f})")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    # Saves the figures in a folder called outdir
    os.makedirs(outdir, exist_ok=True)
    graph_name = f"{p["graph"]}.png"
    fig.savefig(os.path.join(outdir, graph_name), dpi=130)
    plt.close(fig)

def main(params):
    # Parameter values from final.py
    output_path = params["output_path"]
    input_path = params["input_path"]
    csv_name = params["csv_name"]
    plot_folder = params["plot_folder"]
    fom_color_cut = params["fom_color_cut"]

    # Builds the lookup dict and params dict
    lookup = build_cell_lookup(input_path)
    params_list = build_table(output_path, lookup, fom_color_cut, csv_name)

    # Creates plots for each foursome
    for p in params_list:
        plot_row(p, lookup, plot_folder)
    print(f"Plotted {len(params_list)} candidates.")

if __name__ == "__main__":
    main()
