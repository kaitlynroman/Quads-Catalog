"""
Builds the POTENTIAL_QUADS candidate table (CSV for Google Sheets/Excel import).
Creates Witt-Wynne illustrations of each candidate.

Reads:
  1. FoM file (e.g. fsort6deg.txt) -- sorted FORTRAN output, one quartet per
     line. Reading stops at the first line with fom_rms > RMS_MAX (the file is
     sorted ascending). Provides the object IDs, the two figures of merit, and
     the model parameters.
  2. Cell file (e.g. FORTRAN_input) -- ra/dec and grizy magnitudes per object,
     used to compute the color-scatter FoM and to place the images on the plot.

Keeps only candidates with fom_rms < RMS_MAX, sorts by best (smallest) color
scatter, and writes the top TOP_N to POTENTIAL_QUADS.csv with columns (in this order):
  Mugshot, Graph Output, DES Object IDs, RA, Dec, NCat,
  Figure of Merit (Position), Figure of Merit (Magnitude),
  Figure of Merit (Color), Point-Like/Extended (0 = star, 4 = galaxy),
  Lens Center X, Lens Center Y, Semi-Major Deflection, Position Angle (deg),
  Axis Ratio, Source Position X, Source Position Y,
  Hyperbola Center X, Hyperbola Center Y, Hyperbola Semi-Squared (signed),
  Asymptote Angle (deg)

Import into Google Sheets/Excel: File -> Import -> Upload -> POTENTIAL_QUADS.csv.
The Mugshot column is left blank for manual image entry; Graph Output is
stamped with the matching plot filename, which is saved in candidate_plots.
"""

# =============================================================================
# LIBRARY IMPORTING, DEFAULT VALUES, AND HELPER FUNCTIONS
# =============================================================================

import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RMS_MAX = 0.2       # keep only candidates with fom_rms below this
TOP_N = 20      # write only this many with the best color scatter first
GR_VMIN, GR_VMAX = -0.5, 2.5        # fixed g-r colormap range for graphing
MIN_VALID = 2       # minimum valid objects to be meaningful for the color FoM
EXT_EXTENDED = 2    # ext_mash >= this counts as an extended (galaxy-like) object
MAX_EXTENDED = 1    # at most this many: 0 for a quad, 1 for 3 images + a lens galaxy

def absolute_to_offsets(ra_c, dec_c, ras, decs):
    """
    Converts absolute RA/Dec (deg) to arcsec offsets from a reference point
    Parameters:
        ra_c (float): The reference point RA in degrees
        dec_c (float): The reference point Dec in degrees
        ras (list): The object RA positions in degrees
        dec (list): The object Dec positions in degrees
    Returns a tuple of the image positions in arcsec offsets
    """
    ras, decs = np.asarray(ras, float), np.asarray(decs, float)
    return ((ras - ra_c) * np.cos(np.radians(dec_c)) * 3600.0,
            (decs - dec_c) * 3600.0)

def offsets_to_absolute(racell, deccell, dx_arcsec, dy_arcsec):
    """
    Converts RA/Dec arcsec offsets to absolute degrees from a reference point
    Parameters:
        racell (float): The reference point RA in degrees
        deccell (float): The reference point Dec in degrees
        dx_arcsec (list): The object x positions in arcsec offsets from the reference point
        dy_arcsec (list): The object y positions in arcsec offsets from the reference point
    Returns a tuple of the image positions in absolute degrees
    """
    dec = deccell + dy_arcsec / 3600.0
    ra  = racell  + dx_arcsec / (3600.0 * math.cos(math.radians(deccell)))
    return ra, dec

def iau_name(ra_deg, dec_deg, prefix="DES", short=True):
    """
    Makes an IAU-style designation from the J2000 RA/Dec of the lens center
    Parameters:
        ra_deg (float): The RA of the lens center in degrees
        dec_deg (float): The Dec of the lens center in degrees
        prefix (str): The survey the object was found in (default="DES")
        short (bool): Whether to use hhmm+ddmm or hhmmss.ss+ddmmss.s
    Returns a string name for the object based on the IAU designation system
    """
    # Finds the RA hhmm values
    ra_h = (ra_deg % 360.0) / 15.0
    hh = int(ra_h)
    rem_min = (ra_h - hh) * 60.0
    mm = int(rem_min)
    ss = (rem_min - mm) * 60.0

    # Finds the Dec +ddmm
    sign = "+" if dec_deg >= 0 else "-"
    ad = abs(dec_deg)
    dd = int(ad)
    rem_am = (ad - dd) * 60.0
    am = int(rem_am)
    asec = (rem_am - am) * 60.0

    # If short=True it returns the hhmm+ddmm form
    if short:
        return f"{prefix}J{hh:02d}{mm:02d}{sign}{dd:02d}{am:02d}"

    # If short=False it returns the hhmmss.ss+ddmmss.s form
    ss_t   = math.floor(ss * 100) / 100.0
    asec_t = math.floor(asec * 10) / 10.0
    return (f"{prefix}J{hh:02d}{mm:02d}{ss_t:05.2f}"
            f"{sign}{dd:02d}{am:02d}{asec_t:04.1f}")

def invert_shape(semi_f, angle_f, theta, qp_f):
    """
    Since siepbb writes the derived semi, angle, and qp values and not the raw p(3),p(4),p(5) that the
    conic equations need, we must find these raw values for drawing
    Parameters:
        semi_f (float): the semi value as written in the FoM file
        angle_f (float): the angle value as written in the FoM file
        qp_f (float): the qp value as written in the FoM file
        theta (float): the asymptote angle h(4) that equals the raw p(4)
    Returns a tuple of the raw p(3) and p(5) values
    """
    # Finds the delta between angle_f and theta
    d = (angle_f - theta) % 180.0

    # If the angles are basically perpendicular, square the axis ratio and half and square the semi-major axis
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

def build_cell_lookup(cell_path):
    """
    Creates a dictionary mapping object ids to their sky position, grizy magnitudes,
    and the morphology columns used to test whether an image is resolved
    Parameters:
        cell_path (str): The FORTRAN_input filepath containing the magnitude values
    Returns {object_id: (ra, dec, g, r, i, z, y, ext_mash)},
    with sentinel magnitudes as nan. Magnitudes stay at indices 2..6 so
    color_scatter's column indexing is unaffected.
    """
    lookup = {}
    # Opens the FORTRAN_input file to read
    with open(cell_path) as f:
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
    Calculates the color Figure of Merit (the amount of deviation among the foursome objects' g-r and i-z bands)
    Parameters:
        ids (list): The list of object_ids in the foursome
        lookup (dict): The dict from build_cell_lookup() for getting grizy values for each object
    Returns the color Figure of Merit (Euclidean distance of standard deviations for g-r and i-z)
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

def build_table(fom_path, lookup, out_csv="POTENTIAL_QUADS.csv",
                rms_max=RMS_MAX):
    """
    Builds the .csv table of the TOP_N systems with the best color FoM and positional FoMs < rms_max
    Parameters:
        fom_path (str): The FORTRAN output file to pull positional data from
        lookup (dict): The dict from build_cell_lookup() for getting grizy values for each object
        out_csv (str): The file name for the table (default="POTENTIAL_QUADS.csv")
        rms_max (float): fom_rms threshold; reading stops past it (default=RMS_MAX)
    Writes the .csv table and returns a list of params dicts (one per kept system) for graphing.
    """
    # Table Column Names
    header = [
    "Mugshot", "Graph Output", "DES Object IDs", "RA", "Dec", "NCat",
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
    n_ext_rejected = 0
    with open(fom_path) as f:
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

            # The file is sorted ascending by fom_rms, so the first line past the threshold means every later line is too
            if fom_rms > rms_max:
                break

            # Calculates the color FoM for the foursome, skipping ones that cannot be scored
            cscat = color_scatter(ids, lookup)
            if np.isnan(cscat):
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

            # A 4img has 0 extended objects and a 3img_lens system has exactly 1
            # Two or more galaxies means the foursome is neither, so drop it
            n_extended = sum(1 for v in ext_vals if v >= EXT_EXTENDED)
            if n_extended > MAX_EXTENDED:
                n_ext_rejected += 1
                continue

            # Recover the raw conic parameters the curve equations need.
            theta = hyp[3]
            semi2, qd2 = invert_shape(semi, angle, theta, qp)

            # Name the system from its fitted lens center (p6,p7 are arcsec
            ra_l, dec_l = offsets_to_absolute(racell, deccell, p6, p7)
            name = iau_name(ra_l, dec_l)

            # Inputs data for the candidate row for the .csv file
            row = [
                "", "",                                  # Mugshot, Graph Output (stamped below)
                " ".join(map(str, ids)), f"{racell:.10f}", f"{deccell:.10f}", ncat,
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
                "ids": ids, "name": name,
            }
            candidates.append((cscat, row, params))

    # Keeps only the TOP_N systems with the best color FoMs
    candidates.sort(key=lambda c: c[0])
    top = candidates[:TOP_N]

    # Two systems can share a short IAU name (same arcmin box) so we disambiguate with a letter suffix rather than silently overwriting a plot.
    used = {}
    for _, row, params in top:
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
        for _, row, _ in top:
            w.writerow(row)

    # Returns the params dicts for graphing
    print(f"{n_ext_rejected:,} foursomes rejected for having more than {MAX_EXTENDED} "
          f"extended object (ext_mash >= {EXT_EXTENDED}).")
    print(f"{len(candidates):,} candidates passed fom_rms < {rms_max} and had a "
          f"color score.\nWrote the top {len(top)} (by color FoM) to {out_csv}.")
    return [params for _, _, params in top]

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

def plot_row(p, lookup, outdir="candidate_plots"):
    """
    Plots the graph using FORTRAN_input, wynne_ellipse(), and witt_hyperbola()
    Parameters:
        p (dict): The params dict returned by build_table()
        lookup (dict): The dict from build_cell_lookup() for getting grizy values for each object
        outdir (str): The name of the folder to put images in (default="candidate_plots")
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
        ax.text(ox[i] + 0.06, oy[i] + 0.06, lab, fontsize=12, weight="bold")

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
    fig.savefig(os.path.join(outdir, p["graph"]), dpi=130)
    plt.close(fig)

def main():
    # Uses these files for getting data
    fom_path  = "fsort6deg.txt"
    cell_path = "FORTRAN_input"

    # Builds the lookup dict and params dict
    lookup = build_cell_lookup(cell_path)
    params_list = build_table(fom_path, lookup)

    # Creates plots for each foursome
    for p in params_list:
        plot_row(p, lookup)
    print(f"Plotted {len(params_list)} candidates.")

if __name__ == "__main__":
    main()
