import pandas as p
import os
import re
import pickle
import yt
import importlib.resources as pkg_resources
from matplotlib.backends.backend_pdf import PdfPages

def get_dirs(which_halo):
    """
    Collects all the directories with available information.

    Parameters:
        which_halo (str): Six-digit string denoting which halo to run on.

    Returns:
        list[str]: List of available data (by redshift indicator).
    """

    path = "/mnt/research/turbulence/FOGGIE/" + "halo_" + which_halo + "/nref11c_nref9f/"
    return [d for d in sorted(os.listdir(path)) if re.match(r'RD*', d)]  

def read_halo_c_v(which_halo):
    """
    Reads in values for central positions and velocity for a given dataset.

    Parameters:
        dirs (list[str]): List of available data (by redshift indicator).
        which_halo (str): Six-digit string denoting which halo to run on.

    Returns:
        pandas.DataFrame: Sorted list of halo centers stored in a DataFrame with their corresponding redshift indicators.
    """

    dirs = get_dirs(which_halo)

    redshift, name = [], []
    xc, yc, zc, xv, yv, zv = [], [], [], [], [], []

    filename = "/mnt/research/turbulence/FOGGIE/foggie_git_repo/foggie/halo_infos/"+ which_halo +"/nref11c_nref9f/halo_c_v"
    cols = ["redshift ", "name ", "time ", "x_c ", "y_c ", "z_c ", "v_x ", "v_y ", "v_z "]  # Note the spaces to match the file exactly

    # Read in object position listing from FOGGIE data file
    df = p.read_csv(filename, sep='|', lineterminator='\n', skipinitialspace = True, usecols=cols) 
    
    # Strip extra spaces inheading names 
    df = df.rename(columns=lambda x: x.strip()) 
    
    # Iterate over data file and only read in the values for the RD#### rows. Leave everything else. 
    for item in range(len(df['name'])):
        checkString = df.iloc[item]['name'].strip()
        if checkString[0] == "R":
            redshift.append(df.iloc[item]['redshift'])
            name.append(df.iloc[item]['name'].strip())
            xc.append(df.iloc[item]['x_c'])
            yc.append(df.iloc[item]['y_c'])
            zc.append(df.iloc[item]['z_c'])
            xv.append(df.iloc[item]['v_x'])
            yv.append(df.iloc[item]['v_y'])
            zv.append(df.iloc[item]['v_z'])

    dic = {"Redshift": redshift, "Name": name, "XC":xc, "YC":yc, "ZC":zc, "XV":xv, "YV":yv, "ZV":zv}
    df_temp = p.DataFrame(dic)
    df_sort = df_temp.sort_values("Name") # Make sure everything is in order

    for Name in df_sort["Name"]:
        res = any(Name == dir for dir in dirs)
        if res == False: 
            # print("\n \n \n We Do not have data at" + str(dir) + " Removing \n \n \n")
            df_sort = df_sort.drop(df_sort[df_sort["Name"] == Name].index)

    return df_sort

def get_center(which_halo, dir):
    """
    Converts a 3-item list in kpc into a 3-item list in code lengths.

    Parameters:
        dir (str): Redshift indicator.
        df (pandas.DataFrame): DataFrame that includes the information about the centers.
        ds (yt.dataset): Dataset of information to set the quantity.

    Returns:
        list[float]: Three-component list with the x, y, and z location in proper units.
    """
    
    df = read_halo_c_v(which_halo)  # Read in the halo center information for the given halo

    # Pick out central values 
    x_ = df.loc[df.Name == dir, "XC"].values[0]
    y_ = df.loc[df.Name == dir, "YC"].values[0]
    z_ = df.loc[df.Name == dir, "ZC"].values[0]

    return [x_, y_, z_] * yt.units.kpc

def load_foggie_data(which_halo, dir):
    """
    Loads in the FOGGIE data for a given halo and redshift indicator.

    Parameters:
        which_halo (str): Six-digit string denoting which halo to run on.
        dir (str): Redshift indicator.
    """
    
    name = f"/mnt/research/turbulence/FOGGIE/halo_{which_halo}/nref11c_nref9f/{dir}/{dir}"
    ds = yt.load(name)
    return ds

def def_cgm(which_halo, dir, ds, outside_value=None, frac_cutout = 0.05):
    """
    Defines a large sphere with a small sphere removed to represent the CGM.

    Parameters:
        center (list[float]): Three-component list with the x, y, and z location in proper units.
        ds (yt.dataset): Dataset of information to set the quantity.
        outside_value (yt.ds.quan): Value (with units) expressing the virial radius.

    Returns:
        yt.DataObject: Object that represents the area covered by the circumgalactic medium.
    """

    center = get_center(which_halo, dir)

    if outside_value is None:
        _, vir_radius, _ = foggie_virial_quantities(which_halo, dir)
        outside_value = vir_radius

    # Define a general test area of the cgm
    inner_sphere = ds.sphere(center, frac_cutout*outside_value)
    outer_sphere = ds.sphere(center, outside_value) 
    cgm_object = outer_sphere - inner_sphere

    return cgm_object

def foggie_virial_quantities(which_halo, dir):
    resource_path  = pkg_resources.files("my_cgm_helpers.src") / "foggie_virial_parameters.pkl"
    with open(resource_path, 'rb') as handle:
        open_file  = pickle.load(handle)
        data = open_file[which_halo][dir]
        M_vir = data["M_vir"] * yt.units.solMass
        R_vir = data["R_vir"] * yt.units.kpc
        T_vir = data["T_vir"] * yt.units.K
        return M_vir, R_vir, T_vir

def get_important_ions():
    """
    Provides information necessary to extract ion information.

    Parameters:
        None

    Returns:
        list[str], list[str]: Two lists of strings.
    """

    important_ion_list    = ['H I'                , 'Mg II'               , 'Si II'               , 'Si III'              , 'Si IV'               , 'C III'              , 'C IV'               , 'N V'                , 'O VI'               ] 
    important_ion_density = ["H_p0_number_density", "Mg_p1_number_density", "Si_p1_number_density", "Si_p2_number_density", "Si_p3_number_density", "C_p2_number_density", "C_p3_number_density", "N_p4_number_density", "O_p5_number_density"]

    Observe_Values = {"H_p0_number_density"  : [12.5, 24], 
                      "Mg_p1_number_density" : [12.5, 20],  # 12.2 but for consistency for with O VI mark 12.5
                      "Si_p1_number_density" : [11.5, 20], 
                      "Si_p2_number_density" : [11.5, 20],
                      "Si_p3_number_density" : [12  , 20], 
                      "C_p2_number_density"  : [12.5, 20],
                      "C_p3_number_density"  : [12.5, 20], 
                      "N_p4_number_density"  : [13  , 20],
                      "O_p5_number_density"  : [12.5, 20]} 
    
    Observable_Values = p.DataFrame(Observe_Values)

    return important_ion_list , important_ion_density, Observable_Values  

def condense_titles(ax, title, title_x, title_y, fsize = 14):
    """
    Set axes titles.

    Parameters:
        title (str): Overall title of the plot.
        title_x (str): X axis title.
        title_y (str): Y axis title.

    Returns:
        None
    """
    ax.set_title(title, fontsize = int(1.5*fsize))
    ax.set_xlabel(title_x, fontsize = int(1.25*fsize)) #, labelpad = 0)
    ax.set_ylabel(title_y, fontsize = int(1.25*fsize)) #, labelpad = -3)
    ax.tick_params(axis='x', labelsize = fsize)
    ax.tick_params(axis='y', labelsize = fsize)

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, other = float_str.split("e")
        exponent, units = other.split(" ") 
        return r"${0} \times 10^{{{1}}}$ {2}".format(base, int(exponent), units)
    else:
        return float_str

def transform_points(ds, val, data):

    # x,y,z datasets
    x_ls = data[('grid', 'x')].to("code_length") 
    y_ls = data[('grid', 'y')].to("code_length") 
    z_ls = data[('grid', 'z')].to("code_length") 

    # Pull out points
    x_p = x_ls[val].value
    y_p = y_ls[val].value
    z_p = z_ls[val].value

    # Define position vector
    P = [x_p, y_p, z_p] *ds.units.code_length
    return P

def units_projected(): 
    
    return {
            "density": 'g/cm**2', 
            "Mg_p1_number_density": 'cm**(-2)',  # Mg II
            "C_p3_number_density": 'cm**(-2)',   # C IV
            "O_p5_number_density": 'cm**(-2)'    # O VI
            }   

def z_labels(RD_num):
    if RD_num == "RD0016": 
        return r"$\mathrm{z=3.0}$"
    elif RD_num == "RD0020":
        return r"$\mathrm{z=2.0}$"
    elif RD_num == "RD0027":
        return r"$\mathrm{z=1.0}$"
    elif RD_num == "RD0032":
        return r"$\mathrm{z=0.5}$"
    elif RD_num == "RD0042":
        return r"$\mathrm{z=0.0}$"



def get_ion_densities():
    """
    Returns the corresponding yt field name for a given ion.

    Parameters:
        ion list[str]: A string representing the ion (e.g., "Mg II", "C IV").   
    returns:
        str: The corresponding yt field name for the given ion.
    """

    ion_density_mapping = {
        "H I": "H_p0_number_density",
        "Mg II": "Mg_p1_number_density",
        "Si II": "Si_p1_number_density",
        "Si III": "Si_p2_number_density",
        "Si IV": "Si_p3_number_density",
        "C II": "C_p1_number_density",
        "C III": "C_p2_number_density",
        "C IV": "C_p3_number_density",
        "N V": "N_p4_number_density",
        "O VI": "O_p5_number_density"
    }
    return ion_density_mapping

def ion_color(field):
    if field == "density":
        return "darkorange"
    elif field == "Mg_p1_number_density":
        return "cornflowerblue"
    elif field == "C_p3_number_density":
        return "green"
    elif field == "O_p5_number_density":
        return "hotpink"

def ion_label(field):
    if field == "density":
        return r"$\mathrm{N_{gas}}$"
    elif field == "Mg_p1_number_density":
        return r"$\mathrm{N_{Mg~II}}$"
    elif field == "C_p3_number_density":
        return r"$\mathrm{N_{C~IV}}$"
    elif field == "O_p5_number_density":
        return r"$\mathrm{N_{O~VI}}$"



def pdf_draw(figures, pdfname):
    """Save a sequence of matplotlib figures into a single PDF file.
    Parameters:
        figures (iterable[matplotlib.figure.Figure]): An iterable of matplotlib `Figure` objects to be written to the PDF, in order.
        pdfname (str): Output file path for the generated PDF. If the file exists it will be overwritten.
    Returns:
        None: The function writes the PDF to disk as a side effect.
    """
    with PdfPages(pdfname) as pdf:
        for fig in figures:
            pdf.savefig(fig)
