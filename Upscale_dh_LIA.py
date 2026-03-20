import numpy as np
import os
from osgeo import gdal
from scipy.stats import linregress
import pandas as pd

# Skript to canlculate LIA surface, within modern extent by upscaling recent elevation changes using glacier-specific elevation change gradients.
# THe scaling factor used for the calculation is printed during the processing.
# The output has than to be combined with LIA outline points to create the full LIA surface elevation. Additionaly a csv file is created with glacier parameters. 

# Literature : https://doi.org/10.1016/j.geomorph.2024.109321, https://doi.org/10.5194/tc-19-753-2025


# ================= CONFIGURATION =================
work_dir = '' #define working dictionary

# Input Files
dem_path       = os.path.join(work_dir, 'DEM.tif')                      # (modern DEM) DEM_align 
dh_path        = os.path.join(work_dir, 'dhdt.tif')                     # Modern dh (e.g. Hugonnet et al. 2021) [m/y]
rgi_path       = os.path.join(work_dir, 'ID.tif')                       # Glacier IDs (Raster with IDs)
dh_factor_path = os.path.join(work_dir, 'dh_factor.tif')                # Preprocessed scaling factor raster (calculate by dhLIA_simple/dhdt recent)

#Output Files
out_dh_pred   = os.path.join(work_dir, 'dh_predicted_eq6.tif')          # predicted elevation change since LIA within modern boundaries
out_lia_surf  = os.path.join(work_dir, 'LIA_surface_reconstructed.tif') # predicted LIA elevation LIA within modern boundaries
out_params    = os.path.join(work_dir, 'glacier_mean_elevation.tif')    #

# ================= HELPER FUNCTIONS =================

def read_raster(path):
    ds = gdal.Open(path)
    if ds is None: raise FileNotFoundError(f"Could not open {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    # Standardize NoData to NaN
    arr[arr < -9999] = np.nan
    arr[np.isclose(arr, -3.40282e+38)] = np.nan
    # CRITICAL: Handle infinity values that cause 'inf' results in means
    arr[np.isinf(arr)] = np.nan
    return ds, arr

def save_geotiff(filename, data_arr, ref_ds):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data_arr.shape
    out_ds = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    band = out_ds.GetRasterBand(1)
    save_arr = np.where(np.isnan(data_arr), -9999, data_arr)
    band.WriteArray(save_arr)
    band.SetNoDataValue(-9999)
    band.FlushCache()
    print(f"Saved: {filename}")

# ================= 1. LOAD DATA =================
print("Reading rasters...")
ds_dem, dem_arr = read_raster(dem_path)
ds_dh, dh_arr   = read_raster(dh_path)
ds_rgi, rgi_arr = read_raster(rgi_path)
ds_fac, fac_arr = read_raster(dh_factor_path)

# Calculate Area Factor (m^2 to km^2)
gt = ds_dem.GetGeoTransform()
pixel_area_km2 = abs(gt[1] * gt[5]) / 1e6

# Calculate global median from dh_factor (excluding NaNs and Infs)
valid_fac = fac_arr[~np.isnan(fac_arr)]
median_scaling_factor = np.median(valid_fac) if len(valid_fac) > 0 else 1.0
print(f"Global Median Scaling Factor: {median_scaling_factor:.4f}")

# ================= 2. REGRESSION & METRICS LOOP =================
print("Analyzing glaciers...")
shape = dem_arr.shape
out_a1, out_b1 = np.full(shape, np.nan), np.full(shape, np.nan)
out_a2, out_b2 = np.full(shape, np.nan), np.full(shape, np.nan)
out_me = np.full(shape, np.nan)

glacier_results = []
# Ensure background is handled
rgi_arr[rgi_arr == 0] = np.nan
unique_ids = np.unique(rgi_arr[~np.isnan(rgi_arr)])

for gid in unique_ids:
    mask = (rgi_arr == gid)
    
    # Basic Metrics
    z_glacier = dem_arr[mask & ~np.isnan(dem_arr)]
    if len(z_glacier) < 5: continue
    
    z_min, z_max, z_mean = np.nanmin(z_glacier), np.nanmax(z_glacier), np.nanmean(z_glacier)
    area = np.sum(mask) * pixel_area_km2
    
    # Regression
    valid_regr = mask & (~np.isnan(dem_arr)) & (~np.isnan(dh_arr))
    s1, i1, s2, i2 = np.nan, np.nan, np.nan, np.nan
    
    if np.sum(valid_regr) > 10:
        curr_dem = dem_arr[valid_regr]
        curr_dh = dh_arr[valid_regr]
        
        m_abl = curr_dem <= z_mean
        m_acc = curr_dem > z_mean
        
        if np.sum(m_abl) > 2:
            s1, i1, _, _, _ = linregress(curr_dem[m_abl], curr_dh[m_abl])
            out_a1[mask], out_b1[mask] = s1, i1
        if np.sum(m_acc) > 2:
            s2, i2, _, _, _ = linregress(curr_dem[m_acc], curr_dh[m_acc])
            out_a2[mask], out_b2[mask] = s2, i2
            
    out_me[mask] = z_mean
    
    glacier_results.append({
        'Glacier_ID': gid,
        'Area_km2': area,
        'Z_Min': z_min,
        'Z_Max': z_max,
        'Z_Mean': z_mean,
        'Slope_Abl_a1': s1,
        'Intercept_Abl_b1': i1,
        'Slope_Acc_a2': s2,
        'Intercept_Acc_b2': i2,
        'Global_Median_dh_factor': median_scaling_factor
    })

# ================= 3. RECONSTRUCTION =================
print("Calculating surfaces...")
dh_predict = np.full(shape, np.nan)
calc_mask = ~np.isnan(out_me) & ~np.isnan(dem_arr)

# Equation 6
m_abl_idx = calc_mask & (dem_arr <= out_me)
dh_predict[m_abl_idx] = (out_a1[m_abl_idx] * dem_arr[m_abl_idx]) + out_b1[m_abl_idx]

m_acc_idx = calc_mask & (dem_arr > out_me)
dh_predict[m_acc_idx] = (out_a2[m_acc_idx] * dem_arr[m_acc_idx]) + out_b2[m_acc_idx]

# Equation 7
lia_surface = dem_arr - (dh_predict * median_scaling_factor)
lia_surface[np.isnan(rgi_arr)] = np.nan

# Final step: Calculate Mean_dh_predict per glacier for the table
for res in glacier_results:
    gid = res['Glacier_ID']
    vals = dh_predict[rgi_arr == gid]
    res['Mean_dh_predict'] = np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan

# ================= 4. SAVE OUTPUTS =================
save_geotiff(out_dh_pred, dh_predict, ds_dem)
save_geotiff(out_lia_surf, lia_surface, ds_dem)
save_geotiff(out_params, out_me, ds_dem)

df = pd.DataFrame(glacier_results)
df.to_csv(out_csv, index=False)
print(f"Results table saved to: {out_csv}")
print("Done.")