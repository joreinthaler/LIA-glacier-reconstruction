import os
import numpy as np
from osgeo import gdal, osr
import whitebox

# 1. SETUP & CONFIGURATION
base_dir = r'D:/Carrivick_2026/Carrivick_3/TEST3_for_JR/'
target_epsg = 32606

#  inputFiles
points_shp   = os.path.join(base_dir, 'LIA_points.shp')             # All LIA points for interpolation (def value "LIA_elev")
outlines_shp = os.path.join(base_dir, 'LIA_outlines.shp')   # LIA outlines
alos_dem     = os.path.join(base_dir, 'DEM.tif')          # DEM

#output
interp_raw   = os.path.join(base_dir, 'interp_temp.tif')
interp_clip  = os.path.join(base_dir, 'interp_clipped.tif')
alos_resampled = os.path.join(base_dir, 'alos_matched.tif')
final_output = os.path.join(base_dir, 'LIA_surface.tif') #Final LIA surface. Optional smoothing can be applied afterwards.

wbt = whitebox.WhiteboxTools()
wbt.set_working_dir(base_dir)

# ================= HELPER FUNCTIONS =================

def assign_crs(raster_path, epsg):
    """Force assigns a CRS to a raster that lost its metadata."""
    ds = gdal.Open(raster_path, gdal.GA_Update)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)
    ds.SetProjection(sr.ExportToWkt())
    ds = None
    print(f"Assigned EPSG:{epsg} to {raster_path}")

def align_and_masked_max(interp_path, alos_path, output_path):
    """
    1. Resamples ALOS to match Interpolation (50m)
    2. Calculates Max only inside the polygons (Masked)
    """
    print(f"Aligning ALOS DEM to match the new 50m grid...")
    
    # Open the 50m interpolation to get its metadata
    ref_ds = gdal.Open(interp_path)
    gt = ref_ds.GetGeoTransform()
    
    # Calculate Bounding Box
    x_min, y_max = gt[0], gt[3]
    x_max = x_min + (ref_ds.RasterXSize * gt[1])
    y_min = y_max + (ref_ds.RasterYSize * gt[5])
    bounds = [x_min, y_min, x_max, y_max]

    # --- STEP 1: RESAMPLE ALOS ---
    # We use a temporary filename to ensure we aren't reading an old 100m file
    temp_alos = os.path.join(base_dir, 'alos_50m_temp.tif')
    
    gdal.Warp(temp_alos, alos_path, 
              dstSRS=f'EPSG:{target_epsg}',
              outputBounds=bounds,
              xRes=gt[1], yRes=gt[5],
              width=ref_ds.RasterXSize,
              height=ref_ds.RasterYSize,
              resampleAlg='max')

    # --- STEP 2: MASKED MAX MATH ---
    print("Calculating final surface (Masked Max)...")
    ds_int = gdal.Open(interp_path)
    ds_alos = gdal.Open(temp_alos)
    
    arr_int = ds_int.ReadAsArray().astype(np.float32)
    arr_alos = ds_alos.ReadAsArray().astype(np.float32)
    
    # Check if shapes match now (Safety Check)
    if arr_int.shape != arr_alos.shape:
        print(f"CRITICAL ERROR: Shapes still mismatch! {arr_int.shape} vs {arr_alos.shape}")
        return

    # Standardize NoData
    arr_int[arr_int <= -9999] = np.nan
    arr_alos[arr_alos <= -9999] = np.nan
    
    # Create the cookie-cutter mask
    glacier_mask = ~np.isnan(arr_int)
    result = np.full(arr_int.shape, -9999, dtype=np.float32)
    
    # Apply logic only inside polygons
    result[glacier_mask] = np.fmax(arr_int[glacier_mask], arr_alos[glacier_mask])
    
    # --- STEP 3: SAVE ---
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, ds_int.RasterXSize, ds_int.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds_int.GetGeoTransform())
    out_ds.SetProjection(ds_int.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(result)
    out_ds.GetRasterBand(1).SetNoDataValue(-9999)
    out_ds = None 
    
    print(f"Success! Final 50m LIA surface saved: {output_path}")

# ================= EXECUTION =================
# 2. INTERPOLATE
print("Step 1: Interpolating...")
wbt.natural_neighbour_interpolation(i=points_shp, field="LIA_elev", output=interp_raw, cell_size=50.0)

# 3. FIX CRS & CLIP
assign_crs(interp_raw, target_epsg)

print("Step 2: Clipping...")
gdal.Warp(interp_clip, interp_raw, cutlineDSName=outlines_shp, cropToCutline=True, dstNodata=-9999)

# 4. FINAL STEP: ALIGN & MASKED MAX
if os.path.exists(interp_clip):
    align_and_masked_max(interp_clip, alos_dem, final_output)
else:
    print("Error: The clipped interpolation file was not found.")