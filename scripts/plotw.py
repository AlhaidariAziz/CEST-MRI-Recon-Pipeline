# 3D windowed Plot
from ipywidgets import interact, FloatSlider, IntSlider
import matplotlib.pyplot as plt
import numpy as np
import h5py

def plotw(volume, title="Recon Viewer", rotate=True):
    # 1. Move to CPU and handle complex data
    if hasattr(volume, 'get'): 
        volume = volume.get()
    
    # 2. Rotation to match Siemens Orientation (Optional but recommended)
    # This aligns with the 'R >> L' phase encoding in your protocol
    if rotate:
        volume = np.rot90(volume, k=1, axes=(1, 2))
    
    volume_abs = np.abs(volume)
    
    # 3. CRITICAL: Remove NaNs/Infs for statistics
    clean_data = volume_abs[np.isfinite(volume_abs)]
    
    if clean_data.size == 0:
        print("Error: All values in volume are NaN or Inf.")
        return

    # 4. Calculate robust display limits
    # 99th percentile prevents "bright spot" artifacts from ruining contrast
    max_val = float(np.percentile(clean_data, 99))
    min_val = float(np.percentile(clean_data, 1)) # Suggested noise floor
    
    if np.isnan(max_val) or max_val <= 0:
        max_val = 1.0 
        
    def update_view(slice_idx, v_min, v_max):
        # Prevent v_min from being higher than v_max
        if v_min >= v_max:
            v_min = v_max - (v_max * 0.01)

        plt.figure(figsize=(10, 8))
        # vmin and vmax control the windowing
        plt.imshow(volume_abs[slice_idx, :, :], cmap='gray', vmin=v_min, vmax=v_max)
        plt.title(f"{title} | Slice: {slice_idx}")
        plt.colorbar(label='Intensity')
        plt.axis('off')
        plt.show()

    # 5. Create sliders for both min (black level) and max (white level)
    interact(update_view, 
             slice_idx=IntSlider(min=0, max=volume.shape[0]-1, step=1, value=volume.shape[0]//2),
             v_min=FloatSlider(min=0, max=max_val, step=max_val/100, value=0, description='Min (Black)'),
             v_max=FloatSlider(min=max_val/100, max=max_val*3, step=max_val/100, value=max_val, description='Max (White)'))


DATA_DIR='/home/vault/iwbi/iwbi112h/CEST_Data/'
file='CEST_GRAPPA_recons_65.h5'
RO=224
REP=64
imgs=[]
with h5py.File(DATA_DIR+file,'r') as f:
    for RO_idx in range(RO): 
        imgs.append(f[f'CEST_recon_RO_idx_{RO_idx}'][REP,:,:])

imgs=np.array(imgs)
imgs=np.permute_dims(imgs,(1,2,0))
print('imgs shape:',imgs.shape)
plotw(imgs)