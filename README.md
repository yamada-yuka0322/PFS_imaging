# PFS_imaging
This package contains scripts for the PFS Cosmology Survey imaging systematics weights calculation.

## Installation 
```bash
git clone https://github.com/yamada-yuka0322/PFS_imaging.git
cd PFS_imaging
pip install -e . 
```


## getting started
1. You first need to install pfstarget repository
```bash 
git clone https://github.com/pfs-cosmo/pfstarget.git
```

2. Download  dust maps in `src/pfsimaging/dat`.

On `idark`, you can or establish a symlinks to avoid downloading existing data
```bash
# csfd desi dust map
ln -s /lustre/work/jingjing.shi/pfs_co_fa/data_raw/dustmaps/CSFD_DESI_merged_dust_map_NS2048_ring--Equatorial.fits src/pfsimaging/dat/CSFD_DESI_merged_dust_map_NS2048_ring--Equatorial.fits

# csfd dust map 
ln -s /lustre/work/jingjing.shi/pfs_co_fa/data_raw/dustmaps/CSFD_DESI_merged_dust_map_NS2048_ring.fits src/pfsimaging/dat/CSFD_DESI_merged_dust_map_NS2048_ring.fits
```

3. Make a config file under config/ directory to specify the
```bash 
data:
  random_dir: ##Directory path for the downloaded HSC randoms (with the same mask as pfs co targets)
  random_dir_masked: ##Directory path for the downloaded HSC randoms inside the masked region
  target_file:  ##path to the pfs co targets (output of pfstarget)
  patchqa_dir:  ##Directory path for the downloaded HSC patchqa

bsmask:
  random_bsmask:  #path to the random bright star mask
  target_bsmask:  #path to the target bright star mask
  
bgmask:
  random_bgmask: #path to the random bright galaxy mask
  target_bgmask: #path to the target bright star mask
 
Gaia:
    bs: ##Directory path for the downloaded Gaia stars (to generate updated bright stellar mask)
    star: ##Directory path for the downloaded Gaia stars (to calculate the stellar density)

out_dir:
  imaging: #Directory path to save the imaging properties
  optuna: #Directory path to save the optuna output
  weights: #Directory path to store the imaging systematic weights
  
tractlist: #path to the csv file with the entire tractId
```
4. Download Gaia stars to apply updated bright stellar mask
```bash 
python get_gaia_bs.py -c ../configs/my_config.yaml
```
The downloaded stellar catalog will be saved under config["Gaia"]["bs"] directory.
You can apply the updated bright stellar mask for the targets and the randoms using
```bash 
python3 GenerateStarMask.py -c ../configs/my_config.yaml --kind galaxy
python3 GenerateStarMask.py -c ../configs/my_config.yaml --kind random
```

5. Download Gaia stars to estimate the stellar density
```bash 
python get_gaia_star.py -c ../configs/my_config.yaml
```
The downloaded stellar catalog will be saved under config["Gaia"]["star"] directory.

6. Download per-patch imaging attributes information using bin/hscReleaseQuery.py
   You will need a [STARS](https://stars2.naoj.hawaii.edu/) account for this.

```bash 
python hscReleaseQuery.py -u YOUR_STARS_ID -c ../configs/my_config.yaml --kind patchqa
```
The downloaded catalog will be saved under config["data"]["patchqa_dir"]


## Calculating the imagingy systematic weights.
1. You can calculate the imaging properties per pixels using bin/calculate_imaging.py
```bash 
python calculate_imaging.py --config ../configs/my_config.yaml --dustmap csfd_desi --fields autumn spring hectomap -c
```
This will calculate imaging property per pixel as well as the target density and the effective area. The imaging properties would be saved in a fits file under config["out_dir"]["imaging"] and also a combined catalog for the full field saved in "all_property.fits". By specifying attribute -c, the cleaned version in which pixel with nan or 0 effective area removed would be saved in ~_cleaned.fits.


2. In order to calculate the Neural Network weights, you can run bin/train_nn.py to train your model.
```bash 
python train_nn.py --config ../configs/my_config.yaml --trial 200 --save_n 5 -rn test
```
This will run optuna to optimize the nueral network with --trial number of trials. The models would be saved under config["out_dir"]["optuna"]/run_name/ directory, with the best model saved in best_model.pt together with the top --save_n trials.

3. bin/calculate_weights.py calculates the imaging systematic weights for each pixel using linear, quadratic and neural network model. This can be run as
```bash 
python calculate_weights.py --config ../configs/my_config.yaml -m lin quad nn -rn test
```
Be sure that the run name matches the run name you specified when you ran bin/train_nn.py. This would generate fits file with the different weight model under config["out_dir"]["weights"]
