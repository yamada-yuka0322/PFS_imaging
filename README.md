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
2. Make a config file under config/ directory to specify the
```bash 
hsc:
  target_dir: ##Directory path for the CO targets
  random_dir: ##Directory path for the downloaded randoms
  star_dir: ##Directory path for the downloaded stars
  star_sql: ##location of the star sql
  patchqa_sql: ##location of the imaging property sql

target_selection:
  pfstarget: ##path to the pfstarget repository
  random_sql: ## location of the random sql
  galaxy_sql: ## location of the galaxy sql

Gaia:
  output_dir: ##Directory path for the downloaded Gaia stars
  bsmask_dir: ##Directory path to generate updated bright star masks for the galaxies and the randoms

Imaging:
  imaging_dir: ##path to save the imaging properties of each healpixels
  weights_dir: ##path to save the nn_weights
```
3. Download stars, galaxies, randoms and path information using bin/hscReleaseQuery.py
   You will need a [STARS](https://stars2.naoj.hawaii.edu/) account for this.

```bash 
python3 hscReleaseQuery.py -u YOUR_STARS_ID -c ../configs/my_config.yaml --kind star
python3 hscReleaseQuery.py -u YOUR_STARS_ID -c ../configs/my_config.yaml --kind galaxy
python3 hscReleaseQuery.py -u YOUR_STARS_ID -c ../configs/my_config.yaml --kind random
python3 hscReleaseQuery.py -u YOUR_STARS_ID -c ../configs/my_config.yaml --kind patchqa
```

4. In order to apply the updated stellar mask, you need to download the Gaia stars using bin/GetGaiaStars.py
```bash 
python3 GetGaiaStars.py -c ../configs/my_config.yaml
```
and generate Bright Stellar mask for the galaxies, randoms and stars.
```bash 
python3 GenerateStarMask.py -c ../configs/my_config.yaml --kind star
python3 GenerateStarMask.py -c ../configs/my_config.yaml --kind galaxy
python3 GenerateStarMask.py -c ../configs/my_config.yaml --kind random
```

## Calculating the imagingy systematic weights.
1. You can calculate the imaging properties per pixels using get_imaging_property() function in imaging.py.
An example is written in nb/GetImagingProperty.ipynb

2. The imaging systematic weights can be calculated using train.py and Weights.py in bin/ file.
An example is written in nb/ImagingWeights.ipynb