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
  output_dir: ##Directory path for the downloaded HSC
  star_sql: ##location of the star sql
  patchqa_sql: ##location of the imaging property sql

target_selection:
  pfstarget: ##path to the pfstarget repository
  random_sql: ## location of the random sql
  galaxy_sql: ## location of the galaxy sql
```
3. Download stars, galaxies, randoms and path information using bin/hscReleaseQuery.py
   You will need a [STARS](https://stars2.naoj.hawaii.edu/) account for this.

```bash 
python3 bin/hscReleaseQuery.py -u YOUR_STARS_ID -c configs/my_config.yaml --kind star
python3 bin/hscReleaseQuery.py -u YOUR_STARS_ID -c configs/my_config.yaml --kind galaxy
python3 bin/hscReleaseQuery.py -u YOUR_STARS_ID -c configs/my_config.yaml --kind random
python3 bin/hscReleaseQuery.py -u YOUR_STARS_ID -c configs/my_config.yaml --kind patchqa
```