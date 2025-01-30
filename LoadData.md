# Reading instructions for Temperature12k files and serializations

## Reading LiPD files directly

### LiPD files
Temp12k_v1_0_0_LiPD.zip need to be unzipped, or use the directory "Temp12k_directory_LiPD_files", then individual LiPD files may be accessed via the [LiPD utilities](https://github.com/nickmckay/LiPD-utilities)

## Loading Serializations

### Matlab
`load(Temp12k_v1_0_0.mat)`
(should be compatible with version >6)
code to reproduce the figures of the Descriptor is available on [GitHub](https://github.com/nickmckay/Temperature12k/)

### Python
`T12k = pickle.load(open("Temp12k_v1_0_0.pkl","rb"))`

The following [Jupyter notebooks](https://github.com/LinkedEarth/notebooks/tree/master/PAGES2k) illustrate how to use the LiPD files to perform various tasks in Python

### R
`load('PAGES2k_v2.0.0.RData')`

Sourced from https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/climate12k/temperature/version1.0.0/
Reference documentation at https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/noaa-recon-27330/html
