#pip install s3fs kerchunk zarr rioxarray geocube 

import sys
import subprocess

# # implement pip as a subprocess:
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 's3fs'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kerchunk'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'zarr'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rioxarray'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'geocube'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dask'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'distributed'])

import re
import dask
import numpy
import xarray
import pyproj
import pandas
import requests
import geopandas
from dask.distributed import Client, LocalCluster
from dask.distributed import progress
import zarr
import fsspec
from pyproj import Transformer
from s3fs import S3FileSystem
from kerchunk.combine import MultiZarrToZarr
import rioxarray
import geocube
import pandas as pd
from geocube.api.core import make_geocube

def perform_zonal_computation(ds, cat_id):

    # subset by catchment id
    ds_catchment = ds.where(ds.cat==cat_id, drop=True)
    
    delayed = []
    # loop over variables   
    for variable in ['RAINRATE']:
                
        delay = dask.delayed(compute_zonal_mean)(ds_catchment[variable], variable)
        delayed.append(delay)
        
    res = dask.compute(*delayed)
    
    # combine outputs (list of dicts) into a single dict.
    res = {k: v for d in res for k, v in d.items()}
    
    # return results
    return {f'cat-{int(cat_id)}': res}

def compute_zonal_mean(ds, variable):
    return {variable: ds.mean(dim=['x','y']).values}


def compute_avg_p(client, ds, catchment_ids, year, month_s, month_e):
    
    # define the start and end time of the data we want to use
    start_time = f'{year}-{month_s}-01 00:00'
    if month_s == '12':
        end_time = f'{year}-{month_e}-31 23:00'
    else:
        end_time = f'{year}-{month_e}-01 00:00'
    
    # isolate the desired time period of our data
    ds_subset = ds.sortby('time').sel(time=slice(start_time, end_time))
    print(f'The dataset contains {len(ds_subset.time)} timesteps')
    
    ds_subset = ds_subset.chunk(chunks={'time': 1000})
    ds_subset = ds_subset.drop(['lat','lon'])
    
    # compute
    print('starting the first computation: ')
    ds_subset = ds_subset.compute()
    print('... Finished.')
    
    scattered_ds = client.scatter(ds_subset, broadcast=True)

    delayed = []
    # loop over each catchment in our domain
    # create delayed tasks to compute zonal mean
    for cat_id in catchment_ids:
        delay = dask.delayed(perform_zonal_computation)(scattered_ds, cat_id)
        delayed.append(delay)
        
    # run the computation
    print('starting the second computation: ')
    results = dask.compute(*delayed)
    print('... Finished.')
    
    
    # compute the date range for our data using start and end times
    # that were used in the subsetting process.
    dates = pandas.date_range(start_time, end_time, freq="60min")

    # save the zonal means for each catchment
    for dat in results:
        for cat in dat:
            df = pandas.DataFrame({k:list(v) for k,v in dat[cat].items()})
            
            # add the time index
            if ds_subset.time.shape[0]==dates.shape[0]:
                df['time'] = dates
            else:
                print('Some records are not available:')
                df['time'] = pd.to_datetime(ds_subset.time.values)
                
            df.set_index('time', inplace=True)

            # write to file
            with open(f'./{year}{month_s}_{cat}.csv', 'w') as f:
                df.to_csv(f)


    # print the list of unavailable data
    z = list(set(dates) - set(pd.to_datetime(ds_subset.time.values)))
    print('Unavailable data: ', *z, sep='\n')

