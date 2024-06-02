import os
from pathlib import Path

import numpy as np
import h5py
from histpy import Histogram
import healpy as hp

from cosipy import BinnedData
from cosipy.util import fetch_wasabi_file

# FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FILE_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/44Ti')
DATA_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/data')
WASABI_DIR = Path('COSI-SMEX/DC2')

def FileExists(datapath, filename='Ti44_CasA_3months_unbinned_data.fits', wasabi_path=WASABI_DIR / 'Data/Sources/'):
    # Does datapath exist?
    if datapath.is_dir():
        print(f'Given datapath exists')
    else:
        print(f'Given datapath does not exist. Aborting file check.')
        return 1
    
    # Does file exist?
    if (datapath / filename).is_file():
        print(f'{filename} exists in datapath.')

    elif (datapath / f'{filename}.gz').is_file():
        print(f'{filename} does not exist in file path but a .gz compressed version does. Untarring...')
        os.system('gunzip ' + str(datapath / f'{filename}.gz'))

    elif (datapath / f'{filename}.zip').is_file():
        print(f'{filename} does not exist in file path but a .zip compressed version does. Unzipping...')
        os.system('gunzip ' + str(datapath / f'{filename}.zip'))

    else:
        print(f'{filename} does not exist in datapath. Fetching from wasabi.')
        fetch_wasabi_file(str(wasabi_path / f'{filename}.gz'), output=str(datapath / f'{filename}.gz'))
        os.system('gunzip ' + str(datapath / f'{filename}.gz'))

    print()
    return 0

def FileCheck():
    # Checking source files
    FileExists(datapath=DATA_DIR, filename='Ti44_CasA_3months_unbinned_data.fits', 
              wasabi_path=WASABI_DIR / 'Data/Sources/')
    FileExists(datapath=DATA_DIR, filename='Ti44_G1903_3months_unbinned_data.fits', 
              wasabi_path=WASABI_DIR / 'Data/Sources/')
    FileExists(datapath=DATA_DIR, filename='Ti44_SN1987A_3months_unbinned_data.fits', 
              wasabi_path=WASABI_DIR / 'Data/Sources/')
    FileExists(datapath=DATA_DIR, filename='Ti44_SNsurprise_3months_unbinned_data.fits', 
              wasabi_path=WASABI_DIR / 'Data/Sources/')
    
    # Checking background file
    FileExists(datapath=DATA_DIR, filename='total_bg_3months_unbinned_data.fits', 
              wasabi_path=WASABI_DIR / 'Data/Backgrounds/')
    
    # Checking response file
    FileExists(datapath=DATA_DIR, filename='psr_gal_Ti44_E_1150_1164keV_DC2.h5',
               wasabi_path=WASABI_DIR / 'Responses/PointSourceReponse/')
    
    return 0

def GetBinnedData(config_file='input.yaml', parent_file=FILE_DIR / 'data/Ti44_CasA_3months_unbinned_data.fits', 
              output_file=FILE_DIR / 'data/Ti44_CasA_binned'):
    # Create BinnedData object, read unbinned 
    # data, and write binned data to disk
    binned_data = BinnedData(config_file)      # TODO: Keep input.yaml in folder
    binned_data.get_binned_data(parent_file, output_name=output_file)
    
    return 0

def Derived_FilesCheck():
    # Checking binned source files
    filename = 'data/Ti44_CasA_binned.hdf5'
    if not (FILE_DIR / filename).is_file():
        print(f'{filename} does not exist. Deriving from vanilla file.')
        GetBinnedData(parent_file = DATA_DIR / 'Ti44_CasA_3months_unbinned_data.fits', 
                  output_file = FILE_DIR / 'data/Ti44_CasA_binned')
    else:
        print(f'{filename} exists')
        print()
    
    filename = 'data/Ti44_G1903_binned.hdf5'
    if not (FILE_DIR / filename).is_file():
        print(f'{filename} does not exist. Deriving from vanilla file.')
        GetBinnedData(parent_file = DATA_DIR / 'Ti44_G1903_3months_unbinned_data.fits', 
                  output_file = FILE_DIR / 'data/Ti44_G1903_binned')
    else:
        print(f'{filename} exists')
        print()
        
    filename = 'data/Ti44_SN1987A_binned.hdf5'
    if not (FILE_DIR / filename).is_file():
        print(f'{filename} does not exist. Deriving from vanilla file.')
        GetBinnedData(parent_file = DATA_DIR / 'Ti44_SN1987A_3months_unbinned_data.fits', 
                  output_file = FILE_DIR / 'data/Ti44_SN1987A_binned')
    else:
        print(f'{filename} exists')
        print()
        
    filename = 'data/Ti44_SNsurprise_binned.hdf5'
    if not (FILE_DIR / filename).is_file():
        print(f'{filename} does not exist. Deriving from vanilla file.')
        GetBinnedData(parent_file = DATA_DIR / 'Ti44_SNsurprise_3months_unbinned_data.fits', 
                  output_file = FILE_DIR / 'data/Ti44_SNsurprise_binned')
    else:
        print(f'{filename} exists')
        print()
    
    # Checking binned background file
    filename = 'total_bg_binned_phi3.hdf5'
    if not (DATA_DIR / filename).is_file():
        print(f'{filename} does not exist. Deriving from vanilla file.')
        GetBinnedData(parent_file = DATA_DIR / 'total_bg_3months_unbinned_data.fits', 
                  output_file = DATA_DIR / 'total_bg_binned_phi3')
    else:
        print(f'{filename} exists')
        print()
        
    return 0

def FormattedResponse_FilesCheck():
    # Checking flattened response file
    response_file = 'psr_gal_flattened_Ti44_E_1150_1164keV_DC2.h5'
    if not (DATA_DIR / response_file).is_file():
        print(f'{response_file} flattened response file does not exist. Creating from raw file.')

        # Open parent file
        hf = h5py.File(DATA_DIR / 'psr_gal_Ti44_E_1150_1164keV_DC2.h5', 'r')
        group = hf['hist']
        dset = group['contents']

        # New file properties
        old_shape = dset.shape
        NUMROWS = np.prod(old_shape[:2] - 2)
        NUMCOLS = np.prod(old_shape[2:] - 2)
        new_shape = (NUMROWS, NUMCOLS)

        # Create flatted response file
        with h5py.File(DATA_DIR / response_file, 'w') as output_file:
            dset1 = output_file.create_dataset('response_matrix', data=np.transpose(dset[1:-1, 1, 1, 1:-1, 1:-1], (1,2, 0)).reshape(new_shape))
            print(dset1.shape)
            dset2 = output_file.create_dataset('response_vector', data=np.sum(dset1, axis=0))
            print(dset2.shape)

        # Close parent file
        hf.close()

    else:
        print(f'{response_file} flattened response file exists')
        print()

    return 0

def main():

    status = FileCheck()
    status = Derived_FilesCheck()
    status = FormattedResponse_FilesCheck()

if __name__ == "__main__":
    main()