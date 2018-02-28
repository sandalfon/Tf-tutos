# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:20:23 2018

@author: Sandalfon
"""
import os
import tarfile
import sys
import requests
import math

from tqdm import tqdm

class DataDl(object):
    
#    def __init__(self):
#        options =self.download()
#        options = self.extract(options)
        
    def download(self,options):
        path_dl  = options.getDataOptions('dl_dir')
        filename  = options.getDataOptions('filename')
        url = options.getDataOptions('url')
        force_dl = options.getDataOptions('force_dl')
        if not os.path.exists(path_dl):
             os.makedirs(path_dl)
        path_file = os.path.join(path_dl,filename)
        url_file = url+"/"+filename
        if os.path.isfile(path_file) and not force_dl:
            print("File exist " + path_file)
            return path_file
        r = requests.get(url_file, stream =True)
        total_size = int(r.headers.get('content-length', 0)); 
        block_size = 1024
        wrote = 0 
        with open(path_file, 'wb') as f:
            for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
                wrote = wrote  + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            print("ERROR, something went wrong")  
            return None
        return path_file


    
    def extract(self, options):
        path_dl  = options.getDataOptions('dl_dir')
        filename  = options.getDataOptions('filename')
        path_file = os.path.join(path_dl,filename)
        root = options.getDataOptions('data_dir')
        force_extract = options.getDataOptions('force_extract')
        if os.path.isdir(root) and not force_extract:
    # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, path_file))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(path_file)
            sys.stdout.flush()
            tar.extractall(root)
            tar.close()
            
    def dl_end_extract(self,options):
        self.download(options)
        self.extract(options)