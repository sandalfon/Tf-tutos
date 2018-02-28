# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:56:31 2018

@author: Sandalfon
"""

from options import Options
from data_dl import DataDl
from data import Data

options=Options()


options.loadFromJson('options.json')
dataDl=DataDl()
dataDl.dl_end_extract(options)

Data().batches(options, train_logical=True)