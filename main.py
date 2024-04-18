import astropy.io.ascii
import numpy as np
from matplotlib import pyplot as plt

data = astropy.io.ascii.read('./ZTF_flares.dat') # data from of van Velzen et al. 2019
# print(data)
# print(len(data))
# print('# unknown sources', sum(data['classification']=='None'))

AGN_sources = []
AGN_sources_indices = []
SN_sources = []
SN_sources_indices = []
Unknown_sources = []
Unknown_sources_indices = []

i = 0
for source in data['classification']:
    if source == 'AGN':
        AGN_sources.append(source)
        AGN_sources_indices.append(i)
    if 'SN' in source:
        SN_sources.append(source)
        SN_sources_indices.append(i)
    if source == 'None':
        Unknown_sources.append(source)
        Unknown_sources_indices.append(i)
    i += 1
# print(len(data[AGN_sources_indices]) + len(data[Unknown_sources_indices]) + len(data[SN_sources_indices]))

hist_stack_data = [data['offset_wmean'][AGN_sources_indices], data['offset_wmean'][SN_sources_indices], data['offset_wmean'][Unknown_sources_indices]]
stack_label = ['AGN', 'SNe', 'Unknown']
plt.figure()
plt.title('Weighted mean offset for different objects')
plt.hist(hist_stack_data, bins='auto', label = stack_label, stacked = True)
plt.xlabel('Mean offset [arcsec]')
plt.ylabel('Number of counts')
# plt.hist(data['offset_mean'][SN_sources_indices], bins = 'auto', label = 'SN sources')
# plt.hist(data['offset_mean'][Unknown_sources_indices], bins = 'auto', label = 'Unknown sources')
plt.legend()
plt.show()