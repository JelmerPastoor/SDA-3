import astropy.io.ascii
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


data = astropy.io.ascii.read('./ZTF_flares.dat') # data from of van Velzen et al. 2019
# print(data)
# print(len(data))
# print('# unknown sources', sum(data['classification']=='None'))
def histogram(data, label, title, xlabel, ylabel):
    plt.figure()
    plt.hist(data, bins='auto', label = label, edgecolor = 'black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


# def P_nuclear(r, sigma_xy):
#     integrand_func = lambda r: (np.sqrt(2 * np.pi) * r * np.random.normal(0, sigma_xy)) / sigma_xy
#     P_nuc, _ = sp.quad(integrand_func, 0, r)
#     return P_nuc
def P_nuclear(r, sigma_xy):
    P_nuc = (np.sqrt(2 * np.pi) * r * sp.stats.norm.pdf(r, loc = 0, scale = sigma_xy)) / sigma_xy
    return P_nuc

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
AGN = data[AGN_sources_indices]
SNe = data[SN_sources_indices]
Unknown = data[Unknown_sources_indices]
hist_stack_data = [AGN['offset_mean'], SNe['offset_mean'], Unknown['offset_mean']]
stack_label = ['AGN', 'SNe', 'Unknown']
# plt.figure()
# plt.title('Mean offset distribution for different objects')
# plt.hist(hist_stack_data, bins='auto', label = stack_label, stacked = True, edgecolor = 'black')
# plt.xlabel('Mean offset [arcsec]')
# plt.ylabel('Number of counts')
# plt.legend()
# plt.show()
# - Plot the offset distribution of the SN, AGN and unknown sources. Discuss the following two points:
#   - Describe the difference between the SN and nuclear flare offset distribution.
# histogram(data['offset_mean'][SN_sources_indices], 'Supernovae', 'Distribution of offset of supernovae',  'Offset [arcsec]', 'Number of SNe')
# histogram(data['offset_mean'][Unknown_sources_indices], 'Unknown objects', 'Distribution of offset of different unknown objects',  'Offset [arcsec]', 'Number of objects')
# histogram(data['offset_mean'][AGN_sources_indices], 'Active Galactic Nuclei', 'Distribution of offset of AGN',  'Offset [arcsec]', 'Number of AGN')

#   - Is the offset distribution of the unknown sources consistent with originating solely from the SN offset distribution? Or solely from the AGN offset distribution? (hint: there is a statistical test to quantify this)
SNe_unknown_pvalue = sp.stats.anderson_ksamp([SNe['offset_mean'], Unknown['offset_mean']])
print(f'We can use a hypothesis that reads: The offset distribution of the unknown sources is consistent with originating solely from the SN offset distribution. Using the Anderson Darling test, we find that this results in a p value of {SNe_unknown_pvalue[2]:.3f}, which is lower than 0.05, so we can reject this hypothesis.')
AGN_unknown_pvalue = sp.stats.anderson_ksamp([AGN['offset_mean'], Unknown['offset_mean']])
print(f'We can use a hypothesis that reads: The offset distribution of the unknown sources is consistent with originating solely from the AGN offset distribution. Using the Anderson Darling test, we find that this results in a p value of {AGN_unknown_pvalue[2]:.3f}, which is lower than 0.05, so we can reject this hypothesis.')

# Function for Pnuc
x_dist = np.random.normal(0, 1, 1000)
y_dist = np.random.normal(0, 1, 1000)
r = np.sqrt(x_dist ** 2 + y_dist ** 2)
r_values = np.linspace(0, 4, 1000)
plt.figure()
plt.hist(r, density = True, label = 'Simulated data')
plt.plot(r_values, P_nuclear(r_values, 1), label = r'P$_{nuc}$')
plt.xlabel('r')
plt.ylabel('Probability density')
plt.legend()
plt.show()

# - Use the PDF $P_{\rm nuc}$ to obtain a measurement of $\sigma_{xy}$ for the sample of AGN flares. Plot the result and comment on the result:


#     - Does your inference of $\sigma_{xy}$ match what you expected based on the typical sample variance in the position measurements?
#     - What is the uncertainty on your measurement of $\sigma_{xy}$?
#     - Compute $r_{90}$, the value of $r$ below which we find all nuclear transients: $\int_0^{r_{90}} P_{\rm nuc} dr \equiv 0.9$.