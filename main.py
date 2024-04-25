import astropy.io.ascii
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

data = astropy.io.ascii.read('./ZTF_flares.dat') # data from of van Velzen et al. 2019
# print(data)
# print(len(data))
# print('# unknown sources', sum(data['classification']=='None'))
def histogram(data, label, title, xlabel, ylabel):
    """
    Creates a histogram of the data that is inputted into the definition.
    :param data: The inputted data of which a histogram should be created (list/array)
    :param label: Label of the plotted data (string)
    :param title: Title of the plot (string)
    :param xlabel: Label of the x-axis (string)
    :param ylabel: Label of the y-axis (string)
    """
    plt.figure()
    plt.hist(data, bins='auto', label = label, edgecolor = 'black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def P_nuclear(r, sigma_xy):
    """
    The function for P$_{nuclear}$ which gives the distribution of the nuclear transients
    :param r: The distance from the origin (list/array)
    :param sigma_xy: The typical uncertainty on the position of the x and or y coordinate (float)
    :return: The function that gives the probability distribution of nuclear transients
    """
    P_nuc = (np.sqrt(2 * np.pi) * r * sp.stats.norm.pdf(r, loc = 0, scale = sigma_xy)) / sigma_xy
    return P_nuc
def log_likelihood(r, sigma_xy, f_nuc = None):
    """
    Calculates the log likelihood of the specified mode
    :param r: The distance from the origin (list/array)
    :param sigma_xy: The typical uncertainty on the position of the x and or y coordinate (float)
    :param f_nuc: The fraction of nuclear transients in this population (function, optional)
    :return: The log likelihood of the specified mode (float)
    """
    if f_nuc is None:
        return np.sum(np.log(P_nuclear(r, sigma_xy)))
    else:
        return np.sum(np.log(unknown_distribution(f_nuc, r, sigma_xy)))


# def integrate_pnuc(r, r90, sigma_xy):
#     """
#     :param r90:
#     :param sigma_xy:
#     :return: Returns the integrated function of
#     """
#     integrand_func = lambda r: P_nuclear(r, sigma_xy)
#     P_nuc, _ = sp.integrate.quad(integrand_func, a = 0, b = r90)
#     return P_nuc

AGN_sources = []
AGN_sources_indices = []
SN_sources = []
SN_sources_indices = []
Unknown_sources = []
Unknown_sources_indices = []

i = 0
for source in data['classification']: #This for loop finds the different sources and divides them into 3 different lists
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

# - Plot the offset distribution of the SN, AGN and unknown sources. Discuss the following two points:
#   - Describe the difference between the SN and nuclear flare offset distribution.
AGN = data[AGN_sources_indices]
SNe = data[SN_sources_indices]
Unknown = data[Unknown_sources_indices]
hist_stack_data = [AGN['offset_mean'], SNe['offset_mean'], Unknown['offset_mean']]
stack_label = ['AGN', 'SNe', 'Unknown']
plt.figure()
plt.title('Mean offset distribution for different objects')
plt.hist(hist_stack_data, bins='auto', label = stack_label, stacked = True, edgecolor = 'black')
plt.xlabel('Mean offset [arcsec]')
plt.ylabel('Number of counts')
plt.legend()
plt.show()
# histogram(data['offset_mean'][SN_sources_indices], 'Supernovae', 'Distribution of offset of supernovae',  'Offset [arcsec]', 'Number of SNe')
# histogram(data['offset_mean'][Unknown_sources_indices], 'Unknown objects', 'Distribution of offset of different unknown objects',  'Offset [arcsec]', 'Number of objects')
# histogram(data['offset_mean'][AGN_sources_indices], 'Active Galactic Nuclei', 'Distribution of offset of AGN',  'Offset [arcsec]', 'Number of AGN')

#   - Is the offset distribution of the unknown sources consistent with originating solely from the SN offset distribution? Or solely from the AGN offset distribution? (hint: there is a statistical test to quantify this)
SNe_unknown_pvalue = sp.stats.anderson_ksamp([SNe['offset_mean'], Unknown['offset_mean']])
print(f'We can use a hypothesis that reads: The offset distribution of the unknown sources is consistent with originating solely from the SN offset distribution. Using the Anderson Darling test, we find that this results in a p value of {SNe_unknown_pvalue[2]:.3f}, which is lower than 0.05, so we can reject this hypothesis.')
AGN_unknown_pvalue = sp.stats.anderson_ksamp([AGN['offset_mean'], Unknown['offset_mean']])
print(f'We can use a hypothesis that reads: The offset distribution of the unknown sources is consistent with originating solely from the AGN offset distribution. Using the Anderson Darling test, we find that this results in a p value of {AGN_unknown_pvalue[2]:.3f}, which is lower than 0.05, so we can reject this hypothesis.')

# Simulating data to check the function for Pnuc
x_dist = np.random.normal(0, 1, 1000)
y_dist = np.random.normal(0, 1, 1000)
r = np.sqrt(x_dist ** 2 + y_dist ** 2)
r_values_sim = np.linspace(0, 4, 1000)
plt.figure()
plt.hist(r, density = True, label = 'Simulated data')
plt.plot(r_values_sim, P_nuclear(r_values_sim, 1), label = r'P$_{nuc}$')
plt.xlabel('r')
plt.ylabel('Probability density')
plt.legend()
plt.show()

# - Use the PDF $P_{\rm nuc}$ to obtain a measurement of $\sigma_{xy}$ for the sample of AGN flares. Plot the result and comment on the result:
sigma_logli = []
sigmas = np.linspace(1e-30, 1, 1000)
for i in range(len(sigmas)):
    sigma_logli.append(log_likelihood(AGN['offset_mean'], sigmas[i]))
print(sigma_logli)
sigma_xy_logli_max, sigma_xy_best = np.max(sigma_logli), sigmas[np.argmax(sigma_logli)]
print(sigma_xy_best)
print(f'The maximum log likelihood of this distribution is {sigma_xy_logli_max:.3f}, the corresponding value for f_nuc is {sigma_xy_best:.3f}.')
plt.figure()
plt.scatter(sigma_xy_best, sigma_xy_logli_max, c = 'red', label = rf'$\sigma_{{xy}}$ = {sigma_xy_best:.3f}')
plt.plot(sigmas, sigma_logli, label = 'Log likelihood of AGN')
plt.title('Log likelihood of unknown objects')
plt.xlabel(r'Values for $\sigma_{xy}$')
plt.ylabel('Log likelihood')
plt.legend()
plt.show()

AGN_hist, bins = np.histogram(AGN['offset_mean'], density = True)
bin_center = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
sigma_xy_guess, pcov = sp.optimize.curve_fit(P_nuclear, bin_center, AGN_hist)
print(sigma_xy_guess)
# histogram(data['offset_mean'][AGN_sources_indices], 'Active Galactic Nuclei', 'Distribution of offset of AGN',  'Offset [arcsec]', 'Number of AGN')
r_values = np.linspace(0, 0.8, 1000)
plt.figure()
plt.hist(AGN['offset_mean'], label = 'Active Galactic Nuclei', density = True, bins = 'auto')
plt.plot(r_values, P_nuclear(r_values, sigma_xy_best))
plt.xlabel('Offset [arcsec]')
plt.xlim(0, 0.8)
plt.ylabel('Probability density')
plt.title('Optimized sigma value for AGN flares')
plt.legend()
plt.show()
#     - Does your inference of $\sigma_{xy}$ match what you expected based on the typical sample variance in the position measurements?
expected_var = np.var(AGN['offset_mean'])
print(expected_var, 'DEZE IS NOG NIET GOED') #DEZE HEB IK NOG NIET HELUP
#     - What is the uncertainty on your measurement of $\sigma_{xy}$?
std_sigmaxy = np.sqrt(np.diag(pcov))
print(std_sigmaxy)
#     - Compute $r_{90}$, the value of $r$ below which we find all nuclear transients: $\int_0^{r_{90}} P_{\rm nuc} dr \equiv 0.9$.
P_nuc_int = []
for i in range(len(r_values)):
    P_nuc_int.append(sp.integrate.quad(P_nuclear, 0, r_values[i], args = sigma_xy_best)[0])
interp_r90 = np.interp(0.9, P_nuc_int, r_values)
print(f'The value for r_90 for which the integral over all nuclear transients is equal to 0.9 is r_90 = {interp_r90}')

# - To estimate the PDF for the offset distribution of SN, we a Gaussian Kernel Density Estimation (KDE; as discussed in the lecture of week 5).
SN_KDE = sp.stats.gaussian_kde(SNe['offset_mean'], bw_method = 0.4)
plt.figure()
plt.hist(SNe['offset_mean'], label = 'Supernovae', density = True, bins = 'auto')
plt.plot(r_values, SN_KDE(r_values), label = 'Gaussian KDE SNe')
plt.xlabel('Offset [arcsec]')
plt.ylabel('Probability density')
plt.title('Estimation of PDF of offset distribution of Supernovae')
plt.legend()
plt.show()
# - In the previous two steps, you obtained two PDFs: one for SN and one for nuclear transients. The offset distribution of transients with an unknown classification must originate from one of these two PDFs. You can therefore write own a final PDF for the unknown distribution:
def unknown_distribution(f_nuc, r, sigma_xy):
    """
    The function that gives the PDF of the unknown distribution of unknown objects
    :param f_nuc: The fraction of nuclear transients in this population (function)
    :param r: The distance from the origin (list/array)
    :param sigma_xy: The typical uncertainty on the position of the x and or y coordinate (float)
    :return: The PDF of the unknown distribution of unknown objects
    """
    Nuclear_transient_part = f_nuc * P_nuclear(r, sigma_xy)
    SN_part = (1 - f_nuc) * SN_KDE(r)
    Unknown_dist = Nuclear_transient_part + SN_part
    return Unknown_dist

# Finding the value for f_nuc through the use of the log likelihood
f_nuc_logli = []
f_nuc_values = np.linspace(0, 1, 1000)
for i in range(len(f_nuc_values)):
    f_nuc_logli.append(log_likelihood(Unknown['offset_mean'], sigma_xy_best, f_nuc_values[i]))
f_nuc_logli_max, f_nuc = np.max(f_nuc_logli), f_nuc_values[np.argmax(f_nuc_logli)]
print(f'The maximum log likelihood of this distribution is {f_nuc_logli_max:.3f}, the corresponding value for f_nuc is {f_nuc:.3f}.')

plt.figure()
plt.scatter(f_nuc, f_nuc_logli_max, c = 'red', label = rf'f$_{{nuc}}$ = {f_nuc:.3f}')
plt.plot(f_nuc_values, f_nuc_logli, label = 'Log likelihood of unknown objects')
plt.title('Log likelihood of unknown objects')
plt.xlabel(r'Values for f$_{\mathrm{nuc}}$')
plt.ylabel('Log likelihood')
plt.legend()
plt.show()

plt.figure()
plt.plot(r_values, unknown_distribution(f_nuc, r_values, sigma_xy_best), label = 'Distribution of unknown objects')
plt.plot(r_values, SN_KDE(r_values), label = 'Estimated distribution of supernovae')
plt.plot(r_values, P_nuclear(r_values, sigma_xy_best), label = 'Distribution of AGN')
plt.xlabel('Mean offset [arcsec]')
plt.ylabel('Probability density')
plt.title('WIP')
plt.legend()
plt.show()

plt.figure()
plt.plot(r_values, unknown_distribution(f_nuc, r_values, sigma_xy_best), label = 'Distribution of unknown objects')
plt.hist(Unknown['offset_mean'], bins = 'auto', density = True)
plt.xlabel('Mean offset [arcsec]')
plt.ylabel('Probability density')
plt.title('WIP')
plt.show()