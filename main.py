import astropy.io.ascii
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import warnings
import tqdm
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

def P_nuclear(r, sigma_xy):
    """
    The function for P$_{nuclear}$ which gives the distribution of the nuclear transients
    :param r: The distance from the origin (list/array)
    :param sigma_xy: The typical uncertainty on the position of the x and or y coordinate (float)
    :return: The function that gives the probability distribution of nuclear transients
    """
    P_nuc = (np.sqrt(2 * np.pi) * r * sp.stats.norm.pdf(r, loc = 0, scale = sigma_xy)) / sigma_xy
    return P_nuc

def integrate_pnuc(r90, sigma_xy):
    """
    :param r90:
    :param sigma_xy:
    :return: Returns the integrated function of
    """
    integrand_func = lambda r: P_nuclear(r, sigma_xy)
    P_nuc, _ = sp.integrate.quad(integrand_func, a = 0, b = r90)
    return P_nuc

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
plt.figure()
AGN_hist, bins = np.histogram(AGN['offset_mean'], density = True)
bin_center = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
sigma_xy_guess, pcov = sp.optimize.curve_fit(P_nuclear, bin_center, AGN_hist)
print(sigma_xy_guess)
# histogram(data['offset_mean'][AGN_sources_indices], 'Active Galactic Nuclei', 'Distribution of offset of AGN',  'Offset [arcsec]', 'Number of AGN')
plt.figure()
plt.hist(AGN['offset_mean'], label = 'Active Galactic Nuclei', density = True, bins = 'auto')
plt.plot(r_values, P_nuclear(r_values, sigma_xy_guess))
plt.xlabel('Offset [arcsec]')
plt.xlim(0, 0.8)
plt.ylabel('WORK IN PROGRESS')
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
# value_guesses = [0.5, sigma_xy_guess]
# r90, errors = sp.optimize.curve_fit(integrate_pnuc, xdata = bin_center, ydata = AGN_hist)
# print(r90)

def find_boundary(func, lower, upper, target, tol):
    """
    Find the boundary value for a definite integral with an already known result
    :param func: The function to integrate (function)
    :param lower: The lower limit of integration (float)
    :param upper: The initial guess for the upper limit of integration (float)
    :param target: The result of the integral (float)
    :param tol: The tolerance for the difference between the integral and the target (float)
    :return: The found upper limit of integration (float)
    """
    def integral(x):
        return sp.integrate.quad(func, lower, x, args = sigma_xy_guess)[0] - target
    r90 = sp.optimize.brentq(integral, lower, upper, xtol=tol)
    return r90
r90 = find_boundary(P_nuclear, 0, 0.5, 0.9, 1e-30)
print(sp.integrate.quad(P_nuclear, 0, r90, args = sigma_xy_guess)[0])

SN_KDE = sp.stats.gaussian_kde(SNe['offset_mean'], bw_method = 0.4)
print(SN_KDE)
plt.figure()
plt.hist(SNe['offset_mean'], label = 'Supernovae', density = True, bins = 'auto')
plt.plot(r_values, SN_KDE(r_values), label = 'Gaussian KDE SNe')
plt.xlabel('Offset [arcsec]')
plt.ylabel('Probability density')
plt.xlim(0, 1)
plt.title('Estimation of PDF of offset distribution of Supernovae')
plt.legend()
plt.show()