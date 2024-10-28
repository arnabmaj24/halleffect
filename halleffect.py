import matplotlib.pyplot as plt
import numpy as np
from halluncert import EUncert, Juncert, sigma_b, weighted_average, dmu
from matplotlib.ticker import ScalarFormatter
# Data
E =[-0.00005078720163,-0.0000406297613,-0.00003047232098,-0.00004570848146,-0.0000812595226,-0.00009141696293,-0.0001168105637,-0.0000812595226,-0.00005586592179]
VH =[-1.1,-0.9,-0.7,-1.1,-1.6,-2.1,-2.5,-1.7,-1.2]
Current = [3.45,2.45,1.46,1.46,2.44,3.53,3.68,2.64,1.47]
J = [4.18E+07,2.97E+07,1.77E+07,1.77E+07,2.96E+07,4.28E+07,4.46E+07,3.20E+07,1.78E+07]


E = np.multiply(np.abs(E), 1) # Taking absolute value for analysis
J = np.abs(J)
EU = EUncert(VH)  # Y-axis uncertainties (Electric Field)
JU = Juncert(Current)  # X-axis uncertainties (Current Density)


# Print data with uncertainties
# for i in range(len(E)):
#     print(E[i], c, EU[i], "V/m", "and", J[i], u"\u00B1", JU[i], "A/m^2")

# Linear fits
m_492, b_492 = np.polyfit(J[0:3], E[0:3], 1)
m_623, b_623 = np.polyfit(J[3:6], E[3:6], 1)
m_725, b_725 = np.polyfit(J[6:9], E[6:9], 1)

# Create a figure with two subplots: one for the main plot and one for residuals (side by side)
fig, axs = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [4, 1]})

# Main plot: Electric Field vs. Current Density with error bars
ax1 = axs[0]
ax1.errorbar(J[0:3], E[0:3], yerr=EU[0:3], xerr=JU[0:3], fmt='o', color='r', label='492 mT', capsize=5)
ax1.errorbar(J[3:6], E[3:6], yerr=EU[3:6], xerr=JU[3:6], fmt='o', color='g', label='623 mT', capsize=5)
ax1.errorbar(J[6:9], E[6:9], yerr=EU[6:9], xerr=JU[6:9], fmt='o', color='b', label='725 mT', capsize=5)

# Plot the fitted lines
J_fit_492 = np.linspace(min(J[0:3]), max(J[0:3]), 100)
J_fit_623 = np.linspace(min(J[3:6]), max(J[3:6]), 100)
J_fit_725 = np.linspace(min(J[6:9]), max(J[6:9]), 100)

ax1.plot(J_fit_492, m_492 * J_fit_492 + b_492, color='red', linestyle='--', label='Fit 492 mT')
ax1.plot(J_fit_623, m_623 * J_fit_623 + b_623, color='green', linestyle='--', label='Fit 623 mT')
ax1.plot(J_fit_725, m_725 * J_fit_725 + b_725, color='blue', linestyle='--', label='Fit 725 mT')

# Adding labels, title, and legend
ax1.set_xlabel('Current Density (A/m^2)')
ax1.set_ylabel('Electric Field (V/m)')
ax1.set_title('Electric Field vs Current Density', pad=15)  # Add padding for clarity
ax1.grid(True)
ax1.legend()
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Residuals plot (stacked vertically below the main plot)
ax2 = axs[1]

# Calculate residuals for each group
residuals_492 = E[0:3] - (m_492 * J[0:3] + b_492)
residuals_623 = E[3:6] - (m_623 * J[3:6] + b_623)
residuals_725 = E[6:9] - (m_725 * J[6:9] + b_725)

# Plot residuals for each group with error bars (flipped to be vertically tall)
ax2.errorbar(J[0:3], residuals_492, yerr=EU[0:3], fmt='o', color='r', label='Residuals 492 mT', capsize=5)
ax2.errorbar(J[3:6], residuals_623, yerr=EU[3:6], fmt='o', color='g', label='Residuals 623 mT', capsize=5)
ax2.errorbar(J[6:9], residuals_725, yerr=EU[6:9], fmt='o', color='b', label='Residuals 725 mT', capsize=5)

# Adding labels and grid for residuals plot
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Current Density (A/m^2)')  # X-axis shows current density
ax2.set_ylabel('Residuals (V/m)')
ax2.set_title('Residuals of the Fit')  # Add padding to the title for more space
ax2.grid(True)


# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Print the slopes (m values) and resistances
print(f"Slope for 492 mT: {m_492}")
print(f"Slope for 623 mT: {m_623}")
print(f"Slope for 725 mT: {m_725}")

print(f"R for 492 mT: {m_492 / (492 * 0.001)}")
print(f"R for 623 mT: {m_623 / (623 * 0.001)}")
print(f"R for 725 mT: {m_725 / (725 * 0.001)}")

# Fit function (predicted values)
def fit_function(J, m, b):
    return np.multiply(m,J) + b

# Chi-squared for the 492 mT group
predicted_492 = fit_function(J[0:3], m_492, b_492)
chi2_492 = np.sum(((E[0:3] - predicted_492) / EU[0:3])**2)
# chi2_4922 = np.sum(((E[0:3] - predicted_492))**2 / E[0:3])

# Chi-squared for 623 mT group
predicted_623 = fit_function(J[3:6], m_623, b_623)
chi2_623 = np.sum(((E[3:6] - predicted_623) / EU[3:6])**2)
# chi2_6232 = np.sum(((E[3:6] - predicted_623))**2 / E[3:6])

# Chi-squared for 725 mT group
predicted_725 = fit_function(J[6:9], m_725, b_725)
chi2_725 = np.sum(((E[6:9] - predicted_725) / EU[6:9])**2)

# Print the chi-squared values
print(f"Chi-squared for 492 mT: {chi2_492}")
print(f"Chi-squared for 623 mT: {chi2_623}")
print(f"Chi-squared for 725 mT: {chi2_725}")


# Example calculation for Hall coefficient (Rh) for each group
B_492 = 492 * 0.001  # Convert mT to T
B_623 = 623 * 0.001  # Convert mT to T
B_725 = 725 * 0.001  # Convert mT to T

# Calculate Rh for each group
Rh_492 = m_492/B_492
Rh_623 = m_623/B_623
Rh_725 = m_725/B_725

um492 = sigma_b(J[0:3], EU[0:3])
um623 = sigma_b(J[3:6], EU[3:6])
um725 = sigma_b(J[6:9], EU[6:9])



def unR(r, um, m, b):
    return r * ((0.001/b)**2 + (um/m)**2)**0.5

uncR492 = unR(Rh_492, um492, m_492, B_492)
uncR636 = unR(Rh_623, um623, m_623, B_623)
uncR725 = unR(Rh_725, um725, m_725, B_725)
u = [uncR492, uncR636, uncR725]
r = [Rh_492, Rh_623, Rh_725]


print(um492, um623, um725)
RH, sigmaRH = weighted_average(r,u)
print(RH, sigmaRH)

# # Print Rh values for each group
# print("Hall Coefficient for 492 mT:", Rh_492)
# print("Hall Coefficient for 623 mT:", Rh_623)
# print("Hall Coefficient for 725 mT:", Rh_725)
# print(rms_average,u"\u00B1", standard_deviation)