import numpy as np
import matplotlib.pyplot as plt
import datetime
from numpy import vectorize

C_reyleigh = 10**5.36
k = 0.172  # the median value at the 2800-m level of Mauna Kea


#_________________________________________________________________________

###### Variable Parameters ######

moon_phase_angle = 30  # Unit: Degrees / Range: (-180, 180)
z_moon = np.deg2rad(60)
A_moon = np.deg2rad(180)

#_________________________________________________________________________


def get_distance(z1, A1, z2, A2):
    delta_A = A2 - A1
    cos_sep = np.cos(z1)*np.cos(z2) + np.sin(z1)*np.sin(z2)*np.cos(delta_A)
    return np.arccos(cos_sep)

def get_sky_brightness(z):
    X = get_airmass(z)
    B_zen = 79
    return B_zen * 10**(-0.4 * k * (X-1)) * X

def get_excess_magnitude(B_sky, B_moon):
    mag_sky = 1 / 0.92104 * ( 20.7233 - np.log(B_sky / 34.08) )
    mag_sky_with_moon = 1 / 0.92104 * ( 20.7233 - np.log((B_sky + B_moon) / 34.08) )
    return mag_sky_with_moon - mag_sky

def get_airmass(z):  # z: zenith_angle in radians
    return (1 - 0.96*np.sin(z)**2)**(-0.5)

def get_moon_brightness(z, A):  ## z, A: zenith_angle, azimuth in radians
    theta = get_distance(z, A, z_moon, A_moon)

    # Simple Scattering Function
    # f = C_reyleigh * (1.06 + np.cos(theta)**2) + 10**(6.15 - np.rad2deg(theta) / 40)
    
    # Complex Scattering Function
    f = C_reyleigh * (1.06 + np.cos(theta)**2)
    threshold = 5
    if np.rad2deg(theta) < threshold:  # Flattening for Graph
        f += 6.2 * 10**7 / threshold**2
    elif threshold < np.rad2deg(theta) < 10:
        f += 6.2 * 10**7 / np.rad2deg(theta)**2
    else:
        f += 10**(6.15 - np.rad2deg(theta) / 40)

    # Moon Brightness
    I = 10 ** ( -0.4 * (3.84 + 0.026*abs(moon_phase_angle) + 4*10**(-9) * moon_phase_angle**4) )
    
    X = get_airmass(z)
    X_m = get_airmass(z_moon)
    
    B_moon = f * I * 10**(-0.4*k*X_m) * (1 - 10**(-0.4*k*X))

    B_sky = get_sky_brightness(z)
    delta_mag = get_excess_magnitude(B_sky, B_moon)

    return B_moon
    # return abs(delta_mag)


azimuths = np.deg2rad(np.arange(0, 360.05, 1))
zenith_distances = np.deg2rad(np.arange(0, 90.005, 0.1))
vget_moon_brightness = vectorize(get_moon_brightness)


some_z = np.array([.4, 30, 60, 90, 120])
some_z = np.deg2rad(some_z)
some_B = np.zeros(5)
some_mag = np.zeros(5)
for i in range(5):
    z = z_moon - some_z[i]
    A = np.pi
    if z < 0:
        z = -z
        A = 0
    some_B[i] = vget_moon_brightness(z, A)

print(np.rad2deg(some_z))
print(some_B)


azimuths_grid, zenith_distances_grid = np.meshgrid(azimuths, zenith_distances)
B_moon = vget_moon_brightness(zenith_distances_grid, azimuths_grid)

ax = plt.subplot(projection="polar")
ax.set_ylim(0, 90)
label_locations = np.arange(0, 95, 15)
label_values = np.flip(label_locations)
label_locations = np.delete(label_locations, [0, len(label_locations)-1])
label_values = np.delete(label_values, [0, len(label_values)-1])
plt.yticks(label_locations, label_values, color='white', alpha=0.6)
ax.set_theta_direction(-1)

# Zenith Label
ax.plot(0, 0, '.', color='white', alpha=0.6)
ax.annotate('Z', xy=(0, 0), xytext=(0, 6.5),
            horizontalalignment='right', verticalalignment='center', color='white', alpha=0.6)
# North Pole Label
latitude = np.deg2rad(33.67355)
ax.plot(0, 90-np.rad2deg(latitude), '.', color='white', alpha=0.6)
ax.annotate('P', xy=(0, 90-np.rad2deg(latitude)), xytext=(0, 90-np.rad2deg(latitude)+2.5),
            horizontalalignment='left', verticalalignment='center', color='white', alpha=0.6)

colormap = plt.get_cmap('rainbow')  # rainbow_r for magnitude
im = plt.pcolormesh(azimuths_grid, np.rad2deg(zenith_distances_grid), B_moon, cmap=colormap)
plt.colorbar(im, pad=0.08)
ax.grid(color='white', alpha=0.3)

text = 'Moon Phase = ' + str( round( (1 + np.cos(np.deg2rad(moon_phase_angle))) / 2, 2) )
ax.text(np.deg2rad(130), 132.5, s=text)

plt.show()