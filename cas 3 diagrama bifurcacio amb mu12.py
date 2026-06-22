# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:30:38 2026

@author: noaga
"""

import numpy as np
import matplotlib.pyplot as plt

f1 = 0.4
f2 = 0.5
mu21 = 0.2
delta_f = f1 - f2

mu12_estudi = 0.05
mu12_valors = np.linspace(0, 0.055, 500)

def punts_equilibri(m12):
    a = - delta_f
    b = (delta_f - m12 * f1 - mu21 * f2)
    c = mu21 * f2
    disc = b**2 - 4*a*c
    z1 = (-b + np.sqrt(disc)) / (2*a)
    z2 = (-b - np.sqrt(disc)) / (2*a)
    return z1, z2, disc

z1_valors, z2_valors = [], []
for m in mu12_valors:
    z1, z2, _ = punts_equilibri(m)
    z1_valors.append(z1)
    z2_valors.append(z2)

z1_estudi, z2_estudi, _ = punts_equilibri(mu12_estudi)

plt.figure(figsize=(8, 5))
plt.plot(mu12_valors, z1_valors, color='red', linestyle='--', linewidth=2)
plt.plot(mu12_valors, z2_valors, color='blue', linewidth=2)
plt.plot(mu12_estudi, z1_estudi, 'ko', markersize=7, zorder=5, label='$\mu_{{12}}=0.05$')
plt.plot(mu12_estudi, z2_estudi, 'ko', markersize=7, zorder=5)

z_zero, _, _ = punts_equilibri(0)
plt.plot(0, z_zero, 'o', color='green', markersize=7, zorder=5)

plt.plot([], [], 'b-', lw=2, label='Stable')
plt.plot([], [], 'r--', lw=2, label='Unstable')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$\mu_{12}$', fontsize=15)
plt.ylabel(r'$z^*$', fontsize=15)
plt.axhline(0, color='black', linestyle='--', lw=0.5)
plt.axvline(0, color='black', linestyle='--', lw=0.5)
plt.xlim(-0.0012, 0.057)
plt.legend(
    fontsize=15, 
    loc='lower left', 
    bbox_to_anchor=(0.02, 0.05),
    labelspacing=0.2,   
    handletextpad=0.5, 
    borderaxespad=0.5   
)
plt.grid(False) 
plt.tight_layout()
plt.show()