# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:18:13 2026

@author: noaga
"""

import numpy as np
import matplotlib.pyplot as plt

f1 = 0.2
f2 = 0.1
delta_f = f1 - f2
mu_critic = delta_f / f1

mu12 = np.linspace(0, 1, 500)

eq1 = np.zeros_like(mu12)
eq2 = 1 - (mu12 * f1 / delta_f)

plt.figure(figsize=(8, 5))

estable_eq1 = mu12 > mu_critic
estable_eq2 = mu12 < mu_critic

plt.plot(mu12[~estable_eq1], eq1[~estable_eq1], 'r--', lw=2)
plt.plot(mu12[estable_eq1], eq1[estable_eq1], 'b-', lw=2)

z_rang = np.ma.masked_outside(eq2, 0, 1)

plt.plot(mu12[estable_eq2], z_rang[estable_eq2], 'b-', lw=2)
plt.plot(mu12[~estable_eq2], z_rang[~estable_eq2], 'r--', lw=2)

plt.plot(mu_critic, 0, 'ko', markersize=7, zorder=5, label='$\mu_{{12}}^c$')

plt.plot([], [], 'b-', lw=2, label='Stable')
plt.plot([], [], 'r--', lw=2, label='Unstable')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('$\mu_{12}$', fontsize=15)
plt.ylabel('$z^*$', fontsize=15)
plt.axhline(0, color='black', linestyle='--', lw=0.5)
plt.axvline(0, color='black', linestyle='--', lw=0.5)
plt.xlim(-0.02, 1.02)
plt.legend(
    fontsize=15, 
    loc='best', 
    labelspacing=0.2,    
    handletextpad=0.5,   
    borderaxespad=0.5   
)
plt.grid(False) 

plt.tight_layout()

plt.show()