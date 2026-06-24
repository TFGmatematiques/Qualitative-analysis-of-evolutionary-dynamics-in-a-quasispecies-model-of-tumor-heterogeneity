import numpy as np
import matplotlib.pyplot as plt

z_min = 0.0
z_max = 1.001
punts = 2000

z = np.linspace(z_min, z_max, punts)

def funcio(z, f1, f2, mu12, mu21):
    delta_f = f1 - f2
    return -delta_f * (z**2) + (delta_f - mu12*f1 - mu21*f2)*z + mu21*f2

casos = [
    {'f1': 0.3, 'f2': 0.15, 'mu12': 0.1, 'mu21': 0.01, 'color': 'darkorange',  'label': 'A'},
    {'f1': 0.3, 'f2': 0.3, 'mu12': 0.05, 'mu21': 0.05, 'color': 'crimson',  'label': 'B'},
    {'f1': 0.3, 'f2': 0.3, 'mu12': 0.08, 'mu21': 0.02, 'color': 'forestgreen',  'label': 'C'},
    {'f1': 0.4, 'f2': 0.5, 'mu12': 0.05, 'mu21': 0.2, 'color': 'royalblue',  'label': 'D'},
    {'f1': 0.4, 'f2': 0.5, 'mu12': 0, 'mu21': 0.2, 'color': 'navy',  'label': 'E'}
]

plt.figure(figsize=(8, 5))

for c in casos:
    z_punt = funcio(z, c['f1'], c['f2'], c['mu12'], c['mu21'])
    plt.plot(z, z_punt, color=c['color'], label=c['label'], linewidth=2)

    f1, f2 = c['f1'], c['f2']
    mu12, mu21 = c['mu12'], c['mu21']
    delta_f = f1 - f2
    
    a = -delta_f
    b = (delta_f - mu12*f1 - mu21*f2)
    c_val = mu21*f2
    
    if abs(a) <= 1e-6:
        z_eq = -c_val / b
        if z_min <= z_eq <= z_max:
            color_punt = 'blue' if b < 0 else 'red'
            plt.plot(z_eq, 0, 'o', color=color_punt, markersize=7, zorder=5)
            continue

    disc = b**2 - 4*a*c_val
    tol = 1e-6

    if disc < -tol:
        print("RESULTAT: El discriminant és negatiu. No hi ha punts reals (la corba no toca l'eix).")
        
    elif abs(disc) <= tol:
        z_eq = -b / (2*a)

        if z_min <= z_eq <= z_max:
            plt.plot(z_eq, 0, 'o', color='green', markersize=7, zorder=5)
            
    else:
        arrels = [(-b + np.sqrt(disc)) / (2*a), (-b - np.sqrt(disc)) / (2*a)]
        
        for r in arrels:
            if z_min <= r <= z_max:
                pendent = 2*a*r + b
                color_punt = 'blue' if pendent < 0 else 'red'
                plt.plot(r, 0, 'o', color=color_punt, markersize=7, zorder=5)

plt.axhline(0, color='black', linestyle='--', lw=0.5)
plt.axvline(0, color='black', linestyle='--', lw=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('$z$', fontsize=15)
plt.ylabel(r'$\dot{z}$', fontsize=15)
plt.legend(
    fontsize=15, 
    loc='best', 
    labelspacing=0.2, 
    handletextpad=0.5,  
    borderaxespad=0.5    
)
plt.grid(False) 
plt.xlim(z_min - 0.02, z_max + 0.02)
plt.tight_layout()
plt.show()
