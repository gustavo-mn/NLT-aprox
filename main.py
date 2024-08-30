import numpy as np
import matplotlib.pyplot as plt
import linhaTransmissao as lt
import time

# Constantes e funções
mu0 = 4e-7*np.pi
eps0 = 8.854e-12

# fb = 3000
# omegaMin = 100 * 10**(-3)
# omegaMax =  50 * 10**3

# # logspace lower frequency
# flog = np.logspace(-3, np.log10(2 * np.pi * fb), 32)
# # linspace higher frequency
# flin = np.linspace(2 * np.pi * fb, omegaMax, 256)

# # Combine and sort the unique values from both arrays
# freq = np.sort(np.union1d(flin, flog))

# Tentando reproduzir os resultados do Bjorn antes da amostragem não-uniforme
fmin = 1
fmax = 2*10**(4)
freq = np.linspace(fmin, fmax, 10000)

# para o uso da NLT
Tmax = 25e-3
# c = - np.log(0.001)/Tmax
c = 0
sk = - 1j * c + 2 * np.pi * freq
nf = len(sk)

# coordenadas dos condutores 
xc = np.array([-4.5, 0, 4.5, -2.25, 2.25])
yc = np.array([11.0, 11.0, 11.0, 14.8, 14.8])

# dados dos condutores
r1 = 21.66e-3 / 2
r0 = 0
Rdc = 0.121e-3
rpr = 12.33e-3 / 2 
Rdcpr = 0.359e-3
compr = 25.0e3
Rho = 100.0
npr = 2
rfonte = 0.01
gf = 1 / rfonte

# Initialize v1out as an array of zeros
vout_freq = np.zeros((nf, 6), dtype=complex)  # Assuming 6 elements based on the Join[{100.0/(I \[Omega])}, Table[0, {5}]]

# Timing equivalent in Python
start_time = time.time()

# Loop over nm from 1 to nf
for nm in range(nf):
    omega = sk[nm]
    
    Z, Y = lt.cZYlt2(omega, xc, yc, 1/Rho, Rdc, r1, 0, npr, Rdcpr, rpr)
    
    A, B = lt.ynLT(Z, Y, compr)
    
    # condicoes nos terminais da LT
    # gf = 0.0  # Placeholder value for gf, should be defined
    term1 = np.diag([gf, gf, gf])
    term2 = np.diag([0.0, 0.0, 0.0])
    
    # Matriz do sistema 
    Ynodal = np.block([[A + term1, B], [B, A + term2]])
    
    # vetor de excitacao
    exci = np.concatenate(([100.0 / (1j * omega)], np.zeros(5)))
    
    # resolve o sistema 
    vout_freq[nm] = np.linalg.solve(Ynodal, exci)

# Timing end
end_time = time.time()
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")


v4out_freq = vout_freq[:,3]

# # Introduzindo componente DC na resp. de frequência.
# v4out_freq = np.concatenate(([1+0j], v4out_freq))

plt.loglog(np.abs(v4out_freq))
plt.show()

aux = np.concatenate((v4out_freq, np.conjugate(v4out_freq[::-1])))

v4out_time = np.fft.ifft(aux)

plt.plot(v4out_time)
plt.show()