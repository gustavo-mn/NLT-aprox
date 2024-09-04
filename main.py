import numpy as np
import matplotlib.pyplot as plt
import linhaTransmissao as lt
import time

def calculateFrequencyBoundaries(Z, Y, omega, l):
    
    evals, _ = np.linalg.eig(np.dot(Z, Y))
    d = np.sqrt(evals)

    v = omega / np.imag(np.sqrt(d))
    v_min = np.min(v)

    f_0 = v_min/(4*l) # quarter-wave resonance frequency
    omega_0 = 2 * np.pi * f_0

    omega_next = omega + 2*omega_0

    return omega_next

def calculateTLResponse(omega, A, B, gf):
    
    # condicoes nos terminais da LT
    term1 = np.diag([gf, gf, gf])
    term2 = np.diag([0.0, 0.0, 0.0])
    
    # Matriz do sistema 
    Ynodal = np.block([[A + term1, B], [B, A + term2]])
    
    # vetor de excitacao
    exci = np.concatenate(([100.0 / (1j * omega)], np.zeros(5))) # Entrada do tipo degrau
    # exci = np.concatenate(([100.0], np.zeros(5))) # Entrada do tipo impulso
    
    # resolve o sistema 
    vout_freq = np.linalg.solve(Ynodal, exci)

    return vout_freq

# Constantes e funções
mu0 = 4e-7*np.pi
eps0 = 8.854e-12

fb = 3000
# omegaMin = 100 * 10**(-3)
# omegaMax =  50 * 10**3

f_min = 1*10**(-3)
f_max = 200*10**(3)

# logspace lower frequency
flog = np.logspace(np.log10(f_min), np.log10(fb/2), 32)
# linspace higher frequency
# flin = np.linspace(fb/2, f_max, 256)

# Combine and sort the unique values from both arrays
# freq = np.sort(np.union1d(flin, flog))


# Tentando reproduzir os resultados do Bjorn antes de aplicar uma amostragem não-uniforme
# fmin = 1 # Valor do paper do Bjorn
# fmax = 2*10**(4) # Valor do paper do Bjorn
# freq = np.linspace(fmin, fmax, 10000)

# f_boundary = 3000 # Valor do paper do Bjorn
# omega_boundary = 2*np.pi*f_boundary

# para o uso da NLT
Tmax = 25e-3
# c = - np.log(0.001)/Tmax
c = 0 # Tentando reproduzir os resultados do Bjorn antes de considerar a frequência complexa "s"
# sk = - 1j * c + 2 * np.pi * freq
sk = - 1j * c + 2 * np.pi * flog # Apenas as frequências do intervalo inferior - espaçamento logaritmico entre amostras
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
vout_freq_band_low = np.zeros((nf, 6), dtype=complex)  # Assuming 6 elements based on the Join[{100.0/(I \[Omega])}, Table[0, {5}]]

# Timing equivalent in Python
start_time = time.time()

# Loop over nm from 1 to nf
for nm in range(nf): # Loop do intervalo inferior de frequência
    omega = sk[nm]
    
    Z, Y = lt.cZYlt2(omega, xc, yc, 1/Rho, Rdc, r1, 0, npr, Rdcpr, rpr) 
    A, B = lt.ynLT(Z, Y, compr)

    # gf = 0.0  # Placeholder value for gf, should be defined
    vout_freq_band_low[nm] = calculateTLResponse(omega, A, B, gf)
    
    # # condicoes nos terminais da LT
    # term1 = np.diag([gf, gf, gf])
    # term2 = np.diag([0.0, 0.0, 0.0])
    
    # # Matriz do sistema 
    # Ynodal = np.block([[A + term1, B], [B, A + term2]])
    
    # # vetor de excitacao
    # # exci = np.concatenate(([100.0 / (1j * omega)], np.zeros(5))) # Entrada do tipo degrau
    # exci = np.concatenate(([100.0], np.zeros(5))) # Entrada do tipo impulso
    
    # # resolve o sistema 
    # vout_freq_band_low[nm] = np.linalg.solve(Ynodal, exci)

# Timing end
end_time = time.time()
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")

freq_band_high = []
omega_max = 2*np.pi*f_max

idx = 0

while omega <= omega_max: # Loop do intervalo superior de frequência

    freq_band_high.append(omega)

    # Determinar omega_{k+1} segundo a seção III.B do paper do Bjorn
    omega_next = calculateFrequencyBoundaries(Z,Y,omega)
    
    # TODO: Entre omega e omega_{k+1}, subdividir o intervalo em 100 pontos - seguindo teste do Bjorn - e usar a integração adaptativa de Simpson

    # Para cada ponto de frequência calculado entre omega e omega_{k+1}, obter os valores de saída de tensão na linha de transmissão
    omega = omega_next
    Z, Y = lt.cZYlt2(omega, xc, yc, 1/Rho, Rdc, r1, 0, npr, Rdcpr, rpr) 
    A, B = lt.ynLT(Z, Y, compr)

    vout_freq = calculateTLResponse(omega, A, B, gf)

    if idx == 0:
        vout_freq_band_high = vout_freq
    else:
        vout_freq_band_high = np.vstack((vout_freq_band_high, vout_freq))

# v4out_freq = vout_freq[:,3]
v4out_freq = vout_freq_band_low[:,3]

plt.figure()
# plt.loglog(freq, np.abs(v4out_freq))
plt.loglog(flog, np.abs(v4out_freq))
plt.xlabel('f')
plt.ylabel('|V4|')
plt.grid('on')

full_spectrum = np.concatenate((v4out_freq, np.conjugate(v4out_freq[::-1])))
v4out_time = np.fft.ifft(full_spectrum)

plt.figure()
plt.plot(v4out_time)
plt.show()