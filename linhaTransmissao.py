import numpy as np
from scipy.special import kv as BesselK, iv as BesselI  
from scipy.constants import pi
import matplotlib.pyplot as plt

# Constantes e funções
Pi = np.pi
I = 1j
mu0 = 4e-7*Pi
eps0 = 8.854e-12

### impedância interna de condutores tubulares
def ZintTubo(Omega, Rhoc, rf, rint, Mur=1, Mu=mu0):
    Etac = np.sqrt( (I*Omega*Mur*Mu)/Rhoc )
    ri = rint + 1e-6
    cf = Etac*rf
    ci = Etac*ri
    Den = BesselK(1, ci)*BesselI(1, cf) - BesselK(1, cf)*BesselI(1, ci)
    Num = BesselK(1, ci)*BesselI(0, cf) + BesselK(0, cf)*BesselI(1, ci)
    return Rhoc*Etac*Num/(2*Pi*rf*Den)

# impedância interna de condutor sem alma de aço
# impedância interna de condutores cilíndricos
# Mur=90 para cabos de aço
def Zin(Omega, Rhopr, rpr, Mur=1, Mu=mu0):
    Etapr = np.sqrt( I*Omega*Mu*Mur/Rhopr )
    cr = Etapr*rpr
    civ0 = BesselI(0, cr)
    civ1 = BesselI(1, cr)
    return Etapr*Rhopr*civ0/(2*Pi*rpr*civ1)

def ZSolo(omega, r, h1, h2, sigma_solo, mu=mu0):
    eta_solo = np.sqrt(I*omega*mu*sigma_solo)
    c1 = I*omega*mu/2/Pi
    c2 = (h1 + h2)**2 - r**2
    c3 = r**2 + (h1 + h2)**2
    c4 = eta_solo**2*(r**2 + (h1 + h2)**2)
    c5 = 1 + (h1 + h2)*eta_solo
    c6 = np.exp(-eta_solo*(h1 + h2))
    c7 = BesselK(0, eta_solo*np.sqrt(r**2 + (h1 - h2)**2) )
    c8 = BesselK(2, eta_solo*np.sqrt(r**2 + (h1 + h2)**2) )
    return c1*(c7 + c2/c3*(c8 - 2*(c6*c5)/c4))

# deveria estar funcionando mas deu problemas
def eliprc(m, nc, npr):
   i = nc-npr
   return np.linalg.inv(m)[:i, :i]
# versao alternativa

# def eliprc(m, nc, npr):
#     # Compute the inverse of the matrix
#     m_inv = np.linalg.inv(m)
    
#     # Take the top-left (nc - npr) x (nc - npr) submatrix
#     reduced_m = m_inv[:nc - npr, :nc - npr]
    
#     return reduced_m 

### internal impedance of cylindrical conductors
def zintc(omega, Rhoc, rf, ri, Mur):
    return ZintTubo(omega, Rhoc, rf, ri, Mur)

def zic(omega, rhoc, rf, mur):
    return Zin(omega, rhoc, rf, mur)

# calcula matrizes Z e Y por unidae de comprimento
# case  2: ground wires and unbundled conductors
def cZYlt2(omega, x, y, sigmas, rdc, rf, rint, npr, rdcpr, rpr):
    mu = mu0
    eps = eps0
    nc = x.size
    nf = int(nc - npr)
    rhoc = rdc*Pi*(rf**2 - rint**2)
    rhopr = rdcpr*Pi*rpr**2
    if rint != 0:
        v1 = zintc(omega, rhoc, rf, rint, 1)*np.ones(nc - npr)
        v2 = zic(omega, rhopr, rpr, 1)*np.ones(npr)
        v = np.append(v1, v2)
        zin = np.diag(v)
    else:
        v1 = zic(omega, rhoc, rf, 1)*np.ones(nc - npr)
        v2 = zic(omega, rhopr, rpr, 1)*np.ones(npr)
        v = np.append(v1, v2)
        zin = np.diag(v)
        
    p = np.sqrt( 1/(I*omega*mu*sigmas) )
    table = np.array([])
    for i in range(nc):
        for j in range(nc):
            if i != j:
                novovalor = (np.log(
                        ((x[i] - x[j])**2 + (2*p + y[i] + y[j])**2)/(
                                (x[i] - x[j])**2 +(y[i] - y[j])**2)))/(2)
            elif i <= nc - npr - 1:
                novovalor = np.log( 2.0*(y[i] + p)/rf )
            else:
                novovalor = np.log( 2.0*(y[i] + p)/rpr )
                
            table = np.append(table, novovalor)
        
    ze = I*omega*mu/2/Pi*table
    ze = ze.reshape(zin.shape)
    Z1 = np.linalg.inv(eliprc(zin+ze, nc, npr))
    mp = np.array([])
    for i in range(nc):
        for j in range(nc):
            if i != j:
                novovalor = (1/2)*np.log(
                        ((x[i] - x[j])**2 + (y[i] + y[j])**2)/(
                                (x[i] - x[j])**2 + (y[i] - y[j])**2))
            elif i <= nc - npr - 1:
                novovalor = np.log( (2*y[i])/rf )
            else:
                novovalor = np.log( (2*y[i])/rpr )
                
            mp = np.append(mp, novovalor)
        
    mp = mp.reshape(zin.shape)
    co2 = eliprc(mp, nc, npr)
    Y1 = 3e-12*np.diag( np.ones(nf) ) + I*omega*2*Pi*eps*co2
    return Z1, Y1


# monta matriz Ybarra da LT
def ynLT(Z, Y, length):
    evals, evect = np.linalg.eig(np.dot(Z, Y))
    
    d = np.sqrt(evals)
    
    # Step 3: Transformation matrices
    Tv = evect.T
    Tvi = np.linalg.inv(Tv)
    
    # Step 4: Exponential operation
    hm = np.exp(-d * length)
    
    # Step 5: Calculate Am and Bm
    Am = d * (1 + hm**2) / (1 - hm**2)
    Bm = -2.0 * d * hm / (1 - hm**2)
    
    # Step 6: Compute y11 and y12
    Z_inv = np.linalg.inv(Z)
    y11 = np.dot(np.dot(np.dot(Z_inv, Tv), np.diag(Am)), Tvi)
    y12 = np.dot(np.dot(np.dot(Z_inv, Tv), np.diag(Bm)), Tvi)
    
    return y11, y12