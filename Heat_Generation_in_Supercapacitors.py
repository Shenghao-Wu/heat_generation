#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Heat Generation in Supercapacitors with Porous Electrodes

# Electrochemical model inputs
# Electrode: Porosity; Tortuosity; Electrical conductivity; Volumetric
# capacitance
# Geometry: Electrode thickness; Seperator thickness
# Electrolyte: Concentration; Diffusion coefficient;
# Operating conditions: Current; Voltage range; Rest time perios

# Thermal model inputs
# Geometry: Number of sandwich unit N
# Thermal properties: Heat capacity; Thermal conductivity; Convection
# coefficient (h) at the surface

# Import package
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.animation as animation 
import time 
import pandas as pd


#def Heat(duration):

# Key input for testing
I0=200;                             # Absolute value of imposed current density
duration=1000;                       # total simulation time in s
rest_period=0;                     # rest break time in s

# Input electrical elements
T_inf=298.15;                       # base temperature in K
V_up=2.7;                           # Upper limite of voltage
V_low=0;                            # Lower limit of voltage
h_conv=10;                          # convection coefficient in W m-2 k-1

# Input geometry elements
L_e=50e-6;                          # left electrode length in m
L_s=25e-6;                          # separator length in m
L_al=20e-6;                         # current collector length in m
L_inf=2*L_e+L_s+2*L_al;             # total length of whole system in m

# Global constants
F=96485;                # Faraday constant in C mol-1
Ru=8.3145;              # ideal gas constant in J mol-1 K-1 |
Na=6.0221e23;           # Avogadro constant in mol-1
k_b=1.3806e-23;         # boltzmann constant in J K-1
e=1.6022e-19;           # elementary unit of an electron in Coulomb
count=0;                # simulation time display

# Liquid phase & Solid phase properties
z_p=1;                  # valency of cation | Et4+ of Et4NBF4
z_n=-1;                 # valency of anion |NBF4- of Et4NBF4
c_inf=3000;             # bulk electrolyte concentration in mol m-3 1000 | 930
beta=0.06;              # entropy constant | an intricate function of the ionic radius
dia_e=0.68e-9;          # non-solvated effective ion diameter of 0.68 nm; solvated ion diameter of 1.40 nm
eps_o=8.854e-12;        # vacuum permittivity in C V-1 m-1
eps_r=66.1;             # relative permittivity of the solvent: Et4NBF4 in propylene carbonate (PC)
sigma_ely=0.067;        # electrolyte electrical conductivity in S m-1 | should be 1.2983 / 0.067
sigma_elo=5.21;         # effective electrical conductivity of the solid phase in S m-1
sigma_al=36.9e6;        # effective electrical conductivity of Al in Siemens/m 36.9e6 | 36.9e6
aC=42e6;                # capacitance per unit volume F m-3 | 42e6 unsure ???
area_m=1e6;             # specific surface area in m2 kg-1 | capacitance per unit surface area=aC/a ?v
area_v=0.52e9;          # specific surface area in m2 m-3 ?v
eps_e=0.67;             # porosity of electrode / Void volume of carbon electrodes
eps_s=0.5;              # porosity of seperator / Void volume of separator
#d_ely=1e-11;           # diffusion coefficient of bulk electrolyte | =1.7285e-10 m2 s-1
d_ely=Ru*T_inf*sigma_ely/(c_inf*(z_p*F)**2+c_inf*(z_n*F)**2);
d_elo=d_ely*eps_e**1.5; # diffusion coefficient in electrode region
d_sep=d_ely*eps_s**1.5; # diffusion coefficient in seperator
k_ely=0.164;            # bulk thermal conductivity of solvent in W m-1 K-1 of propylene carbonate (PC) 0.164 | 0.164
k_elo=0.43;             # thermal conductivity of electrode in W m-1 K-1 0.43 | 0.649
k_sep=0.20;             # thermal conductivity of seperator in W m-1 K-1 0.20 | 0.334
k_al=237;               # thermal conductivity of Al in W m-1 K-1 237 | 205
rho_ely=1205;           # mass density of solvent in kg m-3 1205 | 1205
rho_elc=520;            # mass density of electrode in kg m-3 520 | 600
rho_sep=1780;           # mass density of seperator in kg m-3 1780 | 492
rho_al=2700;            # mass density of Al in kg m-3 2700 | 2700
cp_ely=2141;            # bulk specific heat capacity of solvent in J kg-1 K-1 of propylene carbonate (PC) 2141 | 2141
cp_elc=1062;            # specific heat capacity of electrode in J kg-1 K-1 1062 | 700
cp_sep=1400;            # specific heat capacity of seperator in J kg-1 K-1 1400 | 1978
cp_al=900;              # specific heat capacity of Al in J kg-1 K-1 900 | 900
rho_cp_ely=rho_ely*cp_ely;                                  # volume heat capacity of solvent in J m-3 K-1
rho_cp_elc=rho_elc*cp_elc;                                  # volume heat capacity of electrode in J m-3 K-1
rho_cp_sep=rho_sep*cp_sep;                                  # volume heat capacity of seperator in J m-3 K-1
rho_cp_al=rho_al*cp_al;                                     # volume heat capacity of Al in J m-3 K-1
k_elo_por=eps_e*k_ely+(1-eps_e)*k_elo;                      # thermal conductivity of porous electrode in W m-1 K-1
k_sep_por=eps_s*k_sep+(1-eps_s)*k_ely;                      # thermal conductivity of porous seperator in W m-1 K-1
k_avg=(k_al*L_al*2+k_elo_por*L_e*2+k_sep_por*L_s)/L_inf;    # average thermal conductivity in W m-1 K-1
rho_cp_elo_por=eps_e*rho_cp_ely+(1-eps_e)*rho_cp_elc;       # volume heat capacity of porous electrode in J m-3 K-1
rho_cp_sep_por=eps_s*rho_cp_sep+(1-eps_s)*rho_cp_ely;       # volume heat capacity of porous seperator in J m-3 K-1
rho_cp_avg=(rho_cp_al*L_al*2+rho_cp_elo_por*L_e*2+rho_cp_sep_por*L_s)/L_inf; # average volume heat capacityin J m-3 K-1
alpha_por_elc=k_elo_por/rho_cp_elo_por;                     # thermal diffusivity of electrode in m2 s-1
alpha_sep_por=k_sep_por/rho_cp_sep_por;                     # thermal diffusivity of seperator in m2 s-1
alpha_al=k_al/(rho_al*cp_al);                               # thermal diffusivity of Al in m2 s-1
alpha_avg=k_avg/rho_cp_avg;                                 # average thermal diffusivity  in m2 s-1
rho_elc_por=eps_e*rho_ely+(1-eps_e)*rho_elc;                # mass density of porous electrode in kg m-3;
rho_sep_por=eps_s*rho_ely+(1-eps_s)*rho_sep;                # mass density of porous seperator in kg m-3;
rho_net=rho_elc_por*2*L_e+rho_sep_por*L_s;                  # average mass density of whole systems

# Grids and accuracy
N_e=100;                            # grid number of electrodes
N_s=math.floor(N_e*L_s/L_e);        # grid number of seperator
N_al=math.floor(N_e*L_al/L_e);      # grid number of current collector
dx_nd=1/N_e;                        # non-dimentional spatial step
dx_d=dx_nd*L_e;                     # dimentional spatial step
dt_d=0.1;                           # real time step in 0.1 s
N_t=math.floor(duration/dt_d);      # grid number of time
N_rest=math.floor(rest_period/dt_d);# grid number of rest period
N1=math.floor(2*N_e);               # grid number for solid potential
N2=math.floor(2*N_e+N_s);           # grid number for liquid potential, current, ion concentration 
N3=math.floor(2*N_e+N_s+2*N_al);    # grid number for temperature and heat
dt_nd=dt_d/(L_e**2/d_elo);          # non-dementional time step

# Non-dimensional parameters
Pi1=I0*F*L_e/(sigma_ely*Ru*T_inf);                          #
Pi2=I0*L_e/(2*F*d_elo*c_inf);                               #
Pi3=aC*d_elo/sigma_ely;                                     #
Pi4=alpha_al/(d_elo);                                       #
Pi5=L_e**2*I0**2*Ru/(2*F**2*d_elo**2*c_inf*rho_cp_al);     #
Pi6=L_e*beta*I0/(rho_cp_al*T_inf*d_elo);                    #
Pi7=h_conv*L_e/k_al;                                        #

# matrix storing the seven parameters
var_nd=np.zeros(shape=(7,1));
var_nd[0][0]=Pi1;
var_nd[1][0]=Pi2;
var_nd[2][0]=Pi3;
var_nd[3][0]=Pi4;
var_nd[4][0]=Pi5;
var_nd[5][0]=Pi6;
var_nd[6][0]=Pi7;

# Non-dimensional matrices for electrical calculation
I_nd=np.zeros(shape=(N_t,1));                           # non-dementional imposed current
I1_nd=np.zeros(shape=(N_t,N1));                         # matrix for current density of solid part
phi1_nd=np.zeros(shape=(N_t,N1));                       # non-dementional potential of solid phase
phi2_nd=np.zeros(shape=(N_t,N2));                       # non-dementional potential of liquid phase
c_nd=np.zeros(shape=(N_t,N2));                          # non-dementional ion concentration
J1_nd=np.zeros(shape=(N_t,N2));                         # non-dementional current density of solid phase
J2_nd=np.zeros(shape=(N_t,N2));                         # non-dementional current density of liquid phase

# Non-dimensional matrices for heat generation calculation
Q_irr1_nd=np.zeros(shape=(N_t,N2));                     # matrix for Joule heat 1
Q_irr2_nd=np.zeros(shape=(N_t,N2));                     # matrix for Joule heat 2
Q_irr_nd =np.zeros(shape=(N_t,N2));                     # Q_irr_nd=Q_irr1_nd+Q_irr2_nd
Q_rev_nd =np.zeros(shape=(N_t,N2));                     # matrix for reversible heat
Q_tot_nd =np.zeros(shape=(N_t,N3));                     # matrix for total heat including Al
T_nd     =np.zeros(shape=(N_t,N3));                     # matrix for non-dimentional temperature

# dimensional matrices for heat generation calculation
Q_irr1=np.zeros(shape=(N_t,N3));                        # matrix for Joule heat 1
Q_irr2=np.zeros(shape=(N_t,N3));                        # matrix for Joule heat 2
Q_irr =np.zeros(shape=(N_t,N3));                        # matrix: Q_irr=Q_irr1+Q_irr2
Q_rev =np.zeros(shape=(N_t,N3));                        # matrix for reversible heat
Q_tot =np.zeros(shape=(N_t,N3));                        # matrix: Q_tot=Q_irr+Q_rev
T_d   =np.zeros(shape=(N_t,N3));                        # matrix for dimentional temperature

# Numerical matrices for electrical calculation
N4=math.floor(8*N_e+2*N_s);                             # grid number for electrical calculation
EA=np.zeros(shape=(N4,N4));                             # matrix for electrical culculation
EB=np.zeros(shape=(N4,1));                              # matrix for electrical culculation
EC=np.zeros(shape=(N4,N4));                             # matrix for electrical culculation
E_tem=np.zeros(shape=(N4,1));                           # matrix for temporary electrical data

# Numerical matrices for temperature calculation
TA=np.zeros(shape=(N3,N3));                             # matrix for thermal culculation
TB=np.zeros(shape=(N3,1));                              # matrix for thermal culculation
TC=np.zeros(shape=(N3,N3));                             # matrix for thermal culculation
T_tem=np.zeros(shape=(N3,1));                           # matrix for temperature calculation


# In[2]:


# Main sumulation starts ---------------------------------------------------------

for i in range(0,(2*N_e)):
    phi1_nd[0][i]=0;                            
    # Initiate non-dimentional solid potential
    
for i in range(0,(2*N_e+N_s)):
    c_nd[0][i]=1;                            
    # Initiate non-dimentional ion concentration
    
for i in range(0,(2*N_e+N_s+2*N_al)):
    T_nd[0][i]=0;                            
    # Initiate non-dimentional temperature

I_nd[0][0]=-1;  
rest_count=1
# Initiate applied current density for charge
# -1: charge
#  1: discharge
#  0: rest
for j in range(1,N_t):
    
    if (I_nd[j-1][0]<0) and (phi1_nd[j-1][2*N_e-1]*Ru*T_inf/F<V_up):  
        I_nd[j][0]=-1;
        # if the current is negative, and the right potential is lower than 2.5, charge continues 
    elif (I_nd[j-1][0]<0) and (phi1_nd[j-1][2*N_e-1]*Ru*T_inf/F>V_up):     
        I_nd[j][0]=+1;
        # if current is negative, and the right potential is higher 2.5, charge stops and discharge starts
    elif (I_nd[j-1][0]>0) and (phi1_nd[j-1][2*N_e-1]*Ru*T_inf/F>V_low):  
        I_nd[j][0]=+1;
        # if the current is positive, and the right potential is higher than 0, discharge continues
    elif (I_nd[j-1][0]>0) and (phi1_nd[j-1][2*N_e-1]*Ru*T_inf/F<V_low) and (N_rest!=0):   
        I_nd[j][0]=0;
        rest_count=0;
        # if the current if positive, and the right potential is lower than 0, discharge stops and rest starts (with rest)
    elif (I_nd[j-1][0]==0) and (rest_count<N_rest) and (N_rest!=0): 
        I_nd[j][0]=0;
        # if the current if positive, and the right potential is lower than 0, rest continues (with rest)
    elif (I_nd[j-1][0]==0) and (rest_count==N_rest) and (N_rest!=0):
        I_nd[j][0]=-1;        
        # if the current is positive, and the right potential is lower than 0, rest stops and charge starts (with rest)
    elif (I_nd[j-1][0]>0) and (phi1_nd[j-1][2*N_e-1]*Ru*T_inf/F<V_low) and (N_rest==0):   
        I_nd[j][0]=-1;
        # if the current if positive, and the right potential is lower than 0, discharge stops and charge starts (without rest)
        
    if  (I_nd[j][0]==0):
        rest_count=rest_count+1;
        
    # electrical calculation starts ----------------------------------------------
    # Define current density variation
    # Define governing equations and boundary conditions for electrochemical transport
    
    EA[0][0]=1;
    EB[0][0]=0;
    
    for i in range(1,(N_e-1)):
        EA[i][i-1]=-1/dx_nd;
        EA[i][i]=1/dx_nd;
        EA[i][i+4*N_e+N_s]=-Pi1;
        EB[i][0]=-Pi1*I_nd[j][0];
    
    EA[N_e-1][N_e-3]=1/(2*dx_nd);
    EA[N_e-1][N_e-2]=-2/dx_nd;
    EA[N_e-1][N_e-1]=3/(2*dx_nd);
    EB[N_e-1][0]=0;
    EA[N_e][N_e]=-3/(2*dx_nd);
    EA[N_e][N_e+1]=2/dx_nd;
    EA[N_e][N_e+2]=-1/(2*dx_nd);
    EB[N_e][0]=0;
    
    for i in range((N_e+1),(2*N_e-1)):
        EA[i][i-1]=-1/(2*dx_nd);
        EA[i][i+1]=1/(2*dx_nd);
        EA[i][i+4*N_e+N_s]=-Pi1;
        EB[i][0]=-Pi1*I_nd[j][0];
    
    EA[2*N_e-1][2*N_e-3]=1/(2*dx_nd);
    EA[2*N_e-1][2*N_e-2]=-2/dx_nd;
    EA[2*N_e-1][2*N_e-1]=3/(2*dx_nd);
    EB[2*N_e-1][0]=-Pi1*I_nd[j][0];
    EA[2*N_e][2*N_e]=-3/(2*dx_nd);
    EA[2*N_e][2*N_e+1]=2/dx_nd;
    EA[2*N_e][2*N_e+2]=-1/(2*dx_nd);
    EB[2*N_e][0]=0;
    
    for i in range((2*N_e+1),(3*N_e-1)):
        EA[i][i+1]=1/(2*dx_nd);
        EA[i][i-1]=-1/(2*dx_nd);
        EA[i][i+2*N_e+N_s]=Pi2/(c_nd[j-1][i-2*N_e]);
        EB[i][0]=0;

    EA[3*N_e-1][3*N_e-2]=-c_nd[j-1][N_e-1]/dx_nd;
    EA[3*N_e-1][3*N_e-1]=c_nd[j-1][N_e-1]/dx_nd+(eps_s/eps_e)**(1.5)*c_nd[j-1][N_e]/dx_nd;
    EA[3*N_e-1][3*N_e]=-(eps_s/eps_e)**1.5*c_nd[j-1][N_e]/dx_nd;
    EB[3*N_e-1][0]=0;
    
    for i in range((3*N_e),(3*N_e+N_s-1)):
        EA[i][i+1]=1/(2*dx_nd);
        EA[i][i-1]=-1/(2*dx_nd);
        EB[i][0]=-Pi2*(eps_e/eps_s)**1.5*I_nd[j][0]/(c_nd[j-1][i-2*N_e]);
    
    EA[3*N_e+N_s-1][3*N_e+N_s-2]=-(eps_s/eps_e)**1.5*c_nd[j-1][N_e+N_s-1]/(dx_nd);
    EA[3*N_e+N_s-1][3*N_e+N_s-1]=(eps_s/eps_e)**1.5*(c_nd[j-1][N_e+N_s-1])/(dx_nd)+(c_nd[j-1][N_e+N_s-1])/(dx_nd);
    EA[3*N_e+N_s-1][3*N_e+N_s]=-(c_nd[j-1][N_e+N_s-1])/(dx_nd);
    EB[3*N_e+N_s-1][0]=0;
    
    for i in range((3*N_e+N_s),(4*N_e+N_s-1)):
        EA[i][i+1]=1/(2*dx_nd);
        EA[i][i-1]=-1/(2*dx_nd);
        EA[i][i+2*N_e]=Pi2/(c_nd[j-1][i-2*N_e]);
        EB[i][0]=0;
    
    EA[4*N_e+N_s-1][4*N_e+N_s-3]=1/(2*dx_nd);
    EA[4*N_e+N_s-1][4*N_e+N_s-2]=-2/dx_nd;
    EA[4*N_e+N_s-1][4*N_e+N_s-1]=3/(2*dx_nd);
    EB[4*N_e+N_s-1][0]=0;
    EA[4*N_e+N_s][4*N_e+N_s]=1;
    EB[4*N_e+N_s][0]=0;
    
    for i in range((4*N_e+N_s+1),(5*N_e+N_s-1)):
        EA[i][i+1]=1/dx_nd;
        EA[i][i]=-1/dx_nd;
        EA[i][i-4*N_e-N_s]=-Pi3/(Pi1*dt_nd);
        EA[i][i-2*N_e-N_s]=Pi3/(Pi1*dt_nd);
        EB[i][0]=-Pi3/(Pi1*dt_nd)*(phi1_nd[j-1][i-4*N_e-N_s]-phi2_nd[j-1][i-4*N_e-N_s]);
    
    EA[5*N_e+N_s-1][5*N_e+N_s-1]=1;
    EB[5*N_e+N_s-1][0]=I_nd[j][0];
    EA[5*N_e+N_s][5*N_e+N_s]=1;
    EB[5*N_e+N_s][0]=I_nd[j][0];
    
    for i in range((5*N_e+N_s+1),(6*N_e+N_s-1)):
        EA[i][i+1]=1/(2*dx_nd);
        EA[i][i-1]=-1/(2*dx_nd);
        EA[i][i-4*N_e-N_s]=-Pi3/(Pi1*dt_nd);
        EA[i][i-2*N_e]=Pi3/(Pi1*dt_nd);
        EB[i][0]=-Pi3/(Pi1*dt_nd)*(phi1_nd[j-1][i-4*N_e-N_s]-phi2_nd[j-1][i-4*N_e]);
    
    EA[6*N_e+N_s-1][6*N_e+N_s-1]=1;
    EB[6*N_e+N_s-1][0]=0;
    EA[6*N_e+N_s][6*N_e+N_s]=-3/(2*dx_nd);
    EA[6*N_e+N_s][6*N_e+N_s+1]=2/dx_nd;
    EA[6*N_e+N_s][6*N_e+N_s+2]=-1/(2*dx_nd);
    EB[6*N_e+N_s][0]=0;
    
    for i in range((6*N_e+N_s+1),(7*N_e+N_s-1)):
        EA[i][i-1]=-1/dx_nd**2;
        EA[i][i]=eps_e/dt_nd+2/dx_nd**2;
        EA[i][i+1]=-1/dx_nd**2;
        EA[i][i-6*N_e-N_s]=-(Pi2*Pi3)/(Pi1*dt_nd);
        EA[i][i-4*N_e-N_s]=+Pi2*Pi3/(Pi1*dt_nd);
        EB[i][0]=eps_e*c_nd[j-1][i-6*N_e-N_s]/dt_nd-Pi2*Pi3/(Pi1*dt_nd)*(phi1_nd[j-1][i-6*N_e-N_s]-phi2_nd[j-1][i-6*N_e-N_s]);
    
    EA[7*N_e+N_s-1][7*N_e+N_s-2]=-1/dx_nd;
    EA[7*N_e+N_s-1][7*N_e+N_s-1]=1/dx_nd*(1+(eps_s/eps_e)**1.5);
    EA[7*N_e+N_s-1][7*N_e+N_s]=-1/dx_nd*(eps_s/eps_e)**1.5;
    EB[7*N_e+N_s-1][0]=0;
    
    for i in range((7*N_e+N_s),(7*N_e+2*N_s-1)):
        EA[i][i-1]=-1/dx_nd**2*eps_s**0.5/eps_e**1.5;
        EA[i][i]=1/dt_nd+eps_s**(0.5)/eps_e**(1.5)*2/dx_nd**2;
        EA[i][i+1]=-eps_s**(0.5)/eps_e**(1.5)*1/dx_nd**2;
        EB[i][0]=c_nd[j-1][i-6*N_e-N_s]/dt_nd;
    
    EA[7*N_e+2*N_s-1][7*N_e+2*N_s-2]=-1/dx_nd*(eps_s/eps_e)**1.5;
    EA[7*N_e+2*N_s-1][7*N_e+2*N_s-1]=(1+(eps_s/eps_e)**1.5)/dx_nd;
    EA[7*N_e+2*N_s-1][7*N_e+2*N_s]=-1/dx_nd;
    EB[7*N_e+2*N_s-1][0]=0;
    
    for i in range((7*N_e+2*N_s),(8*N_e+2*N_s-1)):
        EA[i][i-1]=-1/dx_nd**2;
        EA[i][i]=eps_e/dt_nd+2/dx_nd**2;
        EA[i][i+1]=-1/dx_nd**2;
        EA[i][i-6*N_e-2*N_s]=-Pi2*Pi3/(Pi1*dt_nd);
        EA[i][i-4*N_e-N_s]=+Pi2*Pi3/(Pi1*dt_nd);
        EB[i][0]=eps_e*c_nd[j-1][i-6*N_e-N_s]/dt_nd-Pi3*Pi2/(Pi1*dt_nd)*(phi1_nd[j-1][i-6*N_e-2*N_s]-phi2_nd[j-1][i-6*N_e-N_s]);
    
    EA[8*N_e+2*N_s-1][8*N_e+2*N_s-3]=1/(2*dx_nd);
    EA[8*N_e+2*N_s-1][8*N_e+2*N_s-2]=-2/dx_nd;
    EA[8*N_e+2*N_s-1][8*N_e+2*N_s-1]=3/(2*dx_nd);
    EB[8*N_e+2*N_s-1][0]=0;
    
    EC=np.matrix(EA);
    ED=np.linalg.inv(EC);
    EE=np.matrix(EB);
    E_tem=ED*EE;

    # Extracting electrical data ----------------------------------
    for k in range(0,(2*N_e)):
        phi1_nd[j][k]=E_tem[k][0];
        # non-dimentional solid potential
    
    for k in range((2*N_e),(4*N_e+N_s)):
        phi2_nd[j][k-2*N_e]=E_tem[k][0];
        # non-dimentional liquid potential
    
    for k in range((4*N_e+N_s),(6*N_e+N_s)):
        I1_nd[j][k-4*N_e-N_s]=E_tem[k][0];
        # non-dimentional current density of electrolyte
    
    for k in range((6*N_e+N_s),(8*N_e+2*N_s)):
        c_nd[j][k-6*N_e-N_s]=E_tem[k][0];
        # non-dimentional ion concentration
    
    # non-dimentional current density of liquid part-------------------------------
    for k in range(0,N_e):
        J2_nd[j][k]=I1_nd[j][k];
        # non-dimentional current density of liquid electrolyte in porous electrode
    
    for k in range((N_e),(N_e+N_s)):
        J2_nd[j][k]=I_nd[j][0];
        # non-dimentional current density of liquid electrolyte in porous seperator
    
    for k in range(N_e+N_s,2*N_e+N_s):
        J2_nd[j][k]=I1_nd[j][k-N_s];
        # non-dimentional current density of liquid electrolyte in porous electrode
    
    # non-dimentional current density of solid part -------------------------------
   
    J1_nd[j][0]=I_nd[j][0];
    # non-dimentional solid current density in Al-electrode interfaces

    for k in range(1,N_e-1):
        J1_nd[j][k]=I_nd[j][0]-J2_nd[j][k];
        # non-dimentional current density of left electrode
    
    for k in range(N_e-1,N_e+N_s+1):            
        J1_nd[j][k]=0;
        # non-dimentional current density of seperator
    
    for k in range(N_e+N_s+1,2*N_e+N_s-1):  
        J1_nd[j][k]=I_nd[j][0]-J2_nd[j][k];
        # non-dimentional current density of right electrode
    
    J1_nd[j][2*N_e+N_s-1]=I_nd[j][0];
    # non-dimentional solid current density in Al-electrode interfaces
    
    # electrical calculation ends -----------------------------------------------
    
    # Formulating heat generation rates -----------------------------------
    for u in range(0,N_e):
        Q_irr1_nd[j][u]=J1_nd[j][u]**2*Pi5*Pi1/Pi2*rho_cp_al/rho_cp_elo_por;
        Q_irr2_nd[j][u]=J2_nd[j][u]**2*Pi5/c_nd[j][u]*(eps_e/eps_s)**1.5*rho_cp_al/rho_cp_elo_por;
        Q_irr_nd[j][u] =Q_irr1_nd[j][u]+Q_irr2_nd[j][u];
        # Joule heat in left electrode
    
    for u in range(N_e,(N_e+N_s)):
        Q_irr1_nd[j][u]=J1_nd[j][u]**2*Pi5*Pi1/Pi2*rho_cp_al/rho_cp_sep_por;
        Q_irr2_nd[j][u]=J2_nd[j][u]**2*Pi5/c_nd[j][u]*(eps_e/eps_s)**1.5*rho_cp_al/rho_cp_sep_por;
        Q_irr_nd[j][u] =Q_irr1_nd[j][u]+Q_irr2_nd[j][u];
        # Joule heat in seperator
    
    for u in range((N_e+N_s),(2*N_e+N_s)):
        Q_irr1_nd[j][u]=J1_nd[j][u]**2*Pi5*Pi1/Pi2*rho_cp_al/rho_cp_elo_por;
        Q_irr2_nd[j][u]=J2_nd[j][u]**2*Pi5/c_nd[j][u]*(eps_e/eps_s)**1.5*rho_cp_al/rho_cp_elo_por;
        Q_irr_nd[j][u] =Q_irr1_nd[j][u]+Q_irr2_nd[j][u];
        # Joule heat in right electrode
    
    if (abs(I_nd[j][0])<0.01):
        for u in range(0,(2*N_e+N_s)):
            Q_rev_nd[j][u]=0;
            # Reversible heat = 0 during rest
    else:
        for u in range(0,N_e-1):
            Q_rev_nd[j][u]=-Pi6*rho_cp_al/rho_cp_elo_por*abs((J2_nd[j][u+1]-J2_nd[j][u])/dx_nd)*I_nd[j][0]/abs(I_nd[j][0]);
            # Reversible heat in left electrode
        
        Q_rev_nd[j][N_e-1]=Q_rev_nd[j][N_e-2];
        # Reversible heat in left electrode-seperator interface
        
        for u in range(N_e,(N_e+N_s)):
            Q_rev_nd[j][u]=0;
            # Reversible heat in seperator
        
        for u in range((N_e+N_s),(2*N_e+N_s-1)):
            Q_rev_nd[j][u]=-Pi6*rho_cp_al/rho_cp_elo_por*abs((J2_nd[j][u+1]-J2_nd[j][u])/dx_nd)*I_nd[j][0]/abs(I_nd[j][0]);
            # Reversible heat in right electrode
        
        Q_rev_nd[j][(2*N_e+N_s-1)]=Q_rev_nd[j][(2*N_e+N_s-2)];
        # Reversible heat in right electrode-seperator interface
    
    # dimentional heat generation including Al collector
    for u in range(0,N_al):
        Q_irr1[j][u]=I0**2/sigma_al;
        Q_irr2[j][u]=0;
        Q_irr[j][u] =Q_irr1[j][u]+Q_irr2[j][u];
        Q_rev[j][u] =0;
        Q_tot[j][u] =Q_irr[j][u]+Q_rev[j][u];
        
    for u in range(N_al,(2*N_e+N_s+N_al)):
        Q_irr1[j][u]=Q_irr1_nd[j][u-N_al]*(rho_cp_elo_por*T_inf*d_elo/L_e**2);
        Q_irr2[j][u]=Q_irr2_nd[j][u-N_al]*(rho_cp_elo_por*T_inf*d_elo/L_e**2);
        Q_irr[j][u] =Q_irr_nd[j][u-N_al]*(rho_cp_elo_por*T_inf*d_elo/L_e**2);
        Q_rev[j][u] =Q_rev_nd[j][u-N_al]*(rho_cp_elo_por*T_inf*d_elo/L_e**2);
        Q_tot[j][u] =Q_irr[j][u]+Q_rev[j][u];
    
    for u in range((2*N_e+N_s+N_al),(2*N_e+N_s+2*N_al)):
        Q_irr1[j][u]=I0**2/sigma_al;
        Q_irr2[j][u]=0;
        Q_irr[j][u] =Q_irr1[j][u]+Q_irr2[j][u];
        Q_rev[j][u] =0;
        Q_tot[j][u] =Q_irr[j][u]+Q_rev[j][u];
        
    # non-dimentional heat generation including Al collector
    for u in range(0,(2*N_e+N_s+2*N_al)):
        Q_tot_nd[j][u]= Q_tot[j][u]/(rho_cp_al*T_inf*d_elo/L_e**2);
        
    # Heat generation calculation ends ---------------------------------------------
    
    # Temperature calculation starts ---------------------------------------------

    TA[0][0]=-1/dx_nd-Pi7;
    TA[0][1]=1/dx_nd;
    TB[0][0]=0;
    
    for k in range(1,(N_al-1)):
        TA[k][k]=1+2*Pi4*dt_nd/dx_nd**2;
        TA[k][k-1]=-Pi4*dt_nd/dx_nd**2;
        TA[k][k+1]=-Pi4*dt_nd/dx_nd**2;
        TB[k][0]=T_nd[j-1][k]+Q_tot_nd[j][k]*dt_nd;

    TA[N_al-1][N_al-2]=-k_al/(k_al+k_elo_por);
    TA[N_al-1][N_al-1]=1;
    TA[N_al-1][N_al]=-k_elo_por/(k_al+k_elo_por);
    TB[N_al-1][0]=0;
    
    for k in range(N_al,(N_e+N_al-1)):
        TA[k][k]=1+2*dt_nd*Pi4*alpha_por_elc/alpha_al/dx_nd**2;
        TA[k][k-1]=-Pi4*dt_nd*alpha_por_elc/alpha_al/dx_nd**2;
        TA[k][k+1]=-Pi4*dt_nd*alpha_por_elc/alpha_al/dx_nd**2;
        TB[k][0]=T_nd[j-1][k]+Q_tot_nd[j][k]*dt_nd;
    
    TA[(N_e+N_al-1)][(N_e+N_al-2)]=-k_elo_por/(k_elo_por+k_sep_por);
    TA[(N_e+N_al-1)][(N_e+N_al-1)]=1;
    TA[(N_e+N_al-1)][(N_e+N_al)]=-k_sep_por/(k_elo_por+k_sep_por);
    TB[(N_e+N_al-1)][0]=0;
    
    for k in range((N_e+N_al),(N_e+N_s+N_al-1)):
        TA[k][k]=1+2*Pi4*dt_nd*alpha_sep_por/alpha_al/dx_nd**2;
        TA[k][k-1]=-Pi4*dt_nd*alpha_sep_por/alpha_al/dx_nd**2;
        TA[k][k+1]=-Pi4*dt_nd*alpha_sep_por/alpha_al/dx_nd**2;
        TB[k][0]=T_nd[j-1][k]+Q_tot_nd[j][k]*dt_nd;
    
    TA[(N_e+N_s+N_al-1)][(N_e+N_s+N_al-2)]=-k_sep_por/(k_elo_por+k_sep_por);
    TA[(N_e+N_s+N_al-1)][(N_e+N_s+N_al-1)]=1;
    TA[(N_e+N_s+N_al-1)][(N_e+N_s+N_al)]=-k_elo_por/(k_elo_por+k_sep_por);
    TB[(N_e+N_s+N_al-1)][0]=0;
    
    for k in range((N_e+N_s+N_al),(2*N_e+N_s+N_al-1)):
        TA[k][k]=1+2*dt_nd*Pi4*alpha_por_elc/alpha_al/dx_nd**2;
        TA[k][k-1]=-Pi4*dt_nd*alpha_por_elc/alpha_al/dx_nd**2;
        TA[k][k+1]=-Pi4*dt_nd*alpha_por_elc/alpha_al/dx_nd**2;
        TB[k][0]=T_nd[j-1][k]+Q_tot_nd[j][k]*dt_nd;
    
    TA[(2*N_e+N_s+N_al-1)][(2*N_e+N_s+N_al-2)]=-k_elo_por/(k_al+k_elo_por);
    TA[(2*N_e+N_s+N_al-1)][(2*N_e+N_s+N_al-1)]=1;
    TA[(2*N_e+N_s+N_al-1)][(2*N_e+N_s+N_al)]=-k_al/(k_al+k_elo_por);
    TB[(2*N_e+N_s+N_al-1)][0]=0;
    
    for k in range((2*N_e+N_s+N_al),(2*N_e+N_s+2*N_al-1)):
        TA[k][k]=1+2*dt_nd*Pi4/dx_nd**2;
        TA[k][k-1]=-Pi4*dt_nd/dx_nd**2;
        TA[k][k+1]=-Pi4*dt_nd/dx_nd**2;
        TB[k][0]=T_nd[j-1][k]+Q_tot_nd[j][k]*dt_nd;
      
    TA[(2*N_e+N_s+2*N_al-1)][(2*N_e+N_s+2*N_al-2)]=-1/dx_nd;
    TA[(2*N_e+N_s+2*N_al-1)][(2*N_e+N_s+2*N_al-1)]=1/dx_nd+Pi7;
    TB[(2*N_e+N_s+2*N_al-1)][0]=0;
    
    TC=np.matrix(TA);
    TD=np.linalg.inv(TC);
    TE=np.matrix(TB);
    T_tem=TD*TE;
    
    # Extracting temperature ----------------------------------
    for k in range(0,(2*N_e+N_s+2*N_al)):
        T_nd[j][k]=T_tem[k][0];

    # Temperature calculation ends ---------------------------------------------
    
    count=count+1;
    if (count>99) and (count%100==0):
        time_disp=count//10;
        print("Simulation completes %s s" % time_disp);
        # show the progress of the simulation
# Main sumulation ends ---------------------------------------------------------

print('Simulation of heat generation in supercapacitor has completed')


# In[3]:


# data storage -----------------------------------------------------------------
# matrices for electrical data
time=np.zeros(shape=(N_t,1));                          # time materix
phi1_d=np.zeros(shape=(N_t,N1));                       # potential matrix of solid phase
phi2_d=np.zeros(shape=(N_t,N2));                       # potential matrix of liquid phase
c_d=np.zeros(shape=(N_t,N2));                          # ion concentration in electrolyte
J1_d=np.zeros(shape=(N_t,N2));                         # current density in solid phase
J2_d=np.zeros(shape=(N_t,N2));                         # current density in liquid phase

# matrices for thermal data
Q_irr_av=np.zeros(shape=(N_t,1));                      # matrix of total joule heat
Q_irr1_av=np.zeros(shape=(N_t,1));                     # matrix of Joule heat of solid part
Q_irr2_av=np.zeros(shape=(N_t,1));                     # matrix of Joule heat of liquid part
Q_rev_av=np.zeros(shape=(N_t,1));                      # matrix of reversible heat
Q_tot_av=np.zeros(shape=(N_t,1));                      # matrix of total heat generation

for j in range(0,N_t):
    time[j][0]=dt_d*j;                           
    # dimentional time points
    
    for i in range(0,2*N_e):
        phi1_d[j][i]=phi1_nd[j][i]*Ru*T_inf/F;        
        # dimentional solid potential

    for i in range(0,(2*N_e+N_s)):
        phi2_d[j][i]=phi2_nd[j][i]*Ru*T_inf/F;         # dimentional liquid potential
        c_d[j][i]=c_nd[j][i]*c_inf;                    # dimentional ion concentration
        J1_d[j][i]=J1_nd[j][i]*I0;                     # dimentional solid current density
        J2_d[j][i]=J2_nd[j][i]*I0;                     # dimentinoal liquid current density
    
    for i in range(0,(2*N_e+N_s+2*N_al)):
        Q_irr1_av[j][0]=Q_irr1_av[j][0]+Q_irr1[j][i]/(2*N_e+N_s+2*N_al);
        Q_irr2_av[j][0]=Q_irr2_av[j][0]+Q_irr2[j][i]/(2*N_e+N_s+2*N_al);
        Q_irr_av[j][0] =Q_irr_av[j][0]+Q_irr[j][i]/(2*N_e+N_s+2*N_al);
        Q_rev_av[j][0] =Q_rev_av[j][0]+Q_rev[j][i]/(2*N_e+N_s+2*N_al);
        Q_tot_av[j][0] =Q_tot_av[j][0]+Q_tot[j][i]/(2*N_e+N_s+2*N_al);
        T_d[j][i]=T_inf*(1+T_nd[j][i]);
# -------------------------------------------------------------------------

# matrices of extraction --------------------------------------------------
n_t=10;
n_p=10;
n_p_h=math.floor(n_p /2);
N_t_st=math.floor(N_t/n_t);
N1_st=math.floor(N1/n_p);
N2_st=math.floor(N2/n_p);
N3_st=math.floor(N3/n_p);

time_st=np.zeros(shape=(N_t_st,1));                         # time materix
phi1_nd_st=np.zeros(shape=(N_t_st,N1_st));                  # potential matrix of solid phase
phi2_nd_st=np.zeros(shape=(N_t_st,N2_st));                  # potential matrix of liquid phase
c_nd_st=np.zeros(shape=(N_t_st,N2_st));                     # ion concentration in electrolyte
J1_nd_st=np.zeros(shape=(N_t_st,N2_st));                    # current density in solid phase
J2_nd_st=np.zeros(shape=(N_t_st,N2_st));                    # current density in liquid phase

# dimensional matrices of electrical data
phi1_d_st=np.zeros(shape=(N_t_st,N1_st));                   # potential matrix of solid phase
phi2_d_st=np.zeros(shape=(N_t_st,N2_st));                   # potential matrix of liquid phase
c_d_st=np.zeros(shape=(N_t_st,N2_st));                      # ion concentration in electrolyte
J1_d_st=np.zeros(shape=(N_t_st,N2_st));                     # current density in solid phase
J2_d_st=np.zeros(shape=(N_t_st,N2_st));                     # current density in liquid phase

# extraction matrices of thermal data
Q_irr_st=np.zeros(shape=(N_t_st,N3_st));                    # total joule heat
Q_irr1_st=np.zeros(shape=(N_t_st,N3_st));                   # joule heat of solid part
Q_irr2_st=np.zeros(shape=(N_t_st,N3_st));                   # joule heat of liquid part
Q_rev_st=np.zeros(shape=(N_t_st,N3_st));                    # reversible heat
Q_tot_st=np.zeros(shape=(N_t_st,N3_st));                    # total heat generation
T_nd_st=np.zeros(shape=(N_t_st,N3_st));                     # non-dimentional temperature
T_d_st=np.zeros(shape=(N_t_st,N3_st));                      # dimentional temperature

# extraction matrices of average heat
Q_irr_av_st=np.zeros(shape=(N_t_st,1));                     # storing average total joule heat
Q_irr1_av_st=np.zeros(shape=(N_t_st,1));                    # storing average joule heat of solid part
Q_irr2_av_st=np.zeros(shape=(N_t_st,1));                    # storing average joule heat of liquid part
Q_rev_av_st=np.zeros(shape=(N_t_st,1));                     # storing average reversible heat
Q_tot_av_st=np.zeros(shape=(N_t_st,1));                     # storing average total heat generation

# data extraction -------------------------------------------
for j in range(0,N_t_st):
    time_st[j][0]=dt_d*(j-1)*n_t;                           # dimentional time points

    for i in range(0,N1_st):
        phi1_d_st[j][i]=phi1_d[j*n_t][i*n_p+n_p_h];         # dimentional solid potential
        phi1_nd_st[j][i]=phi1_nd[j*n_t][i*n_p+n_p_h];       # non-dimentional solid potential
  
    for i in range(0,N2_st):
        phi2_d_st[j][i]=phi2_d[j*n_t][i*n_p+n_p_h];         # dimentional liquid potential
        phi2_nd_st[j][i]=phi2_nd[j*n_t][i*n_p+n_p_h];       # non-dimentional liquid potential
        c_d_st[j][i]=c_d[j*n_t][i*n_p+n_p_h];               # dimentional ion concentration
        c_nd_st[j][i]=c_nd[j*n_t][i*n_p+n_p_h];             # non-dimentional ion concentration
        J1_d_st[j][i]=J1_d[j*n_t][i*n_p+n_p_h];             # dimentional solid current density
        J1_nd_st[j][i]=J1_nd[j*n_t][i*n_p+n_p_h];           # non-dimentional solid current density
        J2_d_st[j][i]=J2_d[j*n_t][i*n_p+n_p_h];             # dimentinoal liquid current density
        J2_nd_st[j][i]=J2_nd[j*n_t][i*n_p+n_p_h];           # non-dimentinoal liquid current density       

    
    for i in range(0,N3_st):
        Q_irr1_st[j][i]=Q_irr1[j*n_t][i*n_p+n_p_h];         # Joule heat 1
        Q_irr2_st[j][i]=Q_irr2[j*n_t][i*n_p+n_p_h];         # Joule heat 2
        Q_irr_st[j][i] =Q_irr[j*n_t][i*n_p+n_p_h];          # Joule heat
        Q_rev_st[j][i] =Q_rev[j*n_t][i*n_p+n_p_h];          # reversible heat
        Q_tot_st[j][i] =Q_tot[j*n_t][i*n_p+n_p_h];          # total heat
        Q_tot_st[j][i] =Q_tot[j*n_t][i*n_p+n_p_h];          # total heat
        T_nd_st[j][i]  =T_nd[j*n_t][i*n_p+n_p_h];           # non-dimentional temperature
        T_d_st[j][i]   =T_d[j*n_t][i*n_p+n_p_h];            # dimentional temperature    
    
    Q_irr1_av_st[j][0]=Q_irr1_av[j*n_t][0];                 # average Joule heat 1
    Q_irr2_av_st[j][0]=Q_irr2_av[j*n_t][0];                 # average Joule heat 2
    Q_irr_av_st[j][0]=Q_irr_av[j*n_t][0];                   # average Joule heat
    Q_rev_av_st[j][0]=Q_rev_av[j*n_t][0];                   # average reversible heat
    Q_tot_av_st[j][0]=Q_tot_av[j*n_t][0];                   # average total heat
# data extraction -------------------------------------------
    
# Store spatial position ----------------------------------------------------------
x1=np.zeros(shape=(N1,1));                          # position matrix for potential
x2=np.zeros(shape=(N2,1));                          # position matrix for current
x3=np.zeros(shape=(N3,1));                          # position matrix for heat and temperature
x1_st=np.zeros(shape=(N1_st,1));                    # position matrix for potential
x2_st=np.zeros(shape=(N2_st,1));                    # position matrix for current
x3_st=np.zeros(shape=(N3_st,1));                    # position matrix for heat and temperature
N_e_st=math.floor(N_e/n_p);                         

for i in range(0,N_e):
    x1[i][0]=i*dx_d+dx_d/2;                         
for i in range(N_e,2*N_e):
    x1[i][0]=L_s+i*dx_d+dx_d/2;                     
    # patial position for solid potential

for i in range(0,N_e_st):
    x1_st[i][0]=i*dx_d*n_p+dx_d/2*n_p;          
for i in range(N_e_st,2*N_e_st):
    x1_st[i][0]=L_s+i*dx_d*n_p+dx_d/2*n_p;      
    # spatial position for extracting solid potential

for i in range(0,N2):
    x2[i][0]=i*dx_d+dx_d/2;                     
    # patial position for current and ion concentration

for i in range(0,N2_st):
    x2_st[i][0]=i*dx_d*n_p+dx_d/2*n_p;          
    # spatial position for extracting current and ion concentration

for i in range(0,N3):
    x3[i][0]=i*dx_d+dx_d/2;                     
    # spatial position for temperature and heat generation

for i in range(0,N3_st):
    x3_st[i][0]=i*dx_d*n_p+dx_d/2*n_p;          
    # spatial position for extracting temperature and heat generation

# Charge calculation ------------------------------------------------------
Q_charge=np.zeros(shape=(N_t,2*N_e));
for i in range(0,N_t):
    for j in range(0,N_e):
        Q_charge[i][j]=aC*(phi1_d[i][j]-phi2_d[i][j]);

    for j in range(N_e,2*N_e):
        Q_charge[i][j]=aC*(phi1_d[i][j]-phi2_d[i][j+N_s]);

# Charge extraction ------------------------------------------------------
Q_charge_st=np.zeros(shape=(N_t_st,2*N_e_st));
for i in range(0,N_t_st):
    for j in range(0,2*N_e_st):
        Q_charge_st[i][j]= Q_charge[i*n_t][j*n_p];

# Charge calculation and extraction ---------------------------------------

# Selection of charge and discharge duration -----------------------------------------
cycle_count=0;
for i in range(1,N_t-1):
    if (I_nd[i][0]>0) and (I_nd[i+1][0]==0) and (N_rest!=0):   # end of discharge
        cycle_count=cycle_count+1;

    if (I_nd[i][0]>0) and (I_nd[i+1][0]<0) and (N_rest==0):   # end of discharge
        cycle_count=cycle_count+1; 
    
cyc=np.zeros(shape=(cycle_count,1));                        # matrix of cycle number
t_ch_ori=np.zeros(shape=(cycle_count,1));                   # matrix of time start point of charge
t_ch_end=np.zeros(shape=(cycle_count,1));                   # matrix of time end point of charge
t_dc_ori=np.zeros(shape=(cycle_count,1));                   # matrix of time start point of discharge
t_dc_end=np.zeros(shape=(cycle_count,1));                   # matrix of time end point of discharge
V_ch_ori=np.zeros(shape=(cycle_count,1));                   # matrix of start voltage of charge
V_ch_end=np.zeros(shape=(cycle_count,1));                   # matrix of end voltage of charge
V_dc_ori=np.zeros(shape=(cycle_count,1));                   # matrix of start voltage of discharge
V_dc_end=np.zeros(shape=(cycle_count,1));                   # matrix of end voltage of discharge
pos_ch_ori=np.zeros(shape=(cycle_count,1));                 # matrix of start point of charge
pos_ch_end=np.zeros(shape=(cycle_count,1));                 # matrix of end point of charge
pos_dc_ori=np.zeros(shape=(cycle_count,1));                 # matrix of start point of discharge
pos_dc_end=np.zeros(shape=(cycle_count,1));                 # matrix of end point of discharge
Q_av_ch=np.zeros(shape=(cycle_count,1));                    # matrix of average heat in one charge
Q_av_dc=np.zeros(shape=(cycle_count,1));                    # matrix of  average heat in one discharge

cycle=0;
t_ch_ori[0][0]=time[0][0];
V_ch_ori[0][0]=phi1_d[0][2*N_e-1];
pos_ch_ori[0][0]=0;
for i in range(1,N_t-1):
    if (cycle < cycle_count):
        if (I_nd[i][0]<0) and (I_nd[i+1][0]>0):   # end of charge
            t_ch_end[cycle][0]=time[i][0];
            V_ch_end[cycle][0]=phi1_d[i][2*N_e-1];
            pos_ch_end[cycle][0]=i;

        if (I_nd[i-1][0]<0) and (I_nd[i][0]>0):   # start of discharge
            t_dc_ori[cycle][0]=time[i][0];
            V_dc_ori[cycle][0]=phi1_d[i][2*N_e-1];
            pos_dc_ori[cycle][0]=i;

        if (I_nd[i][0]>0) and (I_nd[i+1][0]==0) and (N_rest!=0):   # end of discharge
            t_dc_end[cycle][0]=time[i][0];
            V_dc_end[cycle][0]=phi1_d[i][2*N_e-1];
            pos_dc_end[cycle][0]=i;
            cyc[cycle][0]=cycle+1;
            cycle=cycle+1;

        if (I_nd[i-1][0]==0) and (I_nd[i][0]<0) and (N_rest!=0):   # start of charge
            t_ch_ori[cycle][0]=time[i][0];
            V_ch_ori[cycle][0]=phi1_d[i][2*N_e-1];
            pos_ch_ori[cycle][0]=i;
            cyc[cycle][0]=cycle+1;

        if (I_nd[i][0]>0) and (I_nd[i+1][0]<0) and (N_rest==0):   # end of discharge
            t_dc_end[cycle][0]=time[i][0];
            V_dc_end[cycle][0]=phi1_d[i][2*N_e-1];
            pos_dc_end[cycle][0]=i;
            cyc[cycle][0]=cycle+1;
            cycle=cycle+1;

        if (I_nd[i-1][0]>0) and (I_nd[i][0]<0) and (N_rest==0):   # start of charge
            t_ch_ori[cycle][0]=time[i][0];
            V_ch_ori[cycle][0]=phi1_d[i][2*N_e-1];
            pos_ch_ori[cycle][0]=i;
            cyc[cycle][0]=cycle+1;
     
# Selection of charge and discharge duration -----------------------------------------

# Capacitance & impedance & energy density & power density ----------------

cap=np.zeros(shape=(cycle_count,1));                        # capacitance of one discahrge
esr=np.zeros(shape=(cycle_count,1));                        # equivalent series resistance of one discahrge
E_sc=np.zeros(shape=(cycle_count,1));                       # energy density of one discahrge
P_sc=np.zeros(shape=(cycle_count,1));                       # spower density of one discahrge

for i in range(0,cycle_count):
    cap[i][0]=I0*(t_dc_end[i][0]-t_dc_ori[i][0])/(V_dc_ori[i][0]-V_dc_end[i][0]);
    esr[i][0]=(V_up-V_dc_ori[i][0])/(2*I0);
    E_sc[i][0]=0.5*cap[i][0]*(V_dc_ori[i][0]**2-V_dc_end[i][0]**2)/(rho_net*3600);
    P_sc[i][0]=E_sc[i][0]*3600/(t_dc_end[i][0]-t_dc_ori[i][0]);
    
# Capacitance & impedance & energy density & power density ----------------

# Average heat generation of each cycle -----------------------------------

for j in range(0,cycle):
    point1=math.floor(pos_ch_ori[j][0]);         # start point of charge
    point2=math.floor(pos_ch_end[j][0]);         # end point of charge
    point3=math.floor(pos_dc_ori[j][0]);         # start point of discharge
    point4=math.floor(pos_dc_end[j][0]);         # end point of discharge
    
    for i in range(point1,point2):
        Q_av_ch[j][0]=Q_av_ch[j][0]+Q_tot_av[i][0]/(point2-point1-1);
        # intergrate heat during charge

    for i in range(point3,point4):  
        Q_av_dc[j][0]=Q_av_dc[j][0]+Q_tot_av[i][0]/(point4-point3-1);
        # intergrate heat during discharge

# Average heat generation calculation -------------------------------------
    
# data for display --------------------------------------------------------
t_o=np.zeros(shape=(N_t_st,1));                     # time
I_o=np.zeros(shape=(N_t_st,1));                     # implied current
V_o=np.zeros(shape=(N_t_st,1));                     # electrode voltage
T_o1=np.zeros(shape=(N_t_st,1));                    # enter temperature
T_o2=np.zeros(shape=(N_t_st,1));                    # boudary temperature
N_temp1=math.floor((2*N_e+N_s+2*N_al)/2-1);         # point of center
N_temp2=math.floor(2*N_e+N_s+2*N_al-1);             # point of right Al surface

for j in range(0,N_t_st):
    t_o[j][0]=dt_d*j*n_t;
    I_o[j][0]=I_nd[j*n_t][0]*I0;
    V_o[j][0]=phi1_d[j*n_t][2*N_e-1];
    T_o1[j][0]=T_d[j*n_t][N_temp1];
    T_o2[j][0]=T_d[j*n_t][N_temp2];

print('Data extraction has completed')


# In[4]:


# Display in Figures ------------------------------------------------------
time_disp=math.floor(duration/n_t);    # display step number
plt.plot(t_o,I_o)
plt.xlabel("Time(s)")
plt.ylabel("Current(A)")
plt.show()

plt.plot(t_o,V_o)
plt.xlabel("Time(s)")
plt.ylabel("Voltage(V)")
plt.show()

plt.plot(t_o,T_o1)
plt.xlabel("Time(s)")
plt.ylabel("Temperature(C)")
plt.show()

plt.plot(t_o,T_o2)
plt.xlabel("Time(s)")
plt.ylabel("Temperature(C)")
plt.show()


# In[5]:


plt.plot(x2_st,c_d_st[time_disp][:])
plt.xlabel("Spatial position (m)")
plt.ylabel("Ion concentration (mol m-3)")
plt.show()

plt.plot(x1_st,phi1_d_st[time_disp][:])
plt.xlabel("Spatial position (m)")
plt.ylabel("Potential (V)")
plt.show()

plt.plot(t_o,Q_tot_av_st)
plt.xlabel("Time(s)")
plt.ylabel("Heat generation (J)")
plt.show()

plt.plot(t_o,Q_irr_av_st)
plt.xlabel("Time(s)")
plt.ylabel("Joule heat generation (J)")
plt.show()

plt.plot(t_o,Q_rev_av_st)
plt.xlabel("Time(s)")
plt.ylabel("Reversible heat generation (J)")
plt.show()


# In[6]:


# Capacitance & impedance & energy density & power density & average heat generation for display ----------------
print("Total charge/discharge cycle = %s" %cycle)
    
plt.plot(cyc,cap)
plt.xlabel("Discharge cycle")
plt.ylabel("Capacitance (F)")
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()

plt.plot(cyc,esr)
plt.xlabel("Discharge cycle")
plt.ylabel("Equivalent series resistance (ohm)")
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()

plt.plot(cyc,E_sc)
plt.xlabel("Discharge cycle")
plt.ylabel("Energy density (kWh m-3)")
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()

plt.plot(cyc,P_sc)
plt.xlabel("Discharge cycle")
plt.ylabel("Sower density (W m-3)")
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()

plt.plot(cyc,Q_av_dc)
plt.xlabel("Discharge cycle")
plt.ylabel("Average heat generation (J m-3)")
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()


# In[7]:


# Writing display data ------------------------------------------------------------
writer = pd.ExcelWriter('./Output/Display_data.xlsx')                    # extrac to excel file
data = {
    'Time':t_o[:,0],
    'Current':I_o[:,0],
    'Voltage':V_o[:,0],
    'Center temperature':T_o1[:,0],
    'Surface temperature':T_o1[:,0],
    'Average heat generation':Q_tot_av_st[:,0],
    'Average Joule heat':Q_irr_av_st[:,0],
    'Average Joule heat 1':Q_irr1_av_st[:,0],
    'Average Joule heat 2':Q_irr2_av_st[:,0],
    'Average reversible heat':Q_rev_av_st[:,0],
}
frame = pd.DataFrame(data)
frame.to_excel(writer, 'Sheet1', float_format='%.5f') # name the sheet
workbook = writer.book
worksheet = writer.sheets['Sheet1']
format = workbook.add_format({'bold': True, 'font_name': 'Times New Roman', 'align': 'left', 'valign': 'vcenter', 'text_wrap': True})
writer.save()
writer.close()


# In[8]:


# Writing electrical data ------------------------------------------------------------
writer = pd.ExcelWriter('./Output/Electrical_data.xlsx')                    # extrac to excel file
frame = pd.DataFrame(time_st)
frame.to_excel(writer, 'time', float_format='%.5f') # name the sheet

frame = pd.DataFrame(x1_st)
frame.to_excel(writer, 'x1', float_format='%.5f') # name the sheet

frame = pd.DataFrame(x2_st)
frame.to_excel(writer, 'x2', float_format='%.5f') # name the sheet

frame = pd.DataFrame(phi1_d_st)
frame.to_excel(writer, 'phi1', float_format='%.5f') # name the sheet

frame = pd.DataFrame(phi2_d_st)
frame.to_excel(writer, 'phi2', float_format='%.5f') # name the sheet

frame = pd.DataFrame(J1_d_st)
frame.to_excel(writer, 'J1', float_format='%.5f') # name the sheet

frame = pd.DataFrame(J2_d_st)
frame.to_excel(writer, 'J2', float_format='%.5f') # name the sheet

frame = pd.DataFrame(c_d_st)
frame.to_excel(writer, 'c', float_format='%.5f') # name the sheet

writer.save()
writer.close()


# In[9]:


# Writing thermal data ------------------------------------------------------------
writer = pd.ExcelWriter('./Output/Thermal_data.xlsx')                    # extrac to excel file
frame = pd.DataFrame(time_st)
frame.to_excel(writer, 'time', float_format='%.5f') # name the sheet

frame = pd.DataFrame(x3_st)
frame.to_excel(writer, 'x3', float_format='%.5f') # name the sheet

frame = pd.DataFrame(T_d_st)
frame.to_excel(writer, 'temperature', float_format='%.5f') # name the sheet

frame = pd.DataFrame(Q_tot_st)
frame.to_excel(writer, 'Q_tot', float_format='%.5f') # name the sheet

frame = pd.DataFrame(Q_irr_st)
frame.to_excel(writer, 'Q_irr', float_format='%.5f') # name the sheet

frame = pd.DataFrame(Q_irr1_st)
frame.to_excel(writer, 'Q_irr1', float_format='%.5f') # name the sheet

frame = pd.DataFrame(Q_irr2_st)
frame.to_excel(writer, 'Q_irr2', float_format='%.5f') # name the sheet

frame = pd.DataFrame(Q_rev_st)
frame.to_excel(writer, 'Q_rev', float_format='%.5f') # name the sheet

writer.save()
writer.close()


# In[10]:


# Writing capacitance data
writer = pd.ExcelWriter('./Output/Capacitance_data.xlsx')                    # extrac to excel file
data1 = {
    'Cycle':cyc[:,0],
    'Charge start':t_ch_ori[:,0],
    'Charge end':t_ch_end[:,0],
    'Low voltage':V_ch_ori[:,0],
    'High voltage':V_ch_end[:,0],
    'Average heat generation':Q_av_ch[:,0],
    'Start point':pos_ch_ori[:,0],
    'End point':pos_ch_end[:,0],
}
frame = pd.DataFrame(data1)
frame.to_excel(writer, 'Charge', float_format='%.5f') # name the sheet


data2 = {  
    'Cycle':cyc[:,0],
    'Discharge start':t_dc_ori[:,0],
    'Discharge end':t_dc_end[:,0],
    'High voltage':V_dc_ori[:,0],
    'Low voltage':V_dc_end[:,0],
    'Average heat generation':Q_av_ch[:,0],
    'Start point':pos_dc_ori[:,0],
    'End point':pos_dc_end[:,0],
}
frame = pd.DataFrame(data2)
frame.to_excel(writer, 'Discharge', float_format='%.5f') # name the sheet


data3 = {  
    'Cycle':cyc[:,0],
    'Capacitance':cap[:,0],
    'Equivalent series resistance ':esr[:,0],
    'Energy density':E_sc[:,0],
    'Power density':P_sc[:,0],
}
frame = pd.DataFrame(data3)
frame.to_excel(writer, 'Performance', float_format='%.5f') # name the sheet

writer.save()
writer.close()

print('all is done')

