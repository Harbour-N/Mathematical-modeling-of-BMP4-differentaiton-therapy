from numba import jit
import numpy as np
import scipy.stats as stats
from scipy.stats import lognorm
import itertools

import matplotlib.pyplot as plt
import pandas as pd


@jit
def gsc_model_dudt(u, n, Ps, k, ms, mv, delta_s, delta_v,r):

    '''
    Function to calculate the RHS of the ODEs of the GSC model
    
    input:
    u ([s, v1, v2, v3, ..., vn] ) = an np.array of length n+1 containing the current value of each compartment of the model
    n = number of generations/compartments
    Ps = GSC probability of self renewal
    k = carrying capacity
    ms = proliferation rate of GSCs
    mv = vector of proliferation rates for non-GSC v1,v2,...,vn 
    delta_s = death rate of GSCs
    delta_v = vector of death rates for non-GSC v1,v2,...,vn

    return:
    dudt ([ds/dt, dv1/dt, dv2/dt, dv3/dt, ..., dvn/dt] ) = an np.array of length n+1 containing the RHS of the ODEs of the GSC model
    '''
    
    dudt = np.zeros(n+1)
    N_ = np.sum(u)
    # u = [s, v1, v2, v3, ..., vn]
    dudt[0] = (2*Ps - 1) * ms * u[0] * (1 - (N_)/k) - delta_s*u[0]
    frac_asymmetric = 2*(1-Ps)*ms*u[0]* (1 - (N_)/k)
    dudt[1] = r[0] * frac_asymmetric - mv[0]*u[1]*(1 - (N_)/k) - delta_v[0]*u[1]
    if n > 1:
    # NOTE: 2:n indexes elements 2, 3, 4, ... n-1
    # Remeber that both mv and delta_v have a total of n-1 values as they are only for the PCs
        for i in range(2,n):
            dudt[i] = r[i-1] * frac_asymmetric +  2*mv[i-2]*u[i-1]*(1 - (N_)/k) - mv[i-1]*u[i]*(1 - (N_)/k) - delta_v[i-1]*u[i]

        dudt[n] =   r[n-1] * frac_asymmetric + 2*mv[n-2]*u[n-1]*(1 - (N_)/k) - delta_v[n-1]*u[n]

    return dudt

#@jit
def alpha_i(N, mv,k, delta_v, i):
    """
    alpha_i relates v_i to v_{i-1}, alpha_i*v_i = v_{i-1}

    input: 
    N  = A scalar, vector or matrix of values of total population size
    mv = A (nx1) vector of proliferation rates of v1,v2,...,vn
    k =  Scalar carrying capacity
    delta_v = A (nx1) vector of decay rates of v1,v2,...,vn
    i =  A positive integer generation number

    output: 
    alpha_i = the same shape as N, values of alpha_i for each value of N
    """
    return 2*mv[i-1]*(1-N/k) / (delta_v[i]+mv[i]*(1-N/k))

#@jit
def alpha_1(N, mv, k, delta_v, n):
    N = np.asarray(N)
    denominator = np.ones_like(N, dtype=np.float64)
    alpha_product = np.ones_like(N, dtype=np.float64)
    for i in range(1, n):
        alpha_product = alpha_product * alpha_i(N, mv, k, delta_v, i)
        denominator = denominator + alpha_product
    return 1.0 / denominator

#@jit
# S_prolif defines total V proliferation rate 
# input: 
# N is a scalar, of total population size
# mv is a (nx1) vector of proliferation rates of v1,v2,...,vn
# k is a scalar carrying capacity
# delta_v is a (nx1) vector of decay rates of v1,v2,...,vn
# n is a positive integer number of generations
#
# output: 
# S_prolif is the same shape as N, values of S_prolif for each value of N
#@jit
def S_prolif(N,mv,k,delta_v,n):
    # first term in numerator and denominator
    alpha_product = 1.0
    numerator = mv[0]
    denominator = 1.0
    # accumulate n-2 terms in numerator and denominator
    for i in range(1,n-1):
      alpha_product = alpha_product*alpha_i(N,mv,k,delta_v,i)
      numerator = numerator + mv[i]*alpha_product
      denominator = denominator + alpha_product

    # add final term to denominator
    alpha_product = alpha_product * alpha_i(N,mv,k,delta_v,n-1)
    denominator = denominator + alpha_product
    return numerator/denominator

#@jit
# input: 
# S_death defines total V proliferation rate 
# N is a scalar, vector or matrix of values of total population size
# mv is a (nx1) vector of proliferation rates of v1,v2,...,vn
# k is a scalar carrying capacity
# delta_v is a (nx1) vector of decay rates of v1,v2,...,vn
# n is a positive integer number of generations
#
# output: 
# S_death is the same shape as N, values of S_death for each value of N
#@jit
def S_death(N,mv,k,delta_v,n):
    # first term in numerator and denominator
    alpha_product = 1.0
    numerator = delta_v[0]
    denominator = 1.0
    # accumulate n-1 terms in numerator and denominator
    for i in range(1,n-1):
      alpha_product = alpha_product*alpha_i(N,mv,k,delta_v,i)
      numerator = numerator + delta_v[i]*alpha_product
      denominator = denominator + alpha_product
    return numerator/denominator

#@jit
def gsc_model_reduced_dudt(t, u, n, Ps, k, ms, mv, delta_s, delta_v):
    
    s, V = u
    N_ = s+V
    dsdt = (2*Ps - 1) * ms * s * (1 - (N_)/k) - delta_s*s
    dVdt = 2*(1 - Ps) * ms * s * (1 - (N_)/k) + S_prolif(N_,mv,k,delta_v,n)*V*(1 - (N_)/k) - S_death(N_,mv,k,delta_v,n)*V
    return np.array([dsdt, dVdt])

#@jit
def gsc_model_reduced_dVdt(s,V,n,Ps,k,ms,mv,delta_v):
    
    N_ = s+V
    dVdt = 2*(1 - Ps) * ms * s * (1 - (N_)/k) + S_prolif(N_,mv,k,delta_v,n)*V*(1 - (N_)/k) - S_death(N_,mv,k,delta_v,n)*V
    return dVdt

#@jit 
def simulate_simple_reduced(t_final,dt,u0,n,Ps,k,ms,mv,delta_s,delta_v,r):
    t = np.arange(0, t_final+dt, dt)
    u = np.zeros((len(t), 2))
    u[0] = u0
    for i in range(len(t)-1):
        u[i+1,:] = u[i,:] + dt * gsc_model_reduced_dudt(t, u[i,:], n, Ps, k, ms, mv, delta_s, delta_v)
    return t,u

@jit
def radiation(u,alpha,beta,eta,mu,d,n):
    
    gamma = np.exp(-eta*(alpha*d + beta*d*d))
    u[0] =   u[0]*gamma
    if n >1:
        gamma = np.exp(-(alpha*d + beta*d*d))
        u[1:-1] = u[1:-1]*gamma
    gamma = np.exp(-mu*(alpha*d + beta*d*d))
    u[-1] =  u[-1]*gamma
    
    return u
@jit
def resection(u,resect_fraction):

    #For now assume that resection is equal among all cell compartments
    u[0] =  u[0]*(1 - resect_fraction)
    u[1:-1] =  u[1:-1]*(1 - resect_fraction)
    u[-1] =  u[-1]*(1 - resect_fraction)
    
    return u
@jit
def detection_death(threshold, N, m, lam ):
    
    #prob = N**m / (threshold**m + N**m)
    
    prob = lam / (1 + np.exp(-m*(N - threshold)))
    
    return prob

# according to this paper rho is linearly related to alpha by factor of 0.005
# 10.1088/0031-9155/55/12/001
@jit
def calc_alpha_from_rho(rhos,alpha_rho_scale=0.005):
    x = rhos * alpha_rho_scale
    return x

# based on the relationship in cell line data
@jit
def calc_alpha_from_rho_new(rhos,alpha_rho_scale=0.2138):
    x = 0.1101 + rhos * alpha_rho_scale
    return x


# assume that alpha/beta = 10 ratio is fixed
@jit
def calc_beta(alpha, ratio =10):
    beta = alpha / ratio
    return beta
#@jit
def LQ_model(alpha,beta,d):
    return np.exp(-alpha*d - beta*d*d)

#@jit
def LQ_model_fixed_ratio(dose, alpha):
    beta = alpha / 10  # Calculate beta based on alpha
    return np.exp(-alpha * dose - beta * dose**2)

@jit
def TLQ_model(dose,F,P,T, alpha, gamma = 0.1376,mu = 0.5):
    beta = alpha / 10  # Calculate beta based on alpha

    return F*np.exp(gamma*(-alpha * dose - beta * dose**2)) + P*np.exp(-alpha * dose - beta * dose**2) + T*np.exp(mu*(-alpha * dose - beta * dose**2))

@jit
def triangle_0(x):
    return np.logical_and(x>=-1,x<0)*(1+x) + np.logical_and(x>=0,x<1)*(1-x)

@jit
def triangle_r(BMP4,n_sen,n_comp=10):

    x = np.arange(0,n_comp)
    r = triangle_0(x-n_sen*BMP4)
    # if going beyond the end, final entry must be 1 
    if n_sen*BMP4 > n_comp-1:
        r[-1] = 1
        
    return r


@jit 
def simulate_simple(t_final,dt,u0,n,Ps,k,ms,mv,delta_s,delta_v,r):
    t = np.arange(0, t_final+dt, dt)
    u = np.zeros((len(t), n+1))
    u[0] = u0
    for i in range(len(t)-1):
        u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:], n, Ps, k, ms, mv, delta_s, delta_v,r)
    return t,u

@jit
def simulate_till(t_final,dt,u0,n,Ps,k,ms,mv,delta_s,delta_v,r, threshold = 0.1):
    t = np.arange(0, t_final+dt, dt)
    u = np.zeros((len(t), n+1))
    u[0] = u0
    for i in range(len(t)-1):
        u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:], n, Ps, k, ms, mv, delta_s, delta_v,r)
        if np.sum(u[i+1,:]) >= threshold:
            return t[:i+2], u[:i+2,:]
    
    #print("Threshold not reached within t_final")
    return t,u


@jit
def simulate_model_n(t_final, u0, psi,n_sen, ms, mv, delta_v, alpha, beta,  rad_on, BMP4_on, resect_on, dt=0.01, Ps_max=0.56, Ps_min=0, n=10, k=1, delta_s=0.001, delta_b=0.5, C=0.5, t_rad = -25, n_RT_repeat=5, n_RT_cycles=6, t_RT_interval=1, t_RT_wait=2, d=2, eta=0.1376, mu=0.5, detect_threshold=0.2, death_threshold=0.7, detection_sensitivity=100, death_sensitivity=20, lam=1, resection_to_RT_delay=30, resect_fraction=0.917, rng_seed=1, m_init=10.1762, m_delivery = "Const"):
    '''
    Simulate the full model.

    Parameters
    ----------
    t_final : float
        The final time of the simulation.
    u0 : np.array
        The initial conditions of the model. A vector of length n+1, containing the initial values of each compartment of the model.
    psi : float
        The sensitivity to BMP4.
    ms : float
        The proliferation rate of GSCs, in units 1/year.
    mv : np.array
        The vector of proliferation rates for non-GSC v1,v2,...,vn. With units 1/year.
    alpha : float
        The radiosensitivity parameter in LQ model.
    beta : float
        The radiosensitivity parameter in LQ model.
    rad_on : [0,1]
        Whether to simulate radiation or not (0 = no radiation, 1 = radiation).
    BMP4_on : [0,1]
        Whether to simulate BMP4 or not (0 = no BMP4, 1 = BMP4).
    resect_on : [0,1]
        Whether to simulate resection or not (0 = no resection, 1 = resection).

    dt : float
        The time step to use in the simulation. The default is 0.01.
    Ps_max : float
        The maximum probability of self renewal. The default is 0.56.
    Ps_min : float
        The minimum probability of self renewal. The default is 0.
    n : int
        The number of generations / compartments in the model. The default is 10 (i.e., 10 generations of non-GSCs so 11 in total includeing the GSCs).
    k : float
        The carrying capacity. The default is 1.
    delta_s : float
        The death rate of GSCs, in units 1/year. The default is 0.001.
    delta_v : np.array
        The vector of death rates for non-GSC v1,v2,...,vn. With units 1/year. The default is np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1]).
    delta_m : float
        The death rate of AMSCs, in units 1/year. The default is 0.25.
    delta_b : float
        The decay rate of BMP4, in units 1/year. The default is 0.5.
    C : float
        The release rate of BMP4 from AMSCs. The default is 0.5.
    u_s : float
        The uptake rate of BMP4 by GSCs. The default is 0.5.
    t_rad : float
        The time to start radiation. The default is -25 days.
    n_RT_repeat : int 
        The number of repeats of the RT schedule, in a single cycle of RT. The default is 5, i.e., 5 days on 2 days off (weekends)
    n_RT_cycles : int
        The number of cycles of RT to run. The default is 6. i.e., 6 weeks of RT
    t_RT_interval : float
        The time interval between RT doses (in days). The default is 1. i.e., during RT cycle RT is given every day.
    t_RT_wait : float
        The time to wait between cycles of RT. The default is 2 days, i.e., the weekend off.
    eta : float
        The radio-protection for GSCs. The default is 0.1376. 
    mu : float
        The radio-protection for non proliferating (terminally differentiated) cells. The default is 0.5.

    Returns
    -------
    u : np.array
        The tumor density at each time step. A 2D array of size (nt,n+1) where n is the number of generations and nt is the number of time steps.
    N : np.array
        The total tumor volume at each time step. A 1D array of size nt.
    t : np.array
        The time grid of the simulation. A 1D array of size nt.
    m : np.array
        The number of AMSCs at each time step. A 1D array of size nt.
    B : np.array
        The concentration of BMP4 at each time step. A 1D array of size nt.
    detect_size : float
        The size of the tumor when it was detected. A scalar value.
    detect_t : float
        The time when the tumor was detected. A scalar value.
    death_size : float
        The size of the tumor when it died. A scalar value.
    death_t : float
        The time when the tumor died. A scalar value.

    '''

    # set random seed
    np.random.seed(rng_seed)
    #print(f'final t = {t_final}')
    #print(f'dt = {dt}')
    # set up time grid
    t = np.arange(0, t_final+dt/2, dt) # arange(a,b,c) uses the open interval [a,b) with steps c
    nt = len(t)

    # set up arrays to store model solution
    u = np.zeros((nt,n+1)) # all model compartments
    VS = np.zeros(nt) # sum of non-GSC compartments
    N = np.zeros(nt) # total tumor density
    m = np.zeros(nt) # AMSC cells
    B = np.zeros(nt) # BMP4 concentration

    # keep track of how many cycles of RT have occurred.
    rad_counter = 0

    # If the tumor is never detected / dies then return 0
    detect_size = 0
    detect_t = 0
    death_size = 0
    death_t = 0

    # define IC
    u[0,:] = u0
    VS[0] = np.sum(u[0,1:n+1]) # sum the 1st to nth entries
    N[0] = u[0,0] + VS[0] # total tumor volume.

    # define flag
    shifted_flag = 0
    RT_started = 0

    for i in range(len(t)-1):


        # test for tumor detection
        rand = np.random.random_sample() # generate a random uniform number between [0,1]
        if (rand < detection_death(detect_threshold, N[i],detection_sensitivity,lam) *dt and detect_threshold >0): 
            detect_threshold = -1
            detect_size = N[i]
            detect_t = t[i]
            # Simulate resection
            if resect_on == 1:
                u[i,:] = resection(u[i,:],resect_fraction)
            t_rad = t[i] + resection_to_RT_delay

            # apply BMP4 at time of resection if single / in OR delivery
            if  BMP4_on ==1 and m_delivery == "Single" :
                m[i] = m_init
                
        # test for death        
        rand = np.random.random_sample()
        if rand < detection_death(death_threshold, N[i],death_sensitivity, lam) * dt and death_threshold > 0 and detect_t > 0:
            # Once death has occurred, it cannot happen again
            death_threshold = -1
            # Save the time and size that death occurs.
            death_size = N[i]
            death_t = t[i]
            break

        # Calculate probability of self renewal (Ps) based on current concentration of BMP4
        Ps = Ps_min + (Ps_max - Ps_min)*(1 / (1 + psi*B[i]))
        
        r = triangle_r(B[i],n_sen)

        # update model compartments.
        u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:],n,Ps,k,ms,mv,delta_s,delta_v,r)
        
        # update BMP4 model
        B[i+1] = B[i] + dt * (C * m[i] - delta_b*B[i])

        ### different delivery methods for BMP4 ###
        
        # After detection and resection, BMP4 delivered at a constant rate
        if detect_threshold ==-1 and rad_counter < n_RT_repeat*n_RT_cycles and BMP4_on ==1 and m_delivery == "Const":
            m[i+1] = m_init

        # BMP4 is Pulsatile in combination with RT
        if abs(t[i]-t_rad) < dt/2 and rad_counter % n_RT_repeat == 0 and rad_counter < n_RT_repeat*n_RT_cycles and m_delivery == "Pulse":
            m[i+1] = m[i] + m_init

        # apply radiation
        if abs(t[i]-t_rad) < dt/2 and rad_counter < n_RT_repeat*n_RT_cycles and rad_on == 1 :
            RT_started = 1
            u[i+1,:] = radiation(u[i+1,:],alpha,beta,eta,mu,d,n)
            rad_counter = rad_counter + 1
            if rad_counter % n_RT_repeat == 0 :
                t_rad = t_rad + t_RT_wait # wait for next cycle
            else:
                t_rad = t_rad + t_RT_interval
                
        # Keep track of the total tumor size.
        N[i+1] = np.sum(u[i+1,:])
            
    return u,N,t,m,B,detect_size,detect_t,death_size,death_t


@jit
def sim_model_new(t_final, u0, psi,n_sen, ms, mv, delta_v, alpha, beta,  rad_on, BMP4_on, resect_on, dt=0.01, Ps_max=0.56, Ps_min=0, n=10, k=1, delta_s=0.001, delta_b=4.15, C_=0.273*1000000, Q_ = 0.0509157, V = 1300, t_rad = -25, n_RT_repeat=5, n_RT_cycles=6, t_RT_interval=1, t_RT_wait=2, d=2, eta=0.1376, mu=0.5, detect_threshold=0.2, death_threshold=0.7, detection_sensitivity=100, death_sensitivity=20, lam=1, resection_to_RT_delay=30, resect_fraction=0.917, rng_seed=1, delivery = "Const", B_tot = 1*1000000):

    # set random seed
    np.random.seed(rng_seed)
    #print(f'final t = {t_final}')
    #print(f'dt = {dt}')
    # set up time grid
    t = np.arange(0, t_final+dt/2, dt) # arange(a,b,c) uses the open interval [a,b) with steps c
    nt = len(t)

    # set up arrays to store model solution
    u = np.zeros((nt,n+1)) # all model compartments
    VS = np.zeros(nt) # sum of non-GSC compartments
    N = np.zeros(nt) # total tumor density
    Q = np.zeros(nt) # flow rate of BMP4 into tumour from CED or AMSCs
    C = np.zeros(nt) # concentration of BMP4 in the source (CED catheter or AMSCs)
    B = np.zeros(nt) # BMP4 concentration

    # keep track of how many cycles of RT have occurred.
    rad_counter = 0

    # If the tumor is never detected / dies then return 0
    detect_size = 0
    detect_t = 0
    death_size = 0
    death_t = 0

    # define IC
    u[0,:] = u0
    VS[0] = np.sum(u[0,1:n+1]) # sum the 1st to nth entries
    N[0] = u[0,0] + VS[0] # total tumor volume.

    # define flag
    shifted_flag = 0
    RT_started = 0

    for i in range(len(t)-1):


        # test for tumor detection
        rand = np.random.random_sample() # generate a random uniform number between [0,1]
        if (rand < detection_death(detect_threshold, N[i],detection_sensitivity,lam) *dt and detect_threshold >0): 
            detect_threshold = -1
            detect_size = N[i]
            detect_t = t[i]
            # Simulate resection
            if resect_on == 1:
                u[i,:] = resection(u[i,:],resect_fraction)
            t_rad = t[i] + resection_to_RT_delay

            # apply BMP4 at time of resection if single / in OR delivery
            if  BMP4_on ==1 and delivery == "Single" :
                B[i] = B_tot / V # this ensures same total dose as continuous 18 mg
                
        # test for death        
        rand = np.random.random_sample()
        if rand < detection_death(death_threshold, N[i],death_sensitivity, lam) * dt and death_threshold > 0 and detect_t > 0:
            # Once death has occurred, it cannot happen again
            death_threshold = -1
            # Save the time and size that death occurs.
            death_size = N[i]
            death_t = t[i]
            break

        # Calculate probability of self renewal (Ps) based on current concentration of BMP4
        Ps = Ps_min + (Ps_max - Ps_min)*(1 / (1 + psi*B[i]))
        
        r = triangle_r(B[i],n_sen)

        # update model compartments.
        u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:],n,Ps,k,ms,mv,delta_s,delta_v,r)
        
        # update BMP4 model
        B[i+1] = B[i] + dt * ( Q[i] * C_ / V - delta_b * B[i] )

        ### different delivery methods for BMP4 ###
        
        # After detection and resection, BMP4 delivered at a constant rate
        if detect_threshold ==-1 and rad_counter < n_RT_repeat*n_RT_cycles and BMP4_on ==1 and delivery == "Const":
            Q[i+1] = Q_ # this is equivalent t turning the flow Q on and off

        # apply radiation
        if abs(t[i]-t_rad) < dt/2 and rad_counter < n_RT_repeat*n_RT_cycles and rad_on == 1 :
            RT_started = 1
            u[i+1,:] = radiation(u[i+1,:],alpha,beta,eta,mu,d,n)
            rad_counter = rad_counter + 1
            if rad_counter % n_RT_repeat == 0 :
                t_rad = t_rad + t_RT_wait # wait for next cycle
            else:
                t_rad = t_rad + t_RT_interval
                
        # Keep track of the total tumor size.
        N[i+1] = np.sum(u[i+1,:])


    return u,N,t,C,B,detect_size,detect_t,death_size,death_t


@jit
def rad_times(detect_t, rad_t = -25, resection_to_RT_delay=30, n_RT_repeat=5, t_RT_wait = 2, n_RT_cycles=6):
    '''
    Return the times at which RT started and ended

    Parameters
    ----------
    detect_t : float
    rad_t : float
        Time of radiation. Defualt to negative i.e., it is defualt assumed that resection occured and RT followed unless a RT time is specified.
    resection_to_RT_delay : float
        Time from resection to RT. Default to 30 days.
    n_RT_repeat : float
        Number of repeats of RT. Default to 5 rounds (weekdays).
    t_RT_wait : float
        Time between repeats of RT. Default to 2 days (weekends).
    n_RT_cycles : float
        Number of RT cycles. Default to 6 (rounds of RT, 6 total weeks).
    
    Returns
    -------
    start_rad : float
        Start time of RT
    end_rad : float
        End time of RT

    '''
    if rad_t == -25:
        start_rad = detect_t + resection_to_RT_delay
    else:
        start_rad = rad_t
    
    end_rad = start_rad + (n_RT_repeat + t_RT_wait) * n_RT_cycles

    return start_rad, end_rad


#@jit
def Therapy_comparison_trial(sample, t_final = 8000,n=10,mod = 10): 
    
    # sample = Latin hyper cube sample of parameter space. Its a 2D array of
    # shape (n_samples, n_params) 

    # number of patients
    n_patients = len(sample[:,0])
    PIDs = np.arange(0,n_patients,1) # Unique patient IDs
   
    # start with a small initial tumor, with a mix of minority GSC
    n0 = 0.001
    s0 = 0.01*n0 # Initial GSC comprise 1% of total tumour
    v_ratio = 1.95 # ratio between successive compartments 
    v0 = ((n0-s0) * (v_ratio -1) / (v_ratio**n-1))*(v_ratio**np.arange(n))
    u0 = np.zeros(n+1)
    u0[0] = s0
    u0[1:] = v0

    #s0 = 0.001 # 0.001K
    #u0 = np.zeros(n+1)
    #u0[0] = s0

    pro_rates_sampled = sample[:,0]
    psi_samples = sample[:,1]
    n_sen_samples = sample[:,2] 
    mv_scale_samples = sample[:,3]
    delta_s_samples = sample[:,4]
    delta_p_scale_samples = sample[:,5]
    alpha_scale_samples = sample[:,6]
    eta_scale_samples = sample[:,7]
    mu_scale_samples = sample[:,8]
    Ps_max_samples = sample[:,9]
    Ps_min_samples = sample[:,10]


    # we want each patient to have a unique random seed so that across all simulations they get the same series of random numbers
    random_seeds = np.arange(0,n_patients,1)

    # save final survival for BMP4 constant delivery arm
    BMP4_const = np.zeros(n_patients)
    status_const = np.ones(n_patients,  dtype=bool)

    # save the final survival for BMP4 single delivery arm
    BMP4_single = np.zeros(n_patients)
    status_single = np.ones(n_patients,  dtype=bool)

    # save the final survival for virtual control arm
    CTRL_survival = np.zeros(n_patients)
    status_CTRL = np.ones(n_patients, dtype=bool)

    # save the detection times for each patient in each arm
    detect_t_save_CTRL = np.zeros(n_patients)
    detect_t_save_const = np.zeros(n_patients)
    detect_t_save_single = np.zeros(n_patients)

    # save the death times for each patient in each arm
    death_t_save_CTRL = np.zeros(n_patients)
    death_t_save_const = np.zeros(n_patients)
    death_t_save_single = np.zeros(n_patients)


    # Save all simulations
    u_CTRL_save = pd.DataFrame(columns=['PID', 'u_CTRL'])
    u_const_save = pd.DataFrame(columns=['PID', 'u_const'])
    u_single_save = pd.DataFrame(columns=['PID', 'u_single'])

    for i in range(len(psi_samples)):
        
            # Extract parameters from Latin-hyper cube sample
            ms = sample[i,0] # 1/days
            psi = sample[i,1]
            n_sen = sample[i,2]
            mv_scale = sample[i,3]
            delta_s = sample[i,4]
            delta_v = delta_s * sample[i,5] * np.ones(n)
            alpha_scale = sample[i,6]
            eta_scale = sample[i,7]
            mu_scale = sample[i,8]
            Ps_max = sample[i,9]
            Ps_min = sample[i,10]

            
            mv = ms*mv_scale*np.ones(n)
            mv[-1] = 0 
    
            # calc alpha as proportional to rho * alpha_scale
            alpha = calc_alpha_from_rho_new(ms,alpha_rho_scale=alpha_scale) # base on the cell line data alpha = 0.1101 + alpha_scale*ms
            beta = calc_beta(alpha) # beta = alpha/10 

            # First set of trials with BMP4 in combination with standard of care
            rad_on = 1        
            BMP4_on = 1
            resect_on = 1
    
            ##### CONSTANT BMP4 DELIVERY arm #####
            # simulate the model, default is with constant delivery
            u_const,N,t,m,B,detect_size_const,detect_t_const,death_size_const,death_t_const = sim_model_new(t_final, u0, psi,n_sen, ms, mv, delta_v, alpha, beta, rad_on, BMP4_on, resect_on, rng_seed=random_seeds[i], eta = eta_scale, mu = mu_scale, Ps_max=Ps_max,Ps_min = Ps_min, delta_s=delta_s, delivery="Const")
            #print(f'Death size {death_size_const}')
            # save the survival time of the BMP4 arm
            if i % mod == 0:
                u_const_save = pd.concat([u_const_save, pd.DataFrame({'PID': PIDs[i], 'u_const': [u_const]})], ignore_index=True)

            if detect_t_const == 0: # if the tumor is never detected we will get rid of it (hence fill with -1 so easy to identify later and remove)
                BMP4_const[i] = -1
            elif death_t_const == 0: # if patient doesn't die before the end of the simulation they have survival time t_end - t_detect and are censored
                BMP4_const[i] = t_final - detect_t_const
                status_const[i] = False
            else:
                BMP4_const[i] = death_t_const-detect_t_const
            
            detect_t_save_const[i] = detect_t_const
            death_t_save_const[i] = death_t_const

            ##### SINGLE BMP4 DELIVERY #####
            # simulate the model, default is with constant delivery
            u_single,N,t,m,B,detect_size_single,detect_t_single,death_size_single,death_t_single = sim_model_new(t_final, u0, psi,n_sen, ms, mv, delta_v, alpha, beta,  rad_on, BMP4_on, resect_on, rng_seed=random_seeds[i] , delivery='Single', eta = eta_scale, mu = mu_scale, Ps_max=Ps_max,Ps_min = Ps_min, delta_s = delta_s)
            if i % mod == 0:
                u_single_save = pd.concat([u_single_save, pd.DataFrame({'PID': PIDs[i], 'u_single': [u_single]})], ignore_index=True)

            # save the survival time of the BMP4 arm
            if detect_t_single == 0:
                BMP4_single[i] = -1
            elif death_t_single == 0:
                BMP4_single[i] = t_final - detect_t_single
                status_single[i] = False
            else:
                BMP4_single[i] = death_t_single-detect_t_single
            
            detect_t_save_single[i] = detect_t_single
            death_t_save_single[i] = death_t_single

            ##### VIRTUAL CONTROL ARM
            # for each of the patients run the same thing again but with no BMP4 to act as a virtual control
            rad_on = 1
            BMP4_on = 0 # turn BMP4 off
            resect_on = 1

            # run the control with the same random seed as the BMP4 so we get the same series of random nums
            u,N,t,m,B,detect_size,detect_t,death_size,death_t = sim_model_new(t_final, u0, psi,n_sen, ms, mv, delta_v, alpha, beta,  rad_on, BMP4_on, resect_on, rng_seed=random_seeds[i], eta = eta_scale, mu = mu_scale, Ps_max=Ps_max,Ps_min = Ps_min, delta_s = delta_s)
            # Only save so many of the full time courses
            if i % mod == 0:
                u_CTRL_save = pd.concat([u_CTRL_save, pd.DataFrame({'PID': PIDs[i], 'u_CTRL': [u]})], ignore_index=True)

            # save the survival time of the control arm
            if detect_t == 0:
                CTRL_survival[i] = -1
            elif death_t == 0:
                CTRL_survival[i] = t_final - detect_t
                status_CTRL[i] = False
            else:
                CTRL_survival[i] = death_t-detect_t

            detect_t_save_CTRL[i] = detect_t
            death_t_save_CTRL[i] = death_t

            


    save_data = {'PID': PIDs,'pro_rate': pro_rates_sampled,'psi': psi_samples,'n_sen': n_sen_samples,
                 'mv_scale': mv_scale_samples,'delta_s': delta_s_samples,'delta_p_scale': delta_p_scale_samples,
                 'alpha_scale': alpha_scale_samples,'eta': eta_scale_samples,'mu':mu_scale_samples,'Ps_max': Ps_max_samples,'Ps_min': Ps_min_samples, 
                'BMP4_const': BMP4_const, 'BMP4_single': BMP4_single, 'Virtual_control': CTRL_survival,
                'Status_const': status_const, 'Status_single': status_single, 'Status_CTRL': status_CTRL, 
                'detect_t_CTRL': detect_t_save_CTRL, 'death_t_CTRL': death_t_save_CTRL, 'detect_t_const': detect_t_save_const, 
                'death_t_const': death_t_save_const, 'detect_t_single': detect_t_save_single, 'death_t_single': death_t_save_single}
    
    survival_BMP4_df = pd.DataFrame(save_data)

    # ALready sorted the samples outside the df
    # sort dataframe by proliferation rate
    #survival_BMP4_df = survival_BMP4_df.sort_values(by='pro_rate')

    # reset index in  case any tumors not detected
    #survival_BMP4_df = survival_BMP4_df.reset_index(drop=True)

    return survival_BMP4_df, u_const_save, u_CTRL_save, u_single_save

# Jit doesn't work with scipystats so don't add it here
def trunc_norm(mu,sigma,n_samples):

    # Define the bounds for the truncated distribution
    amin, amax = 0, np.inf
    amin, amax = (amin - mu) / sigma, (amax - mu) / sigma

    # Generate samples
    samples = stats.truncnorm.rvs(amin, amax, loc=mu, scale=sigma, size=n_samples)

    
    return samples


#@jit
def basic_growth_simulation(t_final,dt,u0,s0,Ps_max,Ps_min,n,k,ms,mv,delta_s,delta_v,max_size, BMP4 = 0, psi =0): 
    t = np.arange(0, t_final+dt/2, dt) # arange(a,b,c) uses the open interval [a,b) with steps c
    nt = len(t)
    u = np.zeros((nt,n+1))
    VS = np.zeros(nt)
    N = np.zeros(nt)

    tp1 = 0 # time to cross 0.1*k
    tp2 = 0 # time to cross 0.2*k

    # define IC
    u[0,:] = u0
    u[0,0] = s0
    VS[0] = np.sum(u[0,1:n+1]) # sum the 1th to nth entries
    N[0] = u[0,0] + VS[0]

    i = 0
    while N[i] < max_size and i<(nt-2):
    
            Ps = Ps_min + (Ps_max - Ps_min)*(1 /(1+ BMP4*psi))
                
            u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:],n,Ps,k,ms,mv,delta_s,delta_v)


            VS[i+1] = np.sum(u[i+1,1:n+1])
            N[i+1] = u[i+1,0] + VS[i+1]
        
            # estimate doubling time
            if (N[i]<0.1*k and N[i+1]>=0.1*k):
                tp1 = t[i+1]
            if (N[i]<0.2*k and N[i+1]>=0.2*k):
                tp2 = t[i+1]

            if N[i] > max_size*k:
                break

            i = i + 1

    # cut all them off at final index
    u = u[0:i,:]
    N = N[0:i]
    VS = VS[0:i]
    t = t[0:i]


    return u,N,VS,t,tp1,tp2

# Helper function to sample middle 50% of an array
def sample_middle_50_percent(arr):
    # Convert the input to a NumPy array if it's not already
    arr = np.array(arr)
    
    # Calculate the start and end indices for the middle 50%
    start_index = len(arr) // 4
    end_index = len(arr) * 3 // 4
    
    # Slice the array to get the middle 50%
    middle_50 = arr[start_index:end_index]
    
    return middle_50



def phase2_trial_fun(n_trials,n_patients,rho_case,psi_case, BMP4_dose_case ,n = 10, t_final = 8000, rng_seed = 0,fixed = False): 

    '''
    # n_trials (int), number of virtual clinical trials to calculate average from.
    # n_patients (int), number of patients per phase 2 virtual trial
    # distinct_arms (bool), whether the BMP4 and noBMP4 arms should be distinct sub-populations
    # rho_case (int), how to select patients from the rho distribution
    '''
   
   # Set random seed for parameter sampling
    np.random.seed(rng_seed)

   # Set IC for each virtual tumor
    u0 = np.zeros(n+1)
    n0 = 0.001
    s0 = 0.01*n0 # fraction of initial tumour
    v_ratio = 1.95 # ratio between successive compartments
    v0 = ((n0-s0)*(v_ratio-1)/(v_ratio**n-1))*(v_ratio**np.arange(n))
    u0[0] = s0
    u0[1:] = v0

    # Psi parameter range
    if psi_case == 0:
        psi_lb = 0
        psi_ub = 0.1
    elif psi_case == 1:
        psi_lb = 0.1
        psi_ub = 0.2
    # No stratification
    elif psi_case == 2:
        psi_lb = 0
        psi_ub = 0.2

    # rho parameter range, units 1/days, range taken from cell line fits
    if rho_case == 0:
        rho_lb = 0.1
        rho_ub = 0.275
    elif rho_case == 1:
        rho_lb = 0.275
        rho_ub = 0.45
    elif rho_case == 2:
        rho_lb = 0.1
        rho_ub = 0.45

    # total dose 0.5mg
    if BMP4_dose_case == 0:
        Q = 0.0244 # release rate of BMP4
        C = 0.273*1000000 # concetration of BMP4 from Bos et al. phase I trial
    # total dose 5mg (72 days between start resection and end of RT)
    elif BMP4_dose_case == 1:
        Q =  0.244 # 5mg total dose
        C = 0.273*1000000

    if fixed == True:
        # all other parameters are fixed at midpoint of the range used previously in the global sensitivity analysis?
        # Or alternative since they don't matter we could have the as still random
        n_sen = 0.05
        mv_scale = 2
        delta_s = 0.001
        delta_v_scale = 10
        delta_v = delta_s*delta_v_scale*np.ones(n)
        delta_t_scale = 5
        alpha_scale = 0.005
        eta_scale = 0.1635
        mu_scale = 0.5
        Ps_max = 0.55
        Ps_min = 0.



    # store all the data we need for each trial
    # for each patient of each trial we want to store
    # 0) trial number, 1) rho_range, 2) psi_range, 3) m_init/BMP4, 4) psi, 5) rho
    # 6) CTRL survival 7) Const BMP4 survival 8) Single BMP4 surv , 9) Status_CTRL (censored or not) 
    # 10) Status_Const_BMP4 11) Status_Single_BMP4
    save_data = pd.DataFrame(columns=['trial','rho_case','psi_case','rho_range','psi_range','m_init','psi','rho','n_sen','mv_scale','delta_s','delta_v_scale',
    'delta_t_scale','u_B','C','alpha_scale','eta_scale','mu_scale','Ps_max','Ps_min','Survival_CTRL','Survival_Const_BMP4','Survival_Single_BMP4', 'Status_CTRL', 'Status_Const','Status_Single'])

    #np.random.seed(rng_seed)

    for trial in range(n_trials):

        # Keep track of how many patients we have simulated (since we will discard patients that never detected)
        i = 0

        while i < n_patients:
            
            
                if fixed == False:
                    # Draw them from their uniform distribution with same range as used in the GSA
                    n_sen= np.random.uniform(0,0.1)
                    mv_scale = np.random.uniform(1,5)
                    delta_s = np.random.uniform(0.0001,0.002)
                    delta_v_scale = np.random.uniform(1,10)
                    alpha_scale = np.random.uniform(0.001,0.25)
                    eta_scale = np.random.uniform(0.005,1)
                    mu_scale = np.random.uniform(0.2,1)
                    Ps_max = np.random.uniform(0.5,0.6)
                    Ps_min = np.random.uniform(0.1,0.3)
                
                ms =  np.random.uniform(rho_lb,rho_ub)
                mv = ms*mv_scale*np.ones(n)
                mv[n-1] = 0 

                delta_v = delta_s*delta_v_scale*np.ones(n)

                psi = np.random.uniform(psi_lb,psi_ub)

                # calc alpha as proportional to rho 
                alpha = calc_alpha_from_rho(ms,alpha_rho_scale=alpha_scale)
                beta = calc_beta(alpha)

                # save the trial number to which virtual patient belongs, we only need to do this once as it is the same across all subsequent trials
                save_data.loc[(trial*n_patients)+i, 'trial'] = trial
                save_data.loc[(trial*n_patients)+i, 'rho_case'] = rho_case
                save_data.loc[(trial*n_patients)+i, 'psi_case'] = psi_case
                # Save the rho range / stratification
                save_data.at[(trial*n_patients)+i, 'rho_range'] = [rho_lb, rho_ub]
                # Save the psi range / stratification
                save_data.at[(trial*n_patients)+i, 'psi_range'] = [psi_lb, psi_ub]
                # save the BMP4 dose for virtual patient
                save_data.loc[(trial*n_patients)+i,'BMP4_dose'] = BMP4_dose_case
                # save psi
                save_data.loc[(trial*n_patients)+i,'psi'] = psi
                # save proliferation rate for virtual patient
                save_data.loc[(trial*n_patients)+i, 'rho'] = ms
                # Save all the parameters for virtual patients
                save_data.loc[(trial*n_patients)+i, 'n_sen'] = n_sen
                save_data.loc[(trial*n_patients)+i, 'mv_scale'] = mv_scale
                save_data.loc[(trial*n_patients)+i, 'delta_s'] = delta_s
                save_data.loc[(trial*n_patients)+i, 'delta_v_scale'] = delta_v_scale
                save_data.loc[(trial*n_patients)+i, 'alpha_scale'] = alpha_scale
                save_data.loc[(trial*n_patients)+i, 'eta_scale'] = eta_scale
                save_data.loc[(trial*n_patients)+i, 'mu_scale'] = mu_scale
                save_data.loc[(trial*n_patients)+i, 'Ps_max'] = Ps_max
                save_data.loc[(trial*n_patients)+i, 'Ps_min'] = Ps_min

                ######
                # CTRL ARM
                #####
                rad_on = 1
                BMP4_on = 0
                resect_on = 1
                u,N,t,m,B,detect_size,detect_t,death_size,death_t = sim_model_new(t_final, u0, psi,n_sen, ms, mv, delta_v, alpha, beta,  rad_on, BMP4_on, resect_on,  rng_seed=(trial*n_patients)+i, eta = eta_scale, mu = mu_scale, Ps_max=Ps_max,Ps_min = Ps_min, delta_s = delta_s, Q_ = Q, C_ = C)

                # Save survival time of all patients in trial and keep track if censored
                if detect_t == 0: # if the tumor is never detected we will re spample
                     continue
                elif death_t == 0: # if patient doesn't die before the end of the simualtion they have surival time t_end - t_detect and are censored
                    save_data.loc[(trial*n_patients)+i, 'Survival_CTRL'] = t_final - detect_t
                    save_data.loc[(trial*n_patients)+i, 'Status_CTRL'] = False
                else:
                    # save the survival time of the BMP4 arm
                    save_data.loc[(trial*n_patients)+i, 'Survival_CTRL'] = death_t-detect_t
                    save_data.loc[(trial*n_patients)+i, 'Status_CTRL'] = True
     
                #########
                # CONSTANT BMP4 DELIVERY arm
                ##########
                rad_on = 1        
                BMP4_on = 1
                resect_on = 1

        
                # simulate the model
                u,N,t,m,B,detect_size,detect_t,death_size,death_t = sim_model_new(t_final, u0, psi,n_sen, ms, mv, delta_v, alpha, beta,  rad_on, BMP4_on, resect_on,  rng_seed=(trial*n_patients)+i, eta = eta_scale, mu = mu_scale, Ps_max=Ps_max,Ps_min = Ps_min, delta_s = delta_s, Q_ = Q, C_ = C)

                # Save survival time of all patients in trial and keep track if censored
                if detect_t == 0: # if the tumor is never detected we will get rid of it (hence fill with -1 so easy to identify later and remove)
                     continue
                elif death_t == 0: # if patient doesn't die before the end of the simualtion they have surival time t_end - t_detect and are censored
                    save_data.loc[(trial*n_patients)+i, 'Survival_Const_BMP4'] = t_final - detect_t
                    save_data.loc[(trial*n_patients)+i, 'Status_Const'] = False
                else:
                    # save the survival time of the BMP4 arm
                    save_data.loc[(trial*n_patients)+i, 'Survival_Const_BMP4'] = death_t-detect_t
                    save_data.loc[(trial*n_patients)+i, 'Status_Const'] = True
                

                i = i + 1


    return save_data

def pairwise_params(t_final=8000, n=10, den=21):
    u0 = IC()
    param_dict = {
        'pro_rate': np.linspace(10, 90, den),
        'psi': np.linspace(0, 0.05, den),
        'n_sen': np.linspace(0, 50, den),
        'mv_scale': np.linspace(1, 5, den),
        'delta_s': np.linspace(0.0001, 0.002, den),
        'delta_p_scale': np.linspace(1, 20, den),
        'delta_t_scale': np.linspace(1, 10, den),
        'u_B': np.linspace(0.1, 1, den),
        'C': np.linspace(0.1, 1, den),
        'alpha_scale': np.linspace(0.001, 0.01, den),
        'eta': np.linspace(0.001, 1, den),
        'mu': np.linspace(0.2, 1, den),
        'Ps_max': np.linspace(0.5, 0.6, den),
        'Ps_min': np.linspace(0.1, 0.3, den),
    }

    param_names = list(param_dict.keys())
    n_params = len(param_names)

    fixed_values = {k: v[den // 2] for k, v in param_dict.items()}
    fig, ax = plt.subplots(n_params, n_params, figsize=(30, 30), constrained_layout=True)

    for i in range(n_params):
        for j in range(n_params):
            if i <= j:
                ax[i, j].axis('off')  # upper triangle and diagonal blank
                continue

            p1 = param_names[i]
            p2 = param_names[j]
            print(f"Simulating {p1} vs {p2} → subplot [{i},{j}]")

            days_gained_surface = np.zeros((den, den))

            for ii in range(den):
                val1 = param_dict[p1][ii]
                for jj in range(den):
                    val2 = param_dict[p2][jj]

                    param_set = fixed_values.copy()
                    param_set[p1] = val1
                    param_set[p2] = val2

                    psi = param_set['psi']
                    n_sen = param_set['n_sen']
                    ms = param_set['pro_rate'] * 2 / 365
                    mv = ms * param_set['mv_scale'] * np.ones(n)
                    mv[-1] = 0
                    delta_s = param_set['delta_s']
                    delta_v = delta_s * param_set['delta_p_scale'] * np.ones(n)
                    delta_v[-1] = delta_v[0] * param_set['delta_t_scale']
                    alpha = calc_alpha_from_rho(param_set['pro_rate'], alpha_rho_scale=param_set['alpha_scale'])
                    beta = calc_beta(alpha)

                    # CTRL
                    u, N, t, m, B, detect_size, detect_t, death_size, death_t_CTRL = simulate_model_n(
                        t_final, u0, psi, n_sen, ms, mv, delta_v, alpha, beta, 1, 0, 1)

                    # BMP4
                    u, N, t, m, B, detect_size, detect_t, death_size, death_t_BMP4 = simulate_model_n(
                        t_final, u0, psi, n_sen, ms, mv, delta_v, alpha, beta, 1, 1, 1)

                    days_gained_surface[ii, jj] = death_t_BMP4 / death_t_CTRL

            im = ax[i, j].contourf(param_dict[p2], param_dict[p1], days_gained_surface, levels=50, cmap='viridis')
            ax[i, j].set_xlabel(p2)
            ax[i, j].set_ylabel(p1)

    plt.suptitle("Pairwise Parameter Interactions — Days Gained", fontsize=20)
    #plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.02, pad=0.05)
    plt.show()
    return None

# Function to set up the IC
def IC(n0 = 0.001, s0 = 0.01, v_ratio = 1.95, n=10):
    # start with a small initial tumor 
    n0 = 0.001
    s0 = 0.01*n0 # Intial GSC comprise 1% of totoal tumor
    v_ratio = 1.95 # ratio between successive compartments 
    v0 = ((n0-s0) * (v_ratio -1) / (v_ratio**n-1))*(v_ratio**np.arange(n))
    u0 = np.zeros(n+1)
    u0[0] = s0
    u0[1:] = v0

    return u0

# convert a doubling time (in hrs) to a proliferation rate (1/days)
def double_to_proliferation_rate(double):
    # convert doubling time from hours to days
    double_days = double / 24
    # calculate proliferation rate as ln(2) / doubling time
    pro_rate = np.log(2) / double_days
    return pro_rate


def plot_all_compartments(t, u, ax, colormap = 'RdYlGn'):

    # Colors from red to green
    cmap = plt.get_cmap(colormap)
    # Interpolated colors at these positions
    col = [cmap(p) for p in np.linspace(0, 1, len(u[1,:]))]

    for i in range(len(u[1,:])):
        ax.plot(t,u[:,i], color  = col[i])
    
    ax.plot(t,np.sum(u, axis=1), color = 'k')

#### From the fitting code we have function to simulate the cell line experiments


# Function to simulate the radiotherapy assay, i.e. 2 days growth followed by RT
def sim_RT_assay(cline_df, cline, psi, n_sen, RT_dose, n = 10, dt = 0.01, t_rad = 2, mv_ms_scale = 2, eta = 0.1635, mu = 0.5, Ps_max = 1, Ps_min = 0, BMP4_dose = 100, delta_s = 0.001, s0 = 0.01, k = 1):

    # time interval is 2 days BMP4 exposure followed by RT, we assume that colonies are counted immediately after RT
    t = np.arange(0, t_rad+2*dt,dt)

    # value from the cline data table
    ms = cline_df[cline_df['cline'] == cline]['pro rate (1/days)'].iloc[0] # units 1/days

    # Radiosensitivity parameters fitted from CTRL conditions
    alpha = cline_df[cline_df['cline'] == cline]['CTRL alpha'].iloc[0]
    alpha = alpha / eta # in CTRL conditions we assumed all GSCs here alpha is relative to PC sensitivity which is assumed 1
    beta = alpha / 10 # fixed following Rockne et al.

    # Define proliferation rates for the different cell pops
    mv = mv_ms_scale*ms*np.ones(n) 
    mv[n-1] = 0 

    delta_v = 0.01 * np.ones(n)

    # set up array to store model solution
    u = np.zeros((len(t),n+1))
    N = np.zeros(len(t))

    # define IC
    u[0,0] = s0
    N[0] = np.sum(u[0,:])

    if isinstance(psi, np.ndarray):
      psi = psi.item()

    for i in range(len(t)-1):

        Ps = Ps_min + (Ps_max - Ps_min)*(1 / (1 + psi*BMP4_dose))
        r = triangle_r(BMP4_dose,n_sen)
        u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:],n,Ps,k,ms,mv,delta_s,delta_v,r)
        # apply radiation
        if t[i] == t_rad :
            u[i+1,:] = radiation(u[i+1,:],alpha,beta,eta,mu,d=RT_dose,n = n)
           

        N[i+1] = np.sum(u[i+1,:])

    return u, N

# Function to simulate the RT assay with different doses
def sim_RT_assay_doses(cline_df, cline, psi, n_sen, RT_doses = [0,2,4,6], n = 10):

    t_rad = 2
    dt = 0.01
    t = np.arange(0, t_rad+2*dt,dt)

    u_all = np.zeros([len(RT_doses),len(t),n+1])
    N_all = np.zeros([len(RT_doses),len(t)])
    final_sizes = np.zeros(len(RT_doses))

    for i in range(len(RT_doses)):

        u, N = sim_RT_assay(cline_df, cline, psi, n_sen, RT_dose = RT_doses[i])

        u_all[i,:,:] = u
        N_all[i,:] = N
        final_sizes[i] = N[-1]

    # Normalise final size as we do with the RT data
    final_sizes = final_sizes/final_sizes[0]

    return final_sizes, u_all, N_all


# Simulate CTRL of pro_assay, Ps = 1 and initially 100% GSC
def sim_CTRL(cline_df, cline, t_final_pro = 7, dt = 0.01, s0 = 0.01, k = 1, delta_s = 0.001):

    t = np.arange(0, t_final_pro+dt, dt)
    Ps = 1
    # value from the cline data table
    ms= cline_df[cline_df['cline'] == cline]['pro rate (1/days)'].iloc[0] # units 1/days

    s = np.zeros(len(t))
    s[0] = s0

    for i in range(len(t)-1):
        s[i+1] = s[i] + dt * ((2*Ps - 1) * ms * s[i] * (1 - (s[i])/k) - delta_s*s[i])

    return s


def sim_pro_assay_n(cline_df, cline, psi, n_sen, t_final_pro = 7, dt = 0.01, s0 = 0.01, k = 1, delta_s = 0.001, n = 10, mv_ms_scale = 2, BMP4_dose = 100):

    # value from the cline data table
    ms = cline_df[cline_df['cline'] == cline]['pro rate (1/days)'].iloc[0] # units 1/days


    t =  np.arange(0, t_final_pro+dt/2, dt)

    Ps_max = 1
    Ps_min = 0
                                
    # With BMP4
    mv = ms*mv_ms_scale*np.ones(n)
    mv[-1] = 0
    delta_v = 0.01 * np.ones(n)


    u0 = np.zeros(n+1)
    u0[0] = s0
    u_BMP4 = np.zeros((len(t),n+1))
    N_BMP4 = np.zeros(len(t))
    u_BMP4[0,:] = u0
    N_BMP4[0] = np.sum(u0)


    for i in range(len(t)-1):

        Ps = Ps_min + (Ps_max - Ps_min)*(1 / (1 + psi*BMP4_dose))
        r = triangle_r(BMP4_dose,n_sen)
        u_BMP4[i+1,:] = u_BMP4[i,:] + dt * gsc_model_dudt(u_BMP4[i,:],n,Ps,k,ms,mv,delta_s,delta_v,r)
                            
      
        N_BMP4[i+1] = np.sum(u_BMP4[i+1,:])


    return N_BMP4
