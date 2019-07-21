import os, sys

dirScript = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirScript)
sys.path.append(os.path.join(dirScript, './../input_data_read'))

from sklearn.model_selection import train_test_split, KFold
import numpy as np
import math as m
import input_data_read

def ReLu(x):
    if x < 0: return 0
    else: return x

def leaky_reLU(x): 
    if x < 0: return 0.01 * x
    else: return 1

def PReLU(a, x):
    if x < 0: return a * x
    else: return  x

def RReLU_2(a, x):
    if x < 0: return a * x
    else: return  x

def ELU(x, a):
    if x < 0: return a * (exp(x) - 1) 
    else: return x
 def SELU (x, a, lambd):
    '''
    lambd = 1.0507
    a = 1.67326
    '''   
 def SReLU(a_r, a_l, x, t_r, t_l):
     if x <= t_l:
        return t_l + a_l * (x - t_l)
     else: 
        if x >= t_r:
            return t_r + a_r * (x - t_r)
        else: return x   
    
def ISRLU(x, a):
    if x < 0: return (x / (1 + a * x**2)**0.5)
    else: return x    
    
def SoftExponential(x, a):
    if a < 0: return -(log(1 - a(x + a))) / a
    elif a == 0: return x
    else: return (exp(a * x)-1) / a - a
    
def sinc(x):
    if x == 1: return 0
    else: return  sin(x) / x    

class ActivationFuncs:
    CONST          = lambda x: 1.0
    SIGMOID        = lambda x: 1.0 / (1.0 + np.exp(-x))
    TH             = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    softsign       = lambda x: x / (1 + abs(x))
    ReLu           = lambda x: ReLu(x)
    leaky_reLU     = lambda x: leaky_reLU(x)
    PReLU          = lambda a, x: PReLU(a, x)
    PReLU_2        = lambda a, x: RReLU_2(a, x)
    ELU            = lambda x, a:ELU(x, a)
    SELU           = lambda x, a, lambd: SELU (x, a, lambd)
    SReLU          = lambda a_r, a_l, x, t_r, t_l: SReLU(a_r, a_l, x, t_r, t_l)
    ISRLU          = lambda x, a: ISRLU(x, a)
    SoftPlus       = lambda x: log(1 + exp(x))
    Bent_identity  = lambda x: (((x**2 + 1)**0.5) - 1) / 2 + x
    SoftExponential= lambda x, a: SoftExponential(x, a)
    sinusoid       = lambda x: sin(x)
    sinc           = lambda x: sinc(x)
    gauss          = lambda x: exp(-(x ** 2))
    ISRU           = lambda x: x / (1 + a * x**2)**0.5 
    arctg          = lambda x: m.atan (x)

    @staticmethod
    def get_derivative(func):
        """ Get derivaty by function activation """
        base = {
            ActivationFuncs.CONST:           Derivatives.dConst,
            ActivationFuncs.SIGMOID:         Derivatives.dSigmoid,
            ActivationFuncs.TH:              Derivatives.dGTanh,
            ActivationFuncs.softsign         Derivatives.dsoftsign
            ActivationFuncs.ReLu             Derivatives.dReLu
            ActivationFuncs.leaky_reLU       Derivatives.dleaky_reLU
            ActivationFuncs.PReLU            Derivatives.dPReLU  
            ActivationFuncs.PReLU_2          Derivatives.dPReLU_2
            ActivationFuncs.ELU              Derivatives.dELU
            ActivationFuncs.SELU             Derivatives.dSELU
            ActivationFuncs.SReLU            Derivatives.dSReLU
            ActivationFuncs.ISRLU            Derivatives.dISRLU 
            ActivationFuncs.SoftPlus         Derivatives.dSoftPlus
            ActivationFuncs.Bent_identity    Derivatives.dBent_identity
            ActivationFuncs.SoftExponential  Derivatives.dSoftExponential
            ActivationFuncs.sinusoid         Derivatives.dsinusoid
            ActivationFuncs.sinc             Derivatives.dsinc
            ActivationFuncs.gauss            Derivatives.dgauss
            ActivationFuncs.ISRU             Derivatives.dISRU
            ActivationFuncs.arctg            Derivatives.darctg 
        }
        return base[func]

def dReLU_dx(x):
    if x < 0: return 0
    else: return 1
    
def dleaky_reLU_dx(x):
    if x < 0: return 0.01
    else: return 1 
    
def dPReLU_dx(a, x):
    if x < 0: return a
    else: return 1
    
def dRReLU_2_dx(a, x):
    if x < 0: return a
    else: return 1    
    
def dELU_dx(x, a):
    if x < 0: return a * (exp(x) - 1) + a * x
    else: return 1
    
def dSELU_dx(x, a, lambd):
    if x < 0: return lambd * exp(x) * a
    else: return lambd    
    
def dSReLU_dx(a_r, a_l, x, t_r, t_l):
    if x <= t_l:
        return a_l
    else: 
        if x >= t_r:
            return a_r
        else: return 1  
        
def dISRLU_dx(x, a):
    if x < 0: return (1 / (1 + a * x**2)**0.5)**3
    else: return 1   
    
def dSoftExponential_dx(x, a):
    if a < 0: return 1 / (1 - a * (a + x))
    else: return exp(a * x) 
    
def dsinc_dx(x):
    if x == 0: return 0
    else: return cos(x) / x - sin(x) / x**2    
    
class Derivatives:
    dConst          = lambda x: 0
    dSigmoid        = lambda x: ActivationFuncs.SIGMOID(x) * (1 - ActivationFuncs.SIGMOID(x))
    dGTanh          = lambda x: 1 - ActivationFuncs.TH(x) ** 2
    dsoftsign       = lambda x: 1 / (1 + abs(x))**2
    dReLu           = lambda x: dReLU_dx(x)
    dleaky_reLU     = lambda x: dleaky_reLU_dx(x)
    dPReLU          = lambda a, x: dPReLU_dx(a, x)
    dReLu_2         = lambda a, x: dRReLU_2_dx(a, x)
    dELU            = lambda x, a: dELU_dx(x, a)
    dSELU           = lambda x, a, lambd: dSELU_dx(x, a, lambd)
    dSReLU          = lambda a_r, a_l, x, t_r, t_l: dSReLU_dx(a_r, a_l, x, t_r, t_l)
    dISRLU          = lambda x, a: dISRLU_dx(x, a)
    dSoftPlus       = lambda x: 1 / (1 + exp(-x))
    dBent_identity  = lambda x: x / (2 * ((x**2 + 1)**0.5)) + 1
    dSoftExponential= lambda x, a: dSoftExponential_dx(x, a)
    dsinusoid       = lambda x: cos(x)
    dsinc           = lambda x: dsinc_dx(x)
    dgauss          = lambda x: (-2) * x * exp(-(x * x))
    dISRU           = lambda x: (1 / (1 + a * x**2)**0.5)**3
    darctg          = lambda x: 1 / (x**2 + 1)
