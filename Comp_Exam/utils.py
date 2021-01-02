# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 08:07:02 2020

@author: frieswd
"""
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

def D_Lele(N,h):
    d=[-1, 0, 1];
    B1=3/8*np.ones(N)
    B2=3/8*np.ones(N)
    B1[-2]=3
    B1[-3]=1/4
    B1[-4]=1/3
    B1[0]=1/4
    B1[1]=1/3
    B2[1]=3
    B2[2]=1/4
    B2[3]=1/3
    B2[-1]=1/4
    B2[-2]=1/3
    A=sp.sparse.spdiags([B1, np.ones(N), B2],d, N,N)
    
    alf=25/32/h
    bet=1/20/h
    gam=-1/480/h
    d=np.arange(-3,4)
    
    # d=-3
    B1=-gam*np.ones(N)
    B1[-4]=1/6/h
    B1[-5]=0
    B1[-6]=0
    # d=-2
    B2=-bet*np.ones(N)
    B2[0]=-1/36/h
    B2[-3]=-3/2/h
    B2[-4]=0
    B2[-5]=-1/36/h
    # d = -1
    B3=-alf*np.ones(N)
    B3[0]=-3/4/h
    B3[1]=-7/9/h
    B3[-2]=-3/2/h
    B3[-3]=-3/4/h
    B3[-4]=-7/9/h
    # d = 0
    B4=np.zeros(N)
    B4[0]=-17/6/h
    B4[-1]=17/6/h
    # d = 1
    B5=alf*np.ones(N)
    B5[1]=3/2/h
    B5[2]=3/4/h
    B5[3]=7/9/h
    B5[-1]=3/4/h
    B5[-2]=7/9/h
    # d = 2
    B6=bet*np.ones(N)
    B6[2]=3/2/h
    B6[3]=0
    B6[4]=1/36/h
    B6[-1]=1/36/h
    # d = 3
    B7=gam*np.ones(N)
    B7[3]=-1/6/h
    B7[4]=0
    B7[5]=0
    B=sp.sparse.spdiags([B1, B2, B3, B4, B5, B6, B7],d,N,N)
    return sp.sparse.linalg.spsolve(A.tocsc(),B.tocsc()) 
