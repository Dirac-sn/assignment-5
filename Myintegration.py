# -*- coding: utf-8 -*-
import numpy as np 
from scipy.special import roots_legendre
from scipy.special import roots_laguerre
from scipy.special import roots_hermite

def MyTrap(func,a,b,m=int(1e3),d=None):
    """
    Integrate `func` from `a` to `b` using composite trapezoidal rule. If `d` is not passed as argument during call then `m` is the number of uniformly spaced subintervals in the interval `[a,b]`. 
    When `d` is passed during call, the returned integral is accurate to atleast `d` significant digits, using a fixed relative-tolerance of ``0.5x10**-d``; Also `m` is now considered to be the upper-limit on the number of subintervals during the calculation of fixed-tolerance integral.
    Parameters
    ----------
    func : function
        A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    m : int, optional
        Number of subintervals for integration and if `d` is given this gives the maximum number of subintervals to calculate for before aborting.
    d : Integer, optional
        Number of significant digits required in the returned integral. 
        Iteration stops when relative error between last two iterates is less than or equal to
        `0.5*10**(-d)`.
    Returns
    -------
    if `d` is not passed as argument.
    val : float
        Trapezoidal rule's approximation to the integral. 
    if `d` is passed as argument.
    val : float
        Trapezoidal rule's approximation(Fixed-tolerance) to integral to d significant digits.
    m : float
        Number of subintervals used for calculating the integral in the last iteration.
    """
    if d is not None and m!=1:
        max_n = np.floor(np.log2(m))
        m_array = np.logspace(0,max_n,base=2,num = int(max_n+1))
        I = np.zeros(m_array.shape)
        h = (b-a)/m_array
        x = np.linspace(a,b,int(m_array[0]+1))
        y = func(x)
        I[0] = (h[0]/2)*np.sum(y[:-1] + y[1:])
        for i in np.arange(0,m_array.shape[0]):
            x = np.linspace(a,b,int(m_array[i]+1))
            midx  = x[1::2]
            I[i] = (1/2)*I[i-1] + h[i]*(np.sum(func(midx)))
            if np.abs(I[i]-I[i-1]) <= 0.5/10**d*np.abs(I[i]):
                val,last_m =I[i],m_array[i]
                return val,last_m
        print("Could not reach desired accuracy with the given upperlimit on the number of intervals.(m) ")
        val,last_m = I[-1],m_array[-1]
        return val,last_m
    x = np.linspace(a,b,m+1)
    y = func(x)
    val = (b-a)/(2*m)*np.sum(y[:-1]+y[1:])
    return val

def MySimp(func,a,b,m=int(1e3),d=None):
    """
    Integrate `func` from `a` to `b` using composite simpson1/3 rule. If `d` is not passed as argument during call then `m` is the number of uniformly spaced subintervals in the interval `[a,b]`. 
    When `d` is passed during call, the returned integral is accurate to atleast `d` significant digits, using a fixed relative-tolerance of ``0.5x10**-d``; Also `m` is now considered to be the upper-limit on the number of subintervals during the calculation of fixed-tolerance integral.
    Parameters
    ----------
    func : function
        A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    m : int, optional
        Number of subintervals for integration and if `d` is given this gives the maximum number of subintervals to calculate for before aborting.
    d : Integer, optional
        Number of significant digits required in the returned integral. 
        Iteration stops when relative error between last two iterates is less than or equal to
        `0.5*10**(-d)`.
    Returns
    -------
    if `d` is not passed as argument.
    val : float
        Simpson1/3 rule's approximation to the integral. 
    if `d` is passed as argument.
    val : float
        Simpson1/3 rule's approximation to integral to d significant digits.
    m : float
        Number of subintervals used for calculating the integral in the last iteration.
    """
    if d is not None and m!=1:
        max_n = np.floor(np.log2(m))
        m_array = np.logspace(1,max_n,base=2,num = int(max_n))
        I = np.zeros(m_array.shape)
        h = (b-a)/m_array
        x = np.linspace(a,b,int(m_array[0]+1))
        y = func(x)
        I[0] = (h[0]/3)*np.sum(y[:-1:2] +4*y[1:-1:2] +y[2::2])
        omidy = y[1::2]
        for i in np.arange(1,m_array.shape[0]):
            x = np.linspace(a,b,int(m_array[i]+1))
            midx = x[1::2] 
            midy = func(midx) 
            I[i] = I[i-1]/2 + h[i]*(4*np.sum(midy) - 2*np.sum(omidy))/3 
            if np.abs(I[i]-I[i-1]) <= 0.5/10**d*np.abs(I[i]):
                val,last_m =I[i],m_array[i]
                return val,last_m
            omidy = midy.copy()
        print("Could not reach desired accuracy with the given upperlimit on the number of intervals.(m) ")
        val,last_m = I[-1],m_array[-1]
        return val,last_m
    x = np.linspace(a,b,m+1)
    y = func(x)
    val = (b-a)/(3*m)*np.sum(y[:-1:2]+4*y[1::2]+y[2::2])
    return val

def MyLegQuadrature(func,a,b,n=5,m=100,d=None):
    """
    Integrate `func` from `a` to `b` using composite Gaussian Quadrature Method. If `d` is not passed as argument during call then `m` is the number of uniformly spaced subintervals in the interval `[a,b]`. 
    When `d` is passed during call, the returned integral is accurate to atleast `d` significant digits, using a fixed relative-tolerance of ``0.5x10**-d``; Also `m` is now considered to be the upper-limit on the number of subintervals during the calculation of fixed-tolerance integral.
    Parameters
    ----------
    func : function
        A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : Integer 
        No. of points to integrate at.
    m : int, optional
        Number of subintervals for integration and if `d` is given this gives the maximum number of subintervals to calculate for before aborting.
    d : Integer, optional
        Number of significant digits required in the returned integral. 
        Iteration stops when relative error between last two iterates is less than or equal to
        `0.5*10**(-d)`.
    Returns
    -------
    if `d` is not passed as argument.
    val : float
        Gaussian quadrature approximation to the integral. 
    if `d` is passed as argument.
    val : float
        Gaussian quadrature approximation to integral to d significant digits.
    m : float
        Number of subintervals used for calculating the integral in the last iteration.
    """
    x,w = roots_legendre(n)
    if np.isinf(a) or np.isinf(b):
        raise ValueError("Gaussian quadrature is only available for finite limits.")
    if d is not None and m!=1:
        max_n = np.floor(np.log2(m))
        m_array = np.logspace(0,max_n,base=2,num = int(max_n+1))
        I = np.zeros(m_array.shape)
        for i in np.arange(0,m_array.shape[0]):
            I[i] = MyLegQuadrature(func,a,b,n,m_array[i])
            if i == 0 :
                continue
            if np.abs(I[i]-I[i-1]) < 0.5/10**d*np.abs(I[i]):
                val,last_m = I[i],m_array[i]
                return val,last_m
        print("Could not reach desired accuracy with the given upperlimit on the number of intervals.(m) ")
        val,last_m = I[-1],m_array[-1]
        return val,last_m
    I,val = 0,0
    subs = np.linspace(a,b,int(m+1))
    l = len(subs)
    for a_i,b_i in zip(subs[:l-1],subs[1:l]):
        shifted_x = (b_i-a_i)*(x+1)/2 + a_i
        I+= (b_i-a_i)/2 * np.sum(w*func(shifted_x))
    val = I
    return val

def MyLeguQuad(func,n=4):
    x,w = roots_laguerre(n)
    return(w.dot(func(x)*np.exp(x)))

def MyHermiteQuad(func,n=4):
    x,w = roots_hermite(n)
    return(w.dot(func(x)*np.exp(x**2)))


MyTrap = np.vectorize(MyTrap)
MySimp = np.vectorize(MySimp)
MyLegQuadrature = np.vectorize(MyLegQuadrature)

MyLeguQuad = np.vectorize(MyLeguQuad)
MyHermiteQuad = np.vectorize(MyHermiteQuad)