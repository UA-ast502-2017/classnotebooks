__author__ = 'JB'

import numpy as np

def gauss2d(x, y, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0, theta = 0, offset = 0):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g


def hat(x, y, radius):
    r2 = x**2+y**2
    return np.array(r2 <= radius*radius,dtype=np.int)

def model_exp(x,m,alpha):
    return np.exp(-alpha*x-m)

def LSQ_model_exp(x,y,m,alpha):
    y_model = model_exp(x,m,alpha)
    return (y-y_model)/np.sqrt(y_model)