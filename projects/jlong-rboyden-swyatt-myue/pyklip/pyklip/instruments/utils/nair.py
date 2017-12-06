__author__ = 'Jason Wang'
import numpy as np

def nMathar(wv, P, T, H=10):
	"""
	Calculate the index of refraction as given by Mathar (2008): http://arxiv.org/pdf/physics/0610256v2.pdf
	***Only valid for between 1.3 and 2.5 microns!

	Inputs:
		wv: wavelength in microns
		P:  Pressure in Pa
		T:  Temperature in Kelvin
		H:  relative humiditiy in % (i.e. between 0 and 100)
	Return:
		n:  index of refratoin
	"""

	#polynomial expansion in wavenumber
	wvnum = 1.e4/wv #cm^-1 	#convert to wavenumbers
	wvnum0 = 1.e4/2.25 #cm^-1 #series expand around this wavenumber

	#do a 6th order expansion
	powers = np.arange(0,6)
	#calculate expansion coefficients
	coeffs = GetCoeff(powers, P, T, H)

	#sum of the power series expansion
	n = 1.0
	for coeff,power in zip(coeffs, powers):
		n += coeff * ((wvnum - wvnum0)**power)

	return n

def GetCoeff(i, P, T, H):
	"""
	Calculate the coefficients for the polynomial series expansion of index of refraction (Mathar (2008))
	***Only valid for between 1.3 and 2.5 microns!

	Inputs:
		i:  degree of expansion in wavenumber
		P:  Pressure in Pa
		T:  Temperature in Kelvin
		H:  relative humiditiy in % (i.e. between 0 and 100)
	Return:
		coeff:  Coefficient [cm^-i]
	"""

	#name all the constants in the model
	#series expansion in evironment parameters
	T0 = 273.15 + 17.5 #Kelvin
	P0 = 75000 #Pa
	H0 = 10 #%

	#delta terms for the expansion
	dT = 1./T - 1./T0
	dP = P - P0
	dH = H - H0

	#loads and loads of coefficients, see equation 7 in Mathar (2008)
	#use the power (i.e. i=[0..6]) to index the proper coefficient for that order
	cref= np.array([ 0.200192e-3, 0.113474e-9, -0.424595e-14, 0.100957e-16,-0.293315e-20, 0.307228e-24]) # cm^i
	cT  = np.array([ 0.588625e-1,-0.385766e-7,  0.888019e-10,-0.567650e-13, 0.166615e-16,-0.174845e-20])  # K cm^i
	cTT = np.array([-3.01513,     0.406167e-3, -0.514544e-6,  0.343161e-9, -0.101189e-12, 0.106749e-16]) # K^2 cm^i
	cH  = np.array([-0.103945e-7, 0.136858e-11,-0.171039e-14, 0.112908e-17,-0.329925e-21, 0.344747e-25]) # cm^i / %
	cHH = np.array([ 0.573256e-12,0.186367e-16,-0.228150e-19, 0.150947e-22,-0.441214e-26, 0.461209e-30]) # cm^i / %^2
	cP  = np.array([ 0.267085e-8, 0.135941e-14, 0.135295e-18, 0.818218e-23,-0.222957e-26, 0.249964e-30]) # cm^i / Pa
	cPP = np.array([ 0.609186e-17,0.519024e-23,-0.419477e-27, 0.434120e-30,-0.122445e-33, 0.134816e-37]) # cm^i / Pa^2
	cTH = np.array([ 0.497859e-4,-0.661752e-8,  0.832034e-11,-0.551793e-14, 0.161899e-17,-0.169901e-21]) # cm^i K / %
	cTP = np.array([ 0.779176e-6, 0.396499e-12, 0.395114e-16, 0.233587e-20,-0.636441e-24, 0.716868e-28]) # cm^i K / Pa
	cHP = np.array([-0.206567e-15,0.106141e-20,-0.149982e-23, 0.984046e-27,-0.288266e-30, 0.299105e-34]) # cm^i / Pa %

	# use numpy arrays to calculate all the coefficients at the same time
	coeff = cref[i] + cT[i]*dT + cTT[i]*(dT**2) + cH[i]*dH + cHH[i]*(dH**2) + cP[i]*dP + cPP[i]*(dP**2) + cTH[i]*dT*dH + cTP[i]*dT*dP + cHP[i]*dH*dP

	return coeff


def nRoe(wv,P,T,fh20=0.0):
	"""
	Compute n for air from the formula in Henry Roe's PASP paper: http://arxiv.org/pdf/astro-ph/0201273v1.pdf
	which in turn is referenced from Allen's Astrophysical Quantities.

	Inputs:
		wv: wavelength in microns
		P:  pressure in Pascal
		T:  temperature in Kelvin
		fh20:fractional partial pressure of water (typically between 0 and 4%)
	Return:
		n:  index of refraction of air
	"""

	#convert pressure from Pa to mbar
	P /= 100.

	#some constants in the function for n
	a1 = 64.328
	a2 = 29498.1
	a3 = 146.0
	a4 = 255.4
	a5 = 41.0

	Ts = 288.15   # K
	Ps = 1013.35 # mb

	#calculate n-1 for dry air
	K1 = 1e-6*(P/Ps * Ts/T)
	n1 = K1*(a1 + a2/(a3-wv**(-2)) + a4/(a5-wv**(-2))      )

	# water vapor correction
	# fraction parital pressure of water vapor is 0-4%
	K2 = -43.49e-6 * fh20
	a6 = -7.956e-3
	nh2o = K2*(1 + a6*wv**(-2))

	return n1 + nh2o + 1