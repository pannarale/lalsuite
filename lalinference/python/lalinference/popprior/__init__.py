# Copyright (C) 2016  Heather Fong, Kipp Cannon
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# copied from calculations_updated.py, restructured to input different rho for select t_k
# sample mass distribution directly from template bank
# new P_tj calculation: using Voronoi diagram
# use Chebyshev nodes to compute values of rho
# Edited with Kipp, July 25

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import special
from scipy.misc import logsumexp
import time
import random
from scipy.optimize import curve_fit
import collections
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d, ConvexHull
import types
from scipy.interpolate import lagrange
from glue.text_progress_bar import ProgressBar

def extended_templates(m,N=50):
    # Builds a buffered ConvexHull around the template in order to bound the Voronoi regions
    n = np.shape(m)[-1] # number of dimensions in template bank
    hull = ConvexHull(m)
    gauss = np.random.randn(N,n)
    new_points = np.zeros(np.shape(gauss))
    for i in range(N):
        new_points[i] = (0.5/gauss[i].dot(gauss[i]))*gauss[i]
    surr = new_points+m[hull.vertices[0]]
    for j in range(len(hull.vertices)-1):
        surr = np.concatenate((surr,new_points+m[hull.vertices[j+1]])) # array of points centred around each ConvexHull vertex
    hull2 = ConvexHull(surr) # Create a new ConvexHull of this array
    return surr[hull2.vertices], Voronoi(np.concatenate((surr[hull2.vertices],m))) # return the vertices of the new convex hull, and the Voronoi object of the combined (bank + new convex hull vertices)

def find_voronoi_volumes(m):
    # Find and compute the volume of each Voronoi region in the bounded Voronoi diagram
    boundary, vor = extended_templates(m) # Create the bounded Voronoi diagram
    all_v = []
    all_p = []
    all_i = []
    length = len(vor.points)
    progress = ProgressBar(max=length)
    for i, p in enumerate(vor.points):
        progress.increment(text=str(i)+"/"+str(length))
        if p not in boundary: # if the template is in the original bank
            region = vor.regions[vor.point_region[i]]
            if -1 in region: # check that the Voronoi region is bounded
                print "infinite region found", i, p
            else:
                pvol = vol(vor, region) # Calculate volume of Voronoi region
                all_v.append(pvol)
                all_p.append(p)
                all_i.append(i-len(boundary))
    print "Voronoi diagram, total volume = "+str(sum(all_v))
    output_string = 'uberbank_voronoi_volumes.hdf5'
    output = h5py.File(output_string,"w")
    output.create_dataset('bank_volumes',data=np.array(all_v))
    output.create_dataset('bank_templates',data=np.array(all_p))
    output.create_dataset('bank_indices',data=np.array(all_i))
    output.close()
    return output_string

def find_ln_p_j(m, f, popt, hdf5_file):
    # Calculate ln(P_j)
    vol = h5py.File(hdf5_file,'r')
    v, p, i = vol['bank_volumes'].value, vol['bank_templates'].value, vol['bank_indices'].value
    # Calculate probability distributions for the source population
    p_m1 = f(p[:,0],*popt)/np.trapz(f(p[:,0][np.argsort(p[:,0])],*popt),p[:,0][np.argsort(p[:,0])]) 
    p_m2 = f(p[:,1],*popt)/np.trapz(f(p[:,1][np.argsort(p[:,1])],*popt),p[:,1][np.argsort(p[:,1])])
    p_a1 = np.ones(len(i))
    p_a2 = np.ones(len(i))
    p_tj = p_m1*p_m2*p_a1*p_a2*np.array(v) # Calculate P_j (unnormalized)
    return np.log(p_tj)-np.log(sum(p_tj)), p, i, v
    
def find_ln_p_j_voronoi(m, f, popt):
    vor = Voronoi(m)
    all_vols = []
    all_p = []
    all_i = []
    length = len(vor.points)
    progress = ProgressBar(max=length)
    for i, p in enumerate(vor.points): # vor.points = coordinates of input points, where coordinates = [m1, m2, chi1, chi2]
        progress.increment(text=str(i)+"/"+str(length))
        region = vor.regions[vor.point_region[i]]
        # vor.point_region[i] = index of Voronoi region for each input point
        # vor.regions[...] = indices of the Voronoi vertices forming Voronoi region (vor.vertices = coordinates of the Voronoi vertices)
        if -1 not in region:
            pvol = vol(vor, region) # calculates hypervolume of the Voronoi region around the input point
            #if pvol > 50. and (p[0]+p[1]) < 10.: # if volume is too large, it probably means the input point is a template near the edge of the bank. So ignore it.
            #    print p, pvol
            #    pvol = 5.
            #elif pvol > 50.0 and p[1] > 3. and (p[0]+p[1]) > 23.:
            #    print p, pvol
            #    pvol = 50.
            all_vols.append(pvol)
            all_p.append(p)
            all_i.append(i)
    print "Voronoi diagram, total volume = "+str(sum(all_vols))
    print "Median:", np.median(all_vols)
    print "Histogram (value of histogram, bin edges):", np.histogram(all_vols,bins=100)
    all_p = np.array(all_p)
    all_i = np.array(all_i)
    p_m1 = f(all_p[:,0],*popt)/np.trapz(f(all_p[:,0][np.argsort(all_p[:,0])],*popt),all_p[:,0][np.argsort(all_p[:,0])])
    p_m2 = f(all_p[:,1],*popt)/np.trapz(f(all_p[:,1][np.argsort(all_p[:,1])],*popt),all_p[:,1][np.argsort(all_p[:,1])])
    p_tj = p_m1*p_m2*np.array(all_vols)
    return np.log(p_tj)-np.log(sum(p_tj)), all_p, all_i

def trivol((a, b, c)):
    # Calculates area of triangle, for templates with coordinates (m1, m2)
    return abs(np.cross((c-a),(c-b)))/2.

def tetravol((a, b, c, d)):
    # Calculates volume of tetrahedron, given vertices a, b, c, d (triplets), for templates with coordinates (m1, m2, chi_eff)
    return abs(np.dot((a-d), np.cross(b-d),(c-d)))/6.

def hypertetravol((a, b, c, d, e)):
    # Calculates a Delaunay triangulated section of the Voronoi region for templates with coordinates (m1, m2, chi1, chi2)
    # Calculates hypervolume of 4d tetrahedron (5-cell), given vertices a, b, c, d, e (quadruplets)
    return abs( (1./24.)*np.linalg.det((a-e, b-e, c-e, d-e)) )

def vol(vor, region):
    # Computes the volume of the Voronoi region using Delaunay triangulation
    dpoints = vor.vertices[region] # finds Voronoi vertices coordinates in the Voronoi region, these are the points to triangulate
    #return sum(tetravol(dpoints[simplex]) for simplex in Delaunay(dpoints).simplices)
    #return sum(trivol(dpoints[simplex]) for simplex in Delaunay(dpoints).simplices)
    return sum(hypertetravol(dpoints[simplex]) for simplex in Delaunay(dpoints).simplices) # takes dpoints and triangulates the region. simplices calls the indices of the points forming the simplices of the triangulation. Sum up all the hypervolumes to get total volume for the Voronoi region.
    
def ln_p_k(ovrlp, rho, t_k, N=4, acc=0.001):
    # Finds probability of template t_k fitting data, given that the signal is rho*t_j
    # ovrlp = vector of the overlaps of (t_j, t_k) for one t_j
    # rho = SNR (float)
    # t_k = templates
    ln_num = ln_p_k_num(ovrlp[t_k],rho,N) # compute an array of numerator terms of p_k in order of t_k
    # for full template bank, don't need ln_p_k_den. just need to do a logexpsum of ln_num.
    ln_den = ln_p_k_den(ovrlp,rho,N,acc=acc) # compute the denominator of p_k (single float value)
    return ln_num-ln_den # returns array of values

def ln_p_k_num(tjtk, rho, N, sqrtpiover2 = np.sqrt(np.pi/2.), sqrt2 = np.sqrt(2.)):
    x = np.array(tjtk*rho)
    #if rho == 0:
    #    return np.zeros(len(x))
    halfxsquared = 0.5*x**2.
    if N == 5:
        return halfxsquared +np.log( sqrtpiover2*(x**4.+6.*x**2.+3)*(special.erf(x/sqrt2))+np.exp(-halfxsquared)*(x**3.+5.*x)) # N = 5
    if N == 4:
        return halfxsquared + np.log( sqrtpiover2*(x**3.+3.*x)*(1.+special.erf(x/sqrt2))+np.exp(-halfxsquared)*(x**2.+2.) ) # N = 4
    if N == 3:
        return halfxsquared + np.log( sqrtpiover2*(x**2.+1.)*(1.+special.erf(x/sqrt2))+np.exp(-halfxsquared)*x ) # N = 3

def ln_p_k_den(tjtk, rho, N, acc=0.001):
    # Finds the denominator part of the probability P(t_k | t_j) for N=4 dimensions (m1, m2, chi_eff, rho)
    # P(t_k | t_j) is a template for the signal rho*t_j (tjtk is the overlap between t_k and t_j)
    # acc = accuracy we wish to obtain
    # Denominator is the sum of all numerator terms
    if rho < 10: # must process full array
        return logsumexp(ln_p_k_num(tjtk,rho,N))
    lny_list = []
    limit = np.log(acc/len(tjtk))
    for w in np.sort(tjtk)[::-1]:
        lny_list.append(ln_p_k_num(w,rho,N))
        if lny_list[-1]-lny_list[0] < limit:
            break
    return logsumexp(lny_list)

def source_population(srcfile):
    if "_DNS" in srcfile:
        pguess = [2.5, 1., 1.3]
    if "BBH" in srcfile:
        pguess = [0.2, 1./50., 30.]
    data = np.loadtxt(srcfile,unpack=True)
    x = data[0]
    y = data[1]/np.trapz(data[1],x) # normalize the probability
    f = lambda x, *p: p[0]*np.exp(-p[1]*(x-p[2])**2)
    popt, pcov = curve_fit(f, x, y, pguess) # fit p(m) to a Gaussian curve
    print popt, pcov
    return f, popt

def load_overlaps(h5_file):
    return h5_file[h5_file.keys()[0]]['overlaps'].value

def load_bank(h5_file):
    # return masses and chis
    return np.vstack((h5_file[h5_file.keys()[0]]['mass1'].value, h5_file[h5_file.keys()[0]]['mass2'].value)).T
    #return np.vstack((h5_file[h5_file.keys()[0]]['mass1'].value, h5_file[h5_file.keys()[0]]['mass2'].value, h5_file[h5_file.keys()[0]]['spin1z'].value, h5_file[h5_file.keys()[0]]['spin2z'].value))
