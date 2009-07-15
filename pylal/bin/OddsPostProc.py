#!/usr/bin/env python

#from numpy import *
import scipy
import matplotlib 
matplotlib.use("Agg")
import sys
import math
from pylab import *
from optparse import OptionParser
import os
import numpy

parser=OptionParser()
parser.add_option("-o","--outpath", dest="outpath",help="make page and plots in DIR", metavar="DIR")
parser.add_option("-N","--Nlive",dest="Nlive",help="number of live points for each of the files")
parser.add_option("-d","--data",dest="data",action="append",help="datafile")

(opts,args)=parser.parse_args()

def logadd(a,b):
    if(a>b): (a,b)=(b,a)
    return (b+log(1+exp(a-b)))
    
def mc2ms(mc,eta):
    root = sqrt(0.25-eta)
    fraction = (0.5+root) / (0.5-root)
    invfraction = 1/fraction

    m1= mc * pow((1+fraction),0.2) / pow(fraction,0.6)

    m2= mc* pow(1+invfraction,0.2) / pow(invfraction,0.6)
    return (m1,m2)

def histN(mat,N):
    Nd=size(N)
    histo=zeros(N)
    scale=array(map(lambda a,b:a/b,map(lambda a,b:(1*a)-b,map(max,mat),map(min,mat)),N))
    axes=array(map(lambda a,N:linspace(min(a),max(a),N),mat,N))
    bins=floor(map(lambda a,b:a/b , map(lambda a,b:a-b, mat, map(min,mat) ),scale*1.01))
    
    hbins=reshape(map(int,bins.flat),bins.shape)
    for co in transpose(hbins):
        t=tuple(co)
        histo[t[::-1]]=histo[t[::-1]]+1
    return (axes,histo)

def nest2pos(samps,Nlive):
    weight = -linspace(1,len/Nlive,len)
    weight = weight + samps[:,-1]
    maxwt = max(weight)
    randoms = rand(len)
    pos = zeros(size(samps,1))
    posidx = find(weight>maxwt+log(randoms))
    pos=samps[posidx,:]
    return pos

outdir=opts.outpath
Nlive=int(opts.Nlive)
print 'Loading ' + opts.data[0]
d=load(opts.data[0])
for infile in opts.data[1:]:
    print 'Loading ' + infile
    tmp=load(infile)
    d=numpy.vstack((d,tmp))

Bfile = opts.data[0]+'_B.txt'
print 'Looking for '+Bfile
if os.access(Bfile,os.R_OK):
    outstat = load(Bfile)
    NoiseZ = outstat[2]
    Bflag=1
else: Bflag=0

Nlive = Nlive * size(opts.data,0)

len=size(d,0)
Nd=size(d,1)
sidx=argsort(d[:,9])
d=d[sidx,:]
d[:,0]=exp(d[:,0])
print 'Exponentiated mc'
maxL = d[-1,-1]

print 'maxL = ' + str(maxL)
# Maximum likelihood point
print 'Max Likelihood point:'
maxPt= map(str,d[-1,:])
out=reduce(lambda a,b: a + ' || ' + b,maxPt)
print '|| ' + out + ' ||'

pos = nest2pos(d,Nlive)

print "Number of posterior samples: " + str(size(pos,0))
# Calculate means
means = mean(pos,axis=0)
meanStr=map(str,means)
out=reduce(lambda a,b:a+'||'+b,meanStr)
print 'Means:'
print '||'+out+'||'

#pos[:,2]=pos[:,2]-means[2]

print 'Applying nested sampling algorithm to ' + str(len) + ' samples'

def nestZ(d,Nlive):
    logw = log(1 - exp(-1.0/Nlive))
    logZ = logw + d[0,-1]
    logw = logw - 1.0/Nlive
    len=size(d,0)
    H=0
    for i in linspace(1,len-2,len):
        logZnew=logadd(logZ,logw+d[i,-1])
        H = exp(logw + d[i,-1] -logZnew)*d[i,-1] \
            + exp(logZ-logZnew)*(H+logZ) - logZnew
        logw = logw - 1.0/Nlive
        logZ=logZnew
    return (logZ,H)

(logZ,H)=nestZ(d,Nlive)

if(Bflag==1):
    BayesFactor = logZ - NoiseZ
    print 'log B = '+str(BayesFactor)
    
#for i in range(0,Nd):
#    for j in range(i+1,Nd):
#        subplot(Nd,Nd,i*Nd+j)
#        hexbin(pos[:,i],pos[:,j])

#hist(array([d[:,1], d[:,2]]))

foo=array([pos[:,0],pos[:,1]])
(bins,myhist)=histN(foo,[30,30])
myfig=figure(1,figsize=(6,4),dpi=80)
contourf(bins[0],bins[1],myhist)
plot([foo[0,-1]],[foo[1,-1]],'x')
grid()
xlabel('chirp mass (Msun)')
ylabel('eta')
myfig.savefig(outdir+'/Meta.png')

myfig.clear()
lims=(min(pos[:,6]),min(pos[:,5]),max(pos[:,6]),max(pos[:,5]))
#myfig.add_axes(lims)
foo=array([pos[:,5],pos[:,6]])
(bins,myhist)=histN(foo,[30,30])
#myfig=figure(2)
contourf(bins[0],bins[1],myhist)
plot([foo[0,-1]],[foo[1,-1]],'x')
grid()
xlabel('Right Ascension')
ylabel('Declination')
myfig.savefig(outdir+'/RAdec.png')

myfig.clear()
foo=array([pos[:,7],pos[:,8]])
(bins,myhist)=histN(foo,[30,30])
#myfig=figure(3)
contour(bins[0],bins[1],myhist)
plot([foo[0,-1]],[foo[1,-1]], 'x')
grid()
xlabel('psi')
ylabel('iota')
myfig.savefig(outdir+'/psiiota.png')
myfig.clear()

(m1,m2)=mc2ms(pos[:,0],pos[:,1])
foo=array([m1,m2])
(bins,myhist)=histN(foo,[30,30])
#myfig=figure()
contourf(bins[0],bins[1],myhist)
xlabel('mass 1')
ylabel('mass 2')
grid()
myfig.savefig(outdir+'/m1m2.png')
myfig.clear()

foo=array([pos[:,4],pos[:,8]])
(bins,myhist)=histN(foo,[30,30])
contourf(bins[0],bins[1],myhist)
xlabel('distance')
ylabel('iota')
grid()
myfig.savefig(outdir+'/Diota.png')
myfig.clear()

paramnames=('Mchirp (Msun)','eta','geocenter time ISCO','phi_c','Distance (Mpc)','RA (rads)','declination (rads)','psi','iota')

htmlfile=open(outdir+'/posplots.html','w')
htmlfile.write('<HTML><HEAD><TITLE>Posterior PDFs</TITLE></HEAD><BODY><h3>Posterior PDFs</h3>')
if(Bflag==1): htmlfile.write('<h4>log Bayes Factor: '+str(BayesFactor)+'</h4><br>')
htmlfile.write('signal evidence: '+str(logZ)+'. Fisher information: '+str(H*1.442)+' bits.<br>')
htmlfile.write('Produced from '+str(size(pos,0))+' posterior samples, taken from '+str(len)+' NS samples using '+str(Nlive)+' live points<br>')
htmlfile.write('<h4>Mean parameter estimates</h4>')
htmlfile.write('<table border=1><tr>')
paramline=reduce(lambda a,b:a+'<td>'+b,paramnames)
htmlfile.write('<td>'+paramline+'<td>logLmax</tr><tr>')
meanline=reduce(lambda a,b:a+'<td>'+b,meanStr)
htmlfile.write('<td>'+meanline+'</tr></table>')

htmlfile.write('<table border=1><tr>')
htmlfile.write('<td width=30%><img width=100% src="m1m2.png"></td>')
htmlfile.write('<td width=30%><img width=100% src="RAdec.png"></td>')
htmlfile.write('<td width=30%><img width=100% src="Meta.png"></td>')
htmlfile.write('</table>')
htmlfile.write('<br><hr>')

for i in [0,1,2,3,4,5,6,7,8]:
    myfig=figure(figsize=(4,3.5),dpi=80)
    hist(pos[:,i],50,normed='true')
    grid()
    xlabel(paramnames[i])
    ylabel('Probability Density')
    myfig.savefig(outdir+'/'+str(i) + '.png')
    htmlfile.write('<img src="'+str(i)+'.png">')


htmlfile.write('</BODY></HTML>')
htmlfile.close()
