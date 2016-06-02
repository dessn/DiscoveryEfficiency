#!/usr/bin/env python
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



zp=31.4
filename = 'Y1reprocFakes/FAKEMATCH_ALL.OUT'
indeces = [3, 18, 31, 50, 51, 52, 53, 24, 1]

data = []
shit = []
with open(filename) as f:
    f.readline()
    varnames= f.readline().split()
    varnames =  numpy.array(varnames)
    # print varnames[indeces]
    for line in f:
        if line.strip():
            values = line.split()
            values = numpy.array(values)
            values = values[indeces]
            data.append(values)
            shit.append(line)

data = numpy.array(data)
band= data[:, 0]
psf= data[:, 1].astype('float')
galsb = data[:, 2].astype('float')
flux = data[:, 3].astype('float')
fluxerr = data[:, 4].astype('float')
errmask = data[:, 5].astype('int')
autoscan = data[:,6].astype('float')
maskfrac = data[:,7].astype('float')

bad = numpy.bitwise_and(errmask, 240)
good = numpy.where(numpy.logical_and(bad ==0, maskfrac<0.1))[0]

ston = flux[good]/fluxerr[good]
galflux =  numpy.pi*psf[good]**2 * 10**(-0.4*(galsb[good]-zp))
fracincrease = flux[good]/galflux
galsb = galsb[good]
found = autoscan[good] >=0.5

with PdfPages('multipage_pdf.pdf') as pdf:

    xedges = numpy.arange(2,20.5,1)
    yedges = (2**numpy.arange(7))/8. #numpy.arange(0,210,50)

    H, xedges, yedges = numpy.histogram2d(ston, fracincrease, bins=[xedges, yedges])
    xticks = (0.5*(xedges+numpy.roll(xedges,-1)))[:-1]
    yticks = (0.5*(yedges+numpy.roll(yedges,-1)))[:-1]

    plt.hist2d(ston, numpy.log(fracincrease), bins=[xedges, numpy.log(yedges)])
    plt.colorbar()
    plt.title("Number of Fakes Per Bin")
    plt.xlabel('STON')
    plt.ylabel('log(Fraction Flux Increase)')
    pdf.savefig()
    plt.close()

    eff = numpy.array(H)
    effston = numpy.zeros(len(xedges)-1)
    eff[:]=0.
    for i in xrange(len(xedges)-1):
        inx = numpy.logical_and(ston > xedges[i], ston < xedges[i+1])
        if inx.sum() !=0:
            effston[i] = 1.*found[inx].sum() / inx.sum()
        for j in xrange(len(yedges)-1):
            iny = numpy.logical_and(fracincrease > yedges[j], fracincrease <= yedges[j+1])
            if H[i,j] !=0:
                inxy = numpy.logical_and(inx, iny)
                eff[i,j] = found[inxy].sum()/H[i,j]



    for i in xrange(len(yedges)-1):
        plt.plot(xticks, eff[:,i], label="[{:.2f}, {:.2f}]".format(yedges[i],yedges[i+1]))
    plt.plot(xticks, effston, label='all')
    plt.xlabel('STON')
    plt.ylabel('eff')
    plt.title('Fraction Increase in Seeing Disk')
    plt.ylim([0,1.05])
    plt.legend(loc=4)
    pdf.savefig()
    plt.close()

    xedges = numpy.arange(2,20.5,1)
    yedges = numpy.arange(17,28,2)

    H, xedges, yedges = numpy.histogram2d(ston, galsb, bins=[xedges, yedges])

    eff = numpy.array(H)
    effston = numpy.zeros(len(xedges)-1)
    eff[:]=0.
    for i in xrange(len(xedges)-1):
        inx = numpy.logical_and(ston > xedges[i], ston < xedges[i+1])
        if inx.sum() !=0:
            effston[i] = 1.*found[inx].sum() / inx.sum()
        for j in xrange(len(yedges)-1):
            iny = numpy.logical_and(galsb > yedges[j], galsb <= yedges[j+1])
            if H[i,j] !=0:
                inxy = numpy.logical_and(inx, iny)
                eff[i,j] = found[inxy].sum()/H[i,j]

    xticks = (0.5*(xedges+numpy.roll(xedges,-1)))[:-1]

    for i in xrange(len(yedges)-1):
        plt.plot(xticks, eff[:,i], label="[{:.1f}, {:.1f}]".format(yedges[i],yedges[i+1]))
    plt.plot(xticks, effston, label='all')
    plt.ylim([0,1.05])
    plt.xlabel('STON')
    plt.ylabel('eff')
    plt.title('Host Surface Brightness')
    plt.legend(loc=4)
    pdf.savefig()
    plt.close()
