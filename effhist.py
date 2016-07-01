#!/usr/bin/env python
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle


fracincrease_min = 0.1
fracincrease_max = 10.
stonmin=4.
stonmax=10.5
npar=7

def effmodel(ston, frac, pars, minlnFI, maxlnFI):
    logfrac = numpy.log(frac)
    logfrac = (logfrac-minlnFI)/(maxlnFI-minlnFI)
    if logfrac < 0:
        logfrac = 0
    elif logfrac>1:
        logfrac=1
    pterm = (1-logfrac)**pars[1]
    aterm = 1+(pars[0]-1)*pterm
    bterm = pars[2]+(pars[3]-pars[2])*pterm
    cterm = pars[4]+(pars[5]-pars[4])*pterm

    dum = numpy.zeros(len(ston))
    ok = (ston-bterm) >0
    argterm = (ston[ok]-bterm)/cterm
    tanhterm = numpy.tanh(argterm)**pars[6]
    dum[ok] = aterm *tanhterm
    return dum

def loadpars(emcee=True):
    if not emcee:
        f = open('stan2.pkl','rb')
        fit,minlogFI, maxlogFI = pickle.load(f)
        pars = numpy.append(numpy.mean(fit['a1']), numpy.mean(fit['a2']))
        pars = numpy.append(pars, numpy.mean(fit['b'],axis=0))
        pars = numpy.append(pars,  numpy.mean(fit['c'],axis=0))
        # pars = numpy.append(pars, numpy.mean(fit['pop']))
    else:
        f = open('emcee.pkl','rb')
        fit,minlogFI, maxlogFI = pickle.load(f)
        f.close()
        fit = fit[:,600:,:].reshape((-1,npar))
        pars = numpy.mean(fit,axis=0)
    print pars
    return pars, fit, minlogFI, maxlogFI

    # n=3
    # pars = numpy.append(numpy.mean(fit['a'][n+0:n+100,:],axis=0), numpy.mean(fit['b'][n+0:n+100,:],axis=0))
    # pars = numpy.append(pars,  numpy.mean(fit['c'][n+0:n+100,:],axis=0))
    # pars = numpy.append(pars, numpy.mean(fit['pop'][n+0:n+100])) 

def surfaceplot():
    from matplotlib import colors, ticker, cm
    from matplotlib.ticker import FormatStrFormatter

    pars, fit, minlnFI, maxlnFI =loadpars()

    frac = numpy.arange(fracincrease_min, fracincrease_max, 0.02)
    ston = numpy.arange(4,10, 0.02)

    ans = numpy.zeros((len(frac),len(ston)))
    for i in xrange(len(frac)):
        ans[i,:] = effmodel(ston, frac[i], pars, minlnFI, maxlnFI)

    totaleff=1-(1-ans)**3

    X, Y = numpy.meshgrid(ston, frac)
    fig1, ax1 = plt.subplots()
    # cs = ax1.imshow(X, Y, totaleff,norm=colors.LogNorm(vmin=0.9, vmax=1),interpolation='nearest')
    cs = ax1.pcolor(X, Y, totaleff,norm=colors.LogNorm(vmin=0.9, vmax=1),rasterized=True)
    ax1.set_xlabel('STON')
    ax1.set_ylabel('Fraction Increase')
    ax1.set_yscale('log')
    ax1.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax1.set_ylim((0.3,10))
    ax1.set_xscale('log')
    ax1.set_xlim((4,10))
    ax1.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax1.set_title('Discovery Efficiency Given Three Opportunities')
    cbar = fig1.colorbar(cs, ax=ax1)
    cbar.set_ticks(numpy.arange(0.9,1.01,.01),update_ticks=True)
    cbar.set_ticklabels(numpy.arange(0.9,1.01,.01),update_ticks=True)
    cbar.set_label('Discovery Efficiency')
    with PdfPages('efficiency_3.pdf') as pdf:
        pdf.savefig()
        plt.close()


def readdataforfit():
    import pickle
    input = open('data.pkl', 'rb')
    [ston, fracincrease, found]=pickle.load(input)
    input.close()

    # prune sample
    use = numpy.logical_and(fracincrease > fracincrease_min, fracincrease <fracincrease_max)
    use = numpy.logical_and(use, ston > stonmin)
    use = numpy.logical_and(use, ston < stonmax)
    # use = numpy.logical_and(use, numpy.logical_or(ston < 10, ston > 16))
    ston = ston[use]
    fracincrease = fracincrease[use]
    found = found[use]

    lnFI = numpy.log(fracincrease)
    minlogFI = lnFI.min()
    maxlogFI = lnFI.max()
    lnFI = (lnFI-minlogFI)/(maxlogFI-minlogFI)

    return ston, found ,lnFI, minlogFI, maxlogFI

def fit():
    import pystan


    ston, found, lnFI, minlogFI, maxlogFI = readdataforfit()

    # input = open('data.pkl', 'rb')
    # [ston, fracincrease, found]=pickle.load(input)
    # input.close()

    # # prune sample
    # use = numpy.logical_and(fracincrease > 0.01, fracincrease <10)
    # ston = ston[use]
    # fracincrease = fracincrease[use]
    # found = found[use]

    # lnFI = numpy.log(fracincrease)
    # minlogFI = lnFI.min()
    # maxlogFI = lnFI.max()
    # lnFI = (lnFI-minlogFI)/(maxlogFI-minlogFI)

    data = {'D': len(ston), 'ston': ston, 'lnFI': lnFI, 'found': found.astype('int')}
    sm = pystan.StanModel(file='effmodel.stan')
    # indict = {'a': [0.996,.999999], 'b': [.1, 4], 'c': [2, 1.],'pop':5.}
    indict = {'a1': 0.6 , 'a2': 4.5, 'b': [3.3, 2.5], 'c': [1.2, 3.5],'pop':4.5}
    fit = sm.sampling(data=data, iter=500, chains=4, init=[indict, indict, indict, indict])

    output = open('stan2.pkl','wb')
    pickle.dump([fit.extract(),minlogFI, maxlogFI], output)
    output.close()


def fit_emcee():

    import sys
    import emcee

    nwalkers=npar*2


    ston, found, lnFI, minlogFI, maxlogFI = readdataforfit()

    xedges = stonmin * (2**numpy.arange(0,2.5,.25)) #numpy.arange(2,20.5,1)
    yedges =  fracincrease_min*(2**numpy.arange(8)) #(4**numpy.arange(6))/64. #numpy.arange(0,210,50)
    fracincrease  = numpy.exp(lnFI*(maxlogFI-minlogFI) + minlogFI)
    H, _,_ = numpy.histogram2d(ston, fracincrease, bins=[xedges, yedges])

    xbin = numpy.digitize(ston,xedges)
    ybin = numpy.digitize(fracincrease,yedges)
    xbin[xbin == xedges.shape] = xedges.shape[0]-1
    ybin[ybin == yedges.shape] = yedges.shape[0]-1

    invH = [H[i-1,j-1] for i,j in zip(xbin,ybin)]
    invH = numpy.array(invH)
    invH= 1.*len(invH)/invH

    # input = open('data.pkl', 'rb')
    # [ston, fracincrease, found]=pickle.load(input)
    # input.close()

    # # prune sample
    # use = numpy.logical_and(fracincrease > 0.01, fracincrease <10)
    # ston = ston[use]
    # fracincrease = fracincrease[use]
    # found = found[use]

    # lnFI = numpy.log(fracincrease)
    # minlogFI = lnFI.min()
    # maxlogFI = lnFI.max()
    # lnFI = (lnFI-minlogFI)/(maxlogFI-minlogFI)

    foundpm = (-1)**found

    def lnprob(p, oneminuslnFI, found, foundpm, invH):
        if (p[0]<0.98 or p[0]>1 or p[1]<1 or  p[2]<=1 or p[3]<=1  or p[4]<=0 or p[5]<=0 or p[6] < 1):
            return -numpy.inf

        pterm = oneminuslnFI**p[1];

        t = (ston-(p[2] + (p[3]-p[2])* pterm)) / (p[4] + (p[5]-p[4])* pterm)
        eff=numpy.zeros(len(t))
        ok = t>0
        eff[ok] = numpy.tanh(t[ok]) **p[6]
        eff[ok] =  (1+(p[0]-1)*pterm[ok])  *eff[ok]
        # eff[t<=0] = 0.

        ans= (1-found) -foundpm*eff
        ans[ans<=0] = 1e-4
        return (invH*numpy.log(ans)).sum()
        # return numpy.log(ans).sum()
    p0 = []
        #         pars = numpy.array([ .987 , 1.5 , 4. ,  4. , .8 , 1.3 , 1])

    avg = numpy.array([0.985,1.3,3.8,3.8,0.9,1.3, 1.3])
    for i in range(nwalkers):
        p0.append(avg+numpy.random.uniform(-.5,.5,size=npar)*numpy.array([0.005,.1,.1,.1,.1,.1,.2]))

    sampler = emcee.EnsembleSampler(nwalkers, npar, lnprob, args=(1-lnFI, found, foundpm, invH))
    sampler.run_mcmc(p0, 50000, thin=50)

    output = open('emcee.pkl','wb')
    chain = numpy.array(sampler.chain)

    pickle.dump([chain,minlogFI, maxlogFI], output)
    output.close()


def savedata():
    import glob
    zp=31.4

    dirname = '/Users/akim/Downloads/DESY1_forcePhoto_fake_snana_text/des_fake_*.dat'
    indeces = [2, 4, 5, 6, 7, 9, 16]
    filts=numpy.array(['g','r','i','z'])
    # VARLIST: MJD  FLT  FIELD FLUXCAL  FLUXCALERR PHOTFLAG PHOTPROB ZPFLUX PSF SKYSIG 
    # SKYSIG_T GAIN XPIX YPIX SIM_MAGOBS MASKFRAC   NITE 

    data = []
    for fn in glob.glob(dirname):
        # print fn
        with open(fn) as f:
            # f.readline()
            # varnames= f.readline().split()
            # varnames =  numpy.array(varnames)
            # print varnames[indeces]
            for line in f:
                if line.strip():
                    values = line.split()
                    values = numpy.array(values)
                    if values[0] == 'HOSTGAL_SB_FLUXCAL:':
                        galsb=values[1:5]
                        galsb=galsb.astype('float')
                        galsbneg = numpy.logical_and(galsb >-800,galsb<=0)
                        galsb[galsbneg]=1e-12
                        galsbpos = galsb>0
                        galsb[galsbpos] = -2.5*numpy.log10(galsb[galsbpos]) + zp
                        # galsb = -2.5*numpy.log10(galsb)+zp
                        # print galsb
                    if values[0] == 'OBS:':
                        mfrac = float(values[indeces[6]])
                        emask = int(values[indeces[3]])
                        gsb = galsb[filts == values[indeces[0]]][0]
                        if mfrac >= 0 and mfrac <= 0.1 and (emask & 240)==0 and gsb > -800:
                            temp = [values[indeces[0]],values[indeces[5]],gsb, values[indeces[1]],values[indeces[2]],values[indeces[3]],values[indeces[4]],values[indeces[6]]]
                            data.append(numpy.array(temp))

    # # data = []
    filename = 'Y1reprocFakes/FAKEMATCH_ALL.OUT'
    indeces = [3, 18, 31, 50, 51, 52, 53, 24]
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
    import pickle
    output = open('data.pkl', 'wb')
    pickle.dump([ston, fracincrease, found],output)
    output.close()

def plotdata(ston, fracincrease, found):

    lnFI = numpy.log(fracincrease)
    # lnFImin=lnFI.min()
    # lnFImax=lnFI.max()

    with PdfPages('multipage_pdf.pdf') as pdf:
        xedges = numpy.arange(2,20.5,1)
        yedges =  0.1*(2**numpy.arange(8)) #(2**numpy.arange(6))/8. #numpy.arange(0,210,50)

        H, _, _ = numpy.histogram2d(ston, fracincrease, bins=[xedges, yedges])
        xticks = (0.5*(xedges+numpy.roll(xedges,-1)))[:-1]
        yticks = (0.5*(yedges+numpy.roll(yedges,-1)))[:-1]

        # plt.hist2d(ston, numpy.log(fracincrease), bins=[xedges, numpy.log(yedges)])
        # plt.colorbar()
        # plt.title("Number of Fakes Per Bin")
        # plt.xlabel('STON')
        # plt.ylabel('log(Fraction Flux Increase)')
        # pdf.savefig()
        # plt.close()

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

        pars, fit, minlnFI, maxlnFI = loadpars()
        # pars[0]=pars[0]/1.4
        # pars[1]=pars[1]*1.2
        # print pars
        # pars = numpy.array([ .987 , 1.5 , 4. ,  4. , .8 , 1.3 , 1])
        # pars[0]=0.985
        # pars[2]=3.5
        # pars[3]=3.5
        # pars[4]=1.4
        # pars[5]=0.8

        for i in xrange(len(yedges)-1):
            plt.plot(xticks, eff[:,i], label="Frac. Inc. [{:.2f}, {:.2f}]".format(yedges[i],yedges[i+1]))
            # dum=(pars[0]+(pars[1]-pars[0])*logyticks[i])*numpy.tanh((xticks-(pars[2]+(pars[3]-pars[2])*logyticks[i]))/(pars[4]+(pars[5]-pars[4])*logyticks[i]))

            dum = effmodel(xticks, yticks[i], pars, minlnFI, maxlnFI)
            plt.plot(xticks, dum, linestyle=':',linewidth=3)
        # plt.plot(xticks, effston, label='all')
        plt.xlabel('STON')
        plt.ylabel('eff')
        # plt.yscale('log')
        # plt.title('Fraction Increase in Seeing Disk')
        plt.ylim([0.1,1.05])
        plt.legend(loc=4)
        plt.show()

        # pdf.savefig()
        # plt.close()

        # xedges = numpy.arange(2,20.5,1)
        # yedges = numpy.arange(17,28,2)

        # H, xedges, yedges = numpy.histogram2d(ston, galsb, bins=[xedges, yedges])

        # eff = numpy.array(H)
        # effston = numpy.zeros(len(xedges)-1)
        # eff[:]=0.
        # for i in xrange(len(xedges)-1):
        #     inx = numpy.logical_and(ston > xedges[i], ston < xedges[i+1])
        #     if inx.sum() !=0:
        #         effston[i] = 1.*found[inx].sum() / inx.sum()
        #     for j in xrange(len(yedges)-1):
        #         iny = numpy.logical_and(galsb > yedges[j], galsb <= yedges[j+1])
        #         if H[i,j] !=0:
        #             inxy = numpy.logical_and(inx, iny)
        #             eff[i,j] = found[inxy].sum()/H[i,j]

        # xticks = (0.5*(xedges+numpy.roll(xedges,-1)))[:-1]

        # for i in xrange(len(yedges)-1):
        #     plt.plot(xticks, eff[:,i], label="[{:.1f}, {:.1f}]".format(yedges[i],yedges[i+1]))
        # plt.plot(xticks, effston, label='all')
        # plt.ylim([0,1.05])
        # plt.xlabel('STON')
        # plt.ylabel('eff')
        # plt.title('Host Surface Brightness')
        # plt.legend(loc=4)
        # pdf.savefig()
        # plt.close()

fit_emcee()
# surfaceplot()
# fit()
# wefwe
# lookatfit()



from matplotlib import colors, ticker, cm
from matplotlib.ticker import FormatStrFormatter
input = open('data.pkl', 'rb')
[ston, fracincrease, found]=pickle.load(input)
input.close()

use = numpy.logical_and(fracincrease > 0.1, fracincrease < 10)
use = numpy.logical_and(use, ston > 4)
ston = ston[use]
fracincrease = fracincrease[use]
found = found[use]
plotdata(ston, fracincrease, found)

# xedges = stonmin * (2**numpy.arange(0,2,.05)) #numpy.arange(2,20.5,1)
# yedges =  fracincrease_min*(2**numpy.arange(0,8,.1)) #(4**numpy.arange(6))/64. #numpy.arange(0,210,50)
# fracincrease  = numpy.exp(lnFI*(maxlogFI-minlogFI) + minlogFI)



# H, _,_ = numpy.histogram2d(ston, fracincrease, bins=[xedges, yedges])


# xbin = numpy.digitize(ston,xedges)
# ybin = numpy.digitize(fracincrease,yedges)

# eff = numpy.array(H)
# eff[:,:]=0.

# for i in xrange(len(found)):
#     eff[xbin[i]-1,ybin[i]-1] += found[i]

# eff /= H

# xticks = (0.5*(xedges+numpy.roll(xedges,-1)))[:-1]
# yticks = (0.5*(yedges+numpy.roll(yedges,-1)))[:-1]

# X, Y = numpy.meshgrid(yticks, xticks)

# fig1, ax1 = plt.subplots()
# # cs = ax1.imshow(X, Y, totaleff,norm=colors.LogNorm(vmin=0.9, vmax=1),interpolation='nearest')
# cs = ax1.pcolor(X, Y, eff,norm=colors.LogNorm(vmin=0.9, vmax=1),rasterized=True)
# ax1.set_xlabel('STON')
# ax1.set_ylabel('Fraction Increase')
# ax1.set_yscale('log')
# ax1.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
# ax1.set_ylim((0.3,10))
# ax1.set_xscale('log')
# ax1.set_xlim((4,10))
# ax1.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
# ax1.set_title('Discovery Efficiency Given Three Opportunities')
# cbar = fig1.colorbar(cs, ax=ax1)
# cbar.set_ticks(numpy.arange(0.9,1.01,.01),update_ticks=True)
# cbar.set_ticklabels(numpy.arange(0.9,1.01,.01),update_ticks=True)
# cbar.set_label('Discovery Efficiency')

# plt.show()


