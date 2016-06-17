#!/usr/bin/env python
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle


def modelplots():
    import scipy.special
    x = numpy.arange(0,5,.01)
    plt.plot(x,scipy.special.erf(x),label='erf')
    plt.plot(x,numpy.tanh(1.5*x),label='tanh')
    plt.legend()
    plt.show()



def fit():
    import pystan

    # import sys
    # import emcee


    # nwalkers=10
    # npar=7

    input = open('data.pkl', 'rb')
    [ston, fracincrease, found]=pickle.load(input)
    input.close()

    # prune sample
    use = numpy.logical_and(fracincrease > 0.01, fracincrease <10)
    ston = ston[use]
    fracincrease = fracincrease[use]
    found = found[use]

    # lnFI = numpy.log(fracincrease)
    # lnFImin = lnFI.min()
    # lnFImax = lnFI.max()
    # lnFI = (lnFI-lnFImin)/(lnFImax-lnFImin)

    # foundpm = (-1)**found

    # def lnprob(p, lnFI, found, foundpm):
    #     p0 = p[0]/10.
    #     p1= p[1]/100.
    #     if (p0<0 or  p0>1 or  p1>1 or p1<0  or p[4]<=0 or p[5]<=0 or p[6]<1 or p[2]<=0 or p[3]<=0 or p[2]>6 or p[3]>6):
    #         return -numpy.inf
    #     eff = (ston-(p[2] + (p[3]-p[2])* lnFI) / (p[4] + (p[5]-p[4])* lnFI))
    #     eff = numpy.tanh(eff)
    #     eff = (p0 + (p1-p0)* lnFI) *eff
    #     eff[eff<=0] = 0.
    #     eff=eff**p[6]
    #     ans= (1-found) -foundpm*eff
    #     ans[ans <=0] = sys.float_info.min
    #     return numpy.log(ans).sum()

    # p0 = []
    # avg = numpy.array([5,99,2.5,2.5,2,2,4])
    # for i in range(nwalkers):
    #     p0.append(avg+numpy.random.uniform(-1,1,size=npar)*numpy.array([0.5,0.2,1,1,1,1,1]))

    # sampler = emcee.EnsembleSampler(nwalkers, npar, lnprob, args=(lnFI, found, foundpm), a=0.5)
    # sampler.run_mcmc(p0, 1000)

    # output = open('emcee.pkl','wb')
    # chain = numpy.array(sampler.chain)
    # chain[:,:,0]=chain[:,:,0]/10.
    # chain[:,:,1]=chain[:,:,1]/100.

    # pickle.dump([chain,lnFImin,lnFImax], output)
    # output.close()

    data = {'D': len(ston), 'ston': ston, 'fracincrease': fracincrease, 'found': found.astype('int')}
    sm = pystan.StanModel(file='effmodel.stan')
    # indict = {'a': [0.996,.999999], 'b': [.1, 4], 'c': [2, 1.],'pop':5.}
    indict = {'a1': 0.5659394 , 'a2': 3, 'b': [3., 3.3], 'c': [1, 5.],'pop':2.}
    fit = sm.sampling(data=data, iter=400, chains=4, init=[indict, indict, indict, indict])
    output = open('stan.pkl','wb')
    pickle.dump(fit.extract(), output)
    output.close()



def lookatfit():
    import corner
    import pickle
    f = open('stan.pkl','rb')
    fit = pickle.load(f)
    pars = numpy.append(numpy.mean(fit['a1']), numpy.mean(fit['a2']))
    pars = numpy.append(pars, numpy.mean(fit['b'],axis=0))
    pars = numpy.append(pars,  numpy.mean(fit['c'],axis=0))
    pars = numpy.append(pars, numpy.mean(fit['pop']))


    plt.plot(fit[0,:,:])
    samples = fit.reshape((-1, 7))
    plt.plot(samples)
    samples = fit[:, 200:, :].reshape((-1, 7))
    print numpy.mean(samples,axis=0)
    with PdfPages('corner.pdf') as pdf:
        figure = corner.corner(fit)
        pdf.savefig()
        plt.close()




def savedata():
    import glob
    zp=31.4

    # dirname = '/Users/akim/Downloads/DESY1_forcePhoto_fake_snana_text/des_fake_002459*.dat'
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
    lnFImin=lnFI.min()
    lnFImax=lnFI.max()

    with PdfPages('multipage_pdf.pdf') as pdf:

        xedges = numpy.arange(2,20.5,1)
        yedges = (4**numpy.arange(6))/64. #numpy.arange(0,210,50)

        H, xedges, yedges = numpy.histogram2d(ston, fracincrease, bins=[xedges, yedges])
        xticks = (0.5*(xedges+numpy.roll(xedges,-1)))[:-1]
        yticks = (0.5*(yedges+numpy.roll(yedges,-1)))[:-1]
        logyticks = numpy.log(yticks)

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


        f = open('stan.pkl','rb')
        fit = pickle.load(f)
        pars = numpy.append(numpy.mean(fit['a1']), numpy.mean(fit['a2']))
        pars = numpy.append(pars, numpy.mean(fit['b'],axis=0))
        pars = numpy.append(pars,  numpy.mean(fit['c'],axis=0))
        pars = numpy.append(pars, numpy.mean(fit['pop']))

        # n=3
        # pars = numpy.append(numpy.mean(fit['a'][n+0:n+100,:],axis=0), numpy.mean(fit['b'][n+0:n+100,:],axis=0))
        # pars = numpy.append(pars,  numpy.mean(fit['c'][n+0:n+100,:],axis=0))
        # pars = numpy.append(pars, numpy.mean(fit['pop'][n+0:n+100]))

        print pars
        logyticks = (logyticks-lnFImin)/(lnFImax-lnFImin)
        # pars=numpy.array([.55,5.,3.4,3.4,2,.8,2.5])
        # pars = numpy.array([ 0.48659394 , 4.84466494 , 2.6719694 ,  2.96490356 , 1.77596779 , 0.90177606 , 4.83420871])
        pars = numpy.array([ .3 , 4. , 3. ,  3.3 , 1. , 5 , .5])
        for i in xrange(len(yedges)-1):
            plt.plot(xticks, eff[:,i], label="[{:.2f}, {:.2f}]".format(yedges[i],yedges[i+1]))
            # dum=(pars[0]+(pars[1]-pars[0])*logyticks[i])*numpy.tanh((xticks-(pars[2]+(pars[3]-pars[2])*logyticks[i]))/(pars[4]+(pars[5]-pars[4])*logyticks[i]))

            dum = numpy.zeros(len(xticks))

            aterm = pars[0]*(1-logyticks[i])**pars[1]
            bterm = pars[2]+(pars[3]-pars[2])*logyticks[i]
            cterm = pars[4]+(pars[5]-pars[4])*(1-logyticks[i])**4

            normterm = numpy.exp(-aterm)
            ok = (xticks-bterm) >0
            argterm = (xticks[ok]-bterm)/cterm
            tanhterm = numpy.tanh(argterm)**pars[6]
            dum[ok] = normterm *tanhterm
            # dum=numpy.exp(-pars[0]*(1-logyticks[i])**pars[1])*numpy.tanh((xticks-(pars[2]+(pars[3]-pars[2])*logyticks[i]))/(pars[4]+(pars[5]-pars[4])*logyticks[i]))**pars[6]
            # dum[xticks-pars[2]<0]=0
            # print  logyticks[i], (dum**(numpy.exp(pars[6]*(1-logyticks[i])))).max()
            # plt.plot(xticks, dum**(numpy.exp(pars[6]*(1-logyticks[i]))), linestyle=':')
            plt.plot(xticks, dum, linestyle=':')
        plt.plot(xticks, effston, label='all')
        plt.xlabel('STON')
        plt.ylabel('eff')
        plt.yscale('log')
        plt.title('Fraction Increase in Seeing Disk')
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

fit()
# wefwe
# lookatfit()

#wefwe

input = open('data.pkl', 'rb')
[ston, fracincrease, found]=pickle.load(input)
input.close()
use = numpy.logical_and(fracincrease > 0.01, fracincrease <16)
ston = ston[use]
fracincrease = fracincrease[use]
found = found[use]
plotdata(ston, fracincrease, found)


