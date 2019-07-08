from pyzota_image_toolbox import imageTools as IT
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import stats
import matplotlib
from matplotlib import rc,rcParams
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
rc('axes', linewidth=2)
rc('font', weight='bold')
def getCondition(fName):
    array = fName.split(".")[0].split("_")[0:2]
    return array[0]+array[1]


dataFolder = "./Analysis/OsmoComparison/"
#dataFolder = "./Analysis/SaltOnly/"
fileNames = sorted(IT.GetFileNamesFromFolder(dataFolder))
fileNamesPlusPath = sorted(IT.GetFileNamesFromFolder(dataFolder,False))
areaCutOff = 10


numberOfFiles = len(fileNames)
print("Number of files = {}".format(numberOfFiles))
conditions =sorted(list(set([ getCondition(i) for i in fileNames ])))
print("Conditions={}".format(conditions))



array = [[] for i in range(len(conditions))]
#Conditions grouped together
dataDic = {"Names":copy.deepcopy(array),"Areas":copy.deepcopy(array),
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array),"Volumes":copy.deepcopy(array), "Intensities":copy.deepcopy(array)}

array = [[] for i in range(numberOfFiles)]
#Each slide individually
dataDic2 = {"Names":copy.deepcopy(array),"Areas":copy.deepcopy(array),
        "Widths":copy.deepcopy(array),"Lengths":copy.deepcopy(array),"Volumes":copy.deepcopy(array), "Intensities":copy.deepcopy(array)}



for n,fName in enumerate(fileNamesPlusPath):
    f = open(fName,'r')
    dataDic["Names"].append(fName)
    dataDic2["Names"][n]=fileNames[n].split(".")[0]
    endOfFile = False
    while not endOfFile:
        try:
            d = pickle.load(f)
            mask = d["BinaryMask"]
            if mask.size > 10:
                mask = IT.ClearBorders(mask)
                area = IT.GetArea(mask)
                if area > areaCutOff:
                    length,width = IT.GetLengthAndWidth(mask,d["RawImage"])
                    meanIntensity = IT.getMeanIntensity(mask,d["RawImage"])
                    volume = ((4/3.0)*np.pi*((width/2.0)**3)) + (np.pi*((width/2)**2)*(length-width))
                    condition = getCondition(fileNames[n])
                    index = conditions.index(condition)
                    dataDic["Areas"][index].append(area)
                    dataDic["Lengths"][index].append(length)
                    dataDic["Widths"][index].append(width)
                    dataDic["Volumes"][index].append(volume)
                    dataDic["Intensities"][index].append(meanIntensity)

                    dataDic2["Areas"][n].append(area)
                    dataDic2["Lengths"][n].append(length)
                    dataDic2["Widths"][n].append(width)
                    dataDic2["Volumes"][n].append(volume)
                    dataDic2["Intensities"][index].append(meanIntensity)
        except EOFError:
            print("End of File")
            endOfFile = True

#****************************************** PLOTTING ***************************



#****************************************** PLOTTING ***************************

plt.ioff()
plt.clf()
#Plot Histograms
Props = ["Areas","Lengths","Widths","Volumes"]
#histBins = [np.linspace(0,2000,50),np.linspace(0,100,50),np.linspace(0,25,50),np.linspace(0,4000,50)]
#histArray = []
#for z,prop in enumerate(Props):
#    print(len(dataDic[prop]))
#    plt.hist([dataDic[prop][1],dataDic[prop][5]],histtype= "step",bins=histBins[z],stacked=False,label=[conditions[1],conditions[5]],alpha=0.9,normed=False)
#    plt.legend()
#    plt.title(prop)
#    plt.xlabel(prop)
#    plt.ylabel("Frequency")
#    plt.savefig("./Analysis/Graphs/"+prop+"Hist")
#    plt.show()

plt.close()

#Plot Box Plots
for prop in Props:
    data = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n in range(len(conditions)):
        data.append(np.asarray(dataDic[prop][n])*0.051)
        ax.annotate('n={}'.format(len(data[n])), xy=(n+1, np.median(data[n])),
                xytext=(n+1, np.median(data[n])), ha='center')
    ax.boxplot(data)

    ax.set_xticklabels(conditions)
    plt.title(prop)
    ax.grid(linestyle='--', linewidth=1,axis="y")
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.savefig("./Analysis/Graphs/"+prop)
    #plt.show()

#Plot Box Plots each slide individually
for prop in Props:
    data = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n in range(len(dataDic2["Names"])):
        data.append(np.asarray(dataDic2[prop][n])*0.05)
        ax.annotate('n={}'.format(len(data[n])), xy=(n+1, np.median(data[n])),
                xytext=(n+1, np.median(data[n])), ha='center')
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    ax.boxplot(data)
    ax.set_xticklabels(dataDic2["Names"])
    plt.title(prop)
    ax.grid(linestyle='--', linewidth=1,axis="y")
    plt.savefig("./Analysis/Graphs/All_"+prop)
    #plt.show()

from matplotlib.patches import Rectangle

#******** THESIS FIGURES *************
prop = "Volumes"
meanVolumes = []
growthRates = [40,45.67,48.9,57.133,80]
#growthRates = [40,45.67,57.133,80]
scaleFactors = 0.051
volumes = []
yErrors = []
for n in range(len(conditions)):
    volume = np.asarray(dataDic[prop][n])*((scaleFactors)**3)
    volumes.append(volume)
    meanVolume = np.mean(volume)
    meanVolumes.append(meanVolume)
    yErr = np.std(volume)
    yErr = yErr/meanVolume
    yErrors.append((yErr/np.sqrt(len(volume))))


#************ GROWTH LAW **************
colors = ['C0',"C1","C2","C3","C4"]
plt.close("all")
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(13,9))
ax[0][0].text(0.05, 0.85 , "A", transform=ax[0][0].transAxes, size=30, weight='bold')
ax[0][1].text(0.90, 0.85 , "B", transform=ax[0][1].transAxes, size=30, weight='bold')
ax[1][0].text(0.05, 0.85 , "C", transform=ax[1][0].transAxes, size=30, weight='bold')
ax[1][1].text(0.05, 0.85 , "D", transform=ax[1][1].transAxes, size=30, weight='bold')

#Bottom left size vs growth rate
x = np.log(2)/(np.asarray(growthRates)/60.0)
y = np.log(np.asarray(meanVolumes))
#From our growth law

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

yPredict = x*slope + intercept
xLine = np.linspace(0,1.5,10)
yLine = xLine*slope + intercept
ax[1][0].plot(xLine,yLine,color="navy",linewidth=3,label="$S = {0:.1f}e^{{ {1:.1f}\\lambda}}$".format(np.exp(intercept),slope))
slope = 1.29
intercept = -1.45
xTheory = np.linspace(0,1.5,10)
yTheory = xTheory*slope+intercept
ax[1][0].errorbar(x,y,yerr=yErrors,fmt="none",markersize=10,capsize=10,elinewidth=3,ecolor="k")
ax[1][0].plot(xTheory,yTheory,color="C9", label="$S = 0.23e^{1.29 \\lambda}$",linewidth=3)
nutrientX = [x[0], 0.46209812, 1.48531539]
nutrientY = [y[0], -0.85105785,  0.46660303]
nutrientErr = [0.019592024593452443, 0.028462536329445776, 0.021280029968116877]
ax[1][0].errorbar(nutrientX,nutrientY,yerr=nutrientErr,fmt="none",markersize=10,capsize=10,elinewidth=3,ecolor="k")
ax[1][0].scatter(nutrientX,nutrientY,s=90,c="C6",zorder=99,alpha=1.0,marker="D",label="Nutrient limitation")
ax[1][0].scatter(x,y,s=90,c=colors,zorder=100,alpha=1.0,marker="o",facecolors='none')
ax[1][0].legend(loc="lower right",fontsize="x-large")
ax[1][0].set_xlabel("Growth rate $\mathbf{\lambda = \\frac{ln(2)}{\\tau}}$ ($h^{-1}$)",fontsize=20,fontweight="bold")
ax[1][0].set_ylabel("$\mathbf{ln}$(Mean cell volume)",fontsize=20,fontweight="bold")
ax[1][0].set_xlim(0,1.6)
ax[1][0].set_ylim(-1.5,0.6)
ax[1][0].tick_params(axis="x", labelsize=15)
ax[1][0].tick_params(axis="y", labelsize=15)
#Bottom right residuals
ax[1][1].axhline(0,color="gray",zorder=0)
ax[1][1].errorbar(x,y-yPredict,yerr=yErrors,fmt='none',markersize=10,capsize=10,elinewidth=3,ecolor="k")
ax[1][1].scatter(x,y-yPredict,s=90,c=colors,zorder=99,alpha=1.0,marker="o",facecolor='none')
ax[1][1].set_xlabel("Growth rate $\mathbf{\lambda}$ ($h^{-1}$)",fontsize=20,fontweight="bold")
ax[1][1].set_ylabel("Residuals",fontsize=20,fontweight="bold")
ax[1][1].tick_params(axis="x", labelsize=15)
ax[1][1].tick_params(axis="y", labelsize=15)
#Top left plot
cons = ["+0mM NaCl", "+200mM NaCl", "+300mM Sorbitol","+400mM NaCl", "+600mM NaCl"]
#cons = ["+0mM NaCl", "+200mM NaCl","+400mM NaCl", "+600mM NaCl"]
labels = []
for i in range(len(cons)):
    labels.append(cons[i] + ", N= {}".format(len(volumes[i])))
#for i,vol in enumerate(volumes):
#    ax[0][0].hist(vol,bins=18,alpha=0.33,density=True,label=labels[i],histtype="bar")
ax[0][0].hist(volumes,bins=30,alpha=0.33,density=True,label=labels,histtype="barstacked")
ax[0][0].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[0][0].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][0].tick_params(axis="x", labelsize=15)
ax[0][0].tick_params(axis="y", labelsize=15)
#ax[0][0].legend(fontsize="x-large")
#Top right

box = ax[0][1].boxplot(volumes,vert=False,whis=[1,99], positions= growthRates,widths=3 , patch_artist=True )
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.33)

ax[0][1].set_ylim(35,85)
ax[0][1].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[0][1].set_ylabel("Doubling time $\\tau$ (min)",fontsize=20,fontweight="bold")
ax[0][1].set_yticks([35,45,55,65,75,85])
ax[0][1].set_yticklabels([35,45,55,65,75,85])
legendItems = []
for color in colors:
    legendItems.append(Rectangle((0, 0), 1, 1, fc=color,alpha=0.33, fill=True, edgecolor='none', linewidth=0))
for i in range(len(volumes)):
    ax[0][1].plot(np.mean(volumes[i]),growthRates[i],'o',alpha=0.5,color=colors[i],markersize=7)
#ax[0][1].legend(legendItems, labels,fontsize="large")
#fig.legend(legendItems, labels,fontsize="xx-large",ncol=3,loc="upper center")#loc = (0.05, 0.97))
fig.legend(legendItems, labels,loc='upper center', bbox_to_anchor=(0.5, 1.08), fancybox=False, shadow=False, ncol=3,fontsize="xx-large")
ax[0][1].tick_params(axis="x", labelsize=15)
ax[0][1].tick_params(axis="y", labelsize=15)
ax[0][1].grid(linestyle='--', linewidth=1,axis="x")
fig.tight_layout()
fig.savefig('ThesisGraphs/GrowthLaw/OsmoComparison.pdf', bbox_inches='tight')
fig.savefig('ThesisGraphs/GrowthLaw/OsmoComparison.png', bbox_inches='tight')
#plt.show()
plt.clf()
plt.close("all")
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************



#**************** WIDTH AND LENGTH COMAPRISONS ******************************#
lengths = []
meanLengths  = []
widths = []
meanWidths = []
surfaceAreas = []
meanSurfaceAreas = []
growthRates = [40,45.67,48.9,57.133,80]
#growthRates = [40,45.67,57.133,80]
intensities = []
meanIntensities = []
scaleFactors = 0.051
for n in range(len(conditions)):
    length = np.asarray(dataDic["Lengths"][n])*((scaleFactors))
    lengths.append(length)
    meanLengths.append(np.mean(length))
    width = np.asarray(dataDic["Widths"][n])*((scaleFactors))
    widths.append(width)
    surfaceArea = ((width**2)*np.pi)  + ((length-width)*width)
    surfaceAreas.append(surfaceArea)
    meanSurfaceAreas.append(np.mean(surfaceArea))
    meanWidths.append(np.mean(width))
    intensity = np.asarray(dataDic["Intensities"][n])
    intensities.append(intensity)
    meanIntensities.append(np.mean(intensity))

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(13,9))
ax[0][0].text(0.05, 0.85 , "A", transform=ax[0][0].transAxes, size=30, weight='bold')
ax[0][1].text(0.05, 0.85 , "B", transform=ax[0][1].transAxes, size=30, weight='bold')
ax[1][0].text(0.05, 0.85 , "C", transform=ax[1][0].transAxes, size=30, weight='bold')
ax[1][1].text(0.05, 0.85 , "D", transform=ax[1][1].transAxes, size=30, weight='bold')


box = ax[1][0].boxplot(lengths,vert=False,whis=[1,99], positions= growthRates,widths=3 , patch_artist=True )
ax[1][0].grid(linestyle='--', linewidth=1,axis="x")
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.33)
for i in range(len(lengths)):
    ax[1][0].plot(np.mean(lengths[i]),growthRates[i],'o',alpha=0.5,color=colors[i],markersize=7)
ax[1][0].set_ylim(35,85)
ax[1][0].set_xlim(0,5)
ax[1][0].set_xlabel("Length($\\mu m$)",fontsize=20,fontweight="bold")
ax[1][0].set_ylabel("Doubling time $\\tau$ (min)",fontsize=20,fontweight="bold")
ax[1][0].set_yticks([35,45,55,65,75,85])
ax[1][0].set_yticklabels([35,45,55,65,75,85])
ax[1][0].tick_params(axis="x", labelsize=15)
ax[1][0].tick_params(axis="y", labelsize=15)


box = ax[1][1].boxplot(widths,vert=False,whis=[1,99], positions= growthRates,widths=3 , patch_artist=True )
ax[1][1].grid(linestyle='--', linewidth=1,axis="x")
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.33)
for i in range(len(widths)):
    ax[1][1].plot(np.mean(widths[i]),growthRates[i],'o',alpha=0.5,color=colors[i],markersize=7)
ax[1][1].set_ylim(35,85)
ax[1][1].set_xlabel("Width ($\\mu m$)",fontsize=20,fontweight="bold")
ax[1][1].set_ylabel("Doubling time $\\tau$ (min)",fontsize=20,fontweight="bold")
ax[1][1].set_yticks([35,45,55,65,75,85])
ax[1][1].set_yticklabels([35,45,55,65,75,85])
ax[1][1].tick_params(axis="x", labelsize=15)
ax[1][1].tick_params(axis="y", labelsize=15)

#HISTS
ax[0][0].set_xlim(0,5)
ax[0][0].hist(lengths,bins=30,alpha=0.33,density=True,label=labels,histtype="barstacked")
ax[0][0].set_xlabel("Length ($\\mu m$)",fontsize=20,fontweight="bold")
ax[0][0].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][0].tick_params(axis="x", labelsize=15)
ax[0][0].tick_params(axis="y", labelsize=15)

ax[0][1].hist(widths,bins=25,alpha=0.33,density=True,label=labels,histtype="barstacked")
ax[0][1].set_xlabel("Width ($\\mu m$)",fontsize=20,fontweight="bold")
ax[0][1].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][1].tick_params(axis="x", labelsize=15)
ax[0][1].tick_params(axis="y", labelsize=15)


fig.legend(legendItems, labels,loc='upper center', bbox_to_anchor=(0.5, 1.08), fancybox=False, shadow=False, ncol=3,fontsize="xx-large")
fig.tight_layout()
fig.savefig('ThesisGraphs/GrowthLaw/OsmoLengthWidth.pdf', bbox_inches='tight')
fig.savefig('ThesisGraphs/GrowthLaw/OsmoLengthWidth.png', bbox_inches='tight')

#*********************************** INTENSITIES *******************************
plt.close("all")
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(13,9))
fig.subplots_adjust(left=0.07, right=0.93)
ax[0][0].text(0.05, 0.85 , "A", transform=ax[0][0].transAxes, size=30, weight='bold')
ax[0][1].text(0.05, 0.85 , "B", transform=ax[0][1].transAxes, size=30, weight='bold')
ax[1][0].text(0.05, 0.85 , "C", transform=ax[1][0].transAxes, size=30, weight='bold',color="black")
ax[1][1].text(0.05, 0.85 , "D", transform=ax[1][1].transAxes, size=30, weight='bold')

#Top left plot volume hist
ax[0][0].hist(volumes[1],alpha=0.33,density=True,color=colors[1])
ax[0][0].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[0][0].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][0].tick_params(axis="x", labelsize=15)
ax[0][0].tick_params(axis="y", labelsize=15)

#Top right plot intensity
ax[0][1].hist(intensities[1]/10000,alpha=0.33,density=True,color=colors[1])
ax[0][1].set_xlabel("Intensity (A.U)",fontsize=20,fontweight="bold")
ax[0][1].set_ylabel("Normalised frequency",fontsize=20,fontweight="bold")
ax[0][1].tick_params(axis="x", labelsize=15)
ax[0][1].tick_params(axis="y", labelsize=15)
#Bottom left correlation
hb = ax[1][0].hexbin(volumes[1],intensities[1]/10000,cmap="inferno",gridsize=30,mincnt=1)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax[1][0])
cax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(hb, cax=cax, orientation='vertical')
ax[1][0].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[1][0].set_ylabel("Intensity (A.U)",fontsize=20,fontweight="bold")
ax[1][0].tick_params(axis="x", labelsize=15)
ax[1][0].tick_params(axis="y", labelsize=15)
#Bottom right
from scipy.stats import spearmanr
for i in range(len(volumes)):
    corr, _ = spearmanr(volumes[i],intensities[i])
    ax[1][1].plot(volumes[i],intensities[i]/10000,'o',alpha=0.33,label ="$\\rho = {0:.2f}$".format(corr) )
ax[1][1].legend(fontsize="x-large")
ax[1][1].set_xlabel("Volume ($\\mu m^3$)",fontsize=20,fontweight="bold")
ax[1][1].set_ylabel("Intensity (A.U)",fontsize=20,fontweight="bold")
ax[1][1].tick_params(axis="x", labelsize=15)
ax[1][1].tick_params(axis="y", labelsize=15)

fig.legend(legendItems, labels,loc='upper center', bbox_to_anchor=(0.5, 1.08), fancybox=False, shadow=False, ncol=3,fontsize="xx-large")
fig.tight_layout()
fig.savefig('ThesisGraphs/GrowthLaw/OsmoIntensity.pdf', bbox_inches='tight')
fig.savefig('ThesisGraphs/GrowthLaw/OsmoIntensity.png', bbox_inches='tight')
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************



#****************** Surface area to volume ratio *******************************
plt.close("all")
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(9,6))
sa = np.asarray(meanSurfaceAreas)/np.asarray(meanVolumes)
sa_err = []
for i in range(len(sa)):
    sa_err.append(np.std(surfaceAreas[i]/volumes[i])/np.sqrt(len(surfaceAreas[i]/volumes[i])))
ax.plot((growthRates), (sa) , '--',linewidth=3,color="C6",zorder=0,label="High osmolarity")
for i in range(len(growthRates)):
    ax.scatter(growthRates[i], sa[i] , c=colors[i],s=200,label=labels[i])
    ax.errorbar(growthRates[i], sa[i], yerr=sa_err[i], ecolor="k", capsize=10, fmt="none",zorder=0 )

growthlawGrowthRates = [28, 40, 90]
growthLawSAs = [2.575425188606645, 3.1357344118273915, 3.9769239021141103]
growthLawSA_err = [0.027295227138177124 , 0.021106382327168875, 0.04251337872168573]
ax.plot(growthlawGrowthRates, growthLawSAs ,'--', color="C9",linewidth=3,zorder=0,label="Nutrient limitation")
colors = ["C1","C0","C2"]
labels = ["M63 + Glycerol, N=147" ,"","RDM, N = 182"]
for i in range(len(growthlawGrowthRates)):
    if i != 1:
        ax.scatter(growthlawGrowthRates[i], growthLawSAs[i] ,marker="D", c=colors[i],s=100,label=labels[i])
        ax.errorbar(growthlawGrowthRates[i], growthLawSAs[i], yerr=growthLawSA_err[i], ecolor="k", capsize=10, fmt="none",zorder=0 )


fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), fancybox=False, shadow=False, ncol=3,fontsize="large")
ax.set_xlabel("Doubling time $\\tau$ (min)",fontsize=20,fontweight="bold")
ax.set_ylabel("Surface area to volume ratio",fontsize=20,fontweight="bold")
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
ax.set_ylim(2.5,4.2)
ax.set_xlim(20,100)
fig.tight_layout()
fig.savefig('ThesisGraphs/GrowthLaw/OsmoSurface.pdf', bbox_inches='tight')
fig.savefig('ThesisGraphs/GrowthLaw/OsmoSurface.png', bbox_inches='tight')
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************


#******************** DAY TO DAY COMPARISON ************************************

plt.close("all")
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(9,6))

volumes = []
for n in range(len(dataDic2[prop])):
    volume = np.asarray(dataDic2["Volumes"][n])*((scaleFactors)**3)
    volumes.append(volume)
    ax.plot(np.mean(volume),n+1,'o',zorder=99)
    #ax.errorbar(np.mean(volume), n+1,xerr= np.std(volume)/np.sqrt(len(volume)),fmt="none",ecolor="black",capsize=10)

box = ax.boxplot(volumes,vert=False,whis=[1,99],widths=0.9 , patch_artist=True )
ax.set_yticklabels(dataDic2["Names"])

plt.show()
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
print("***************DONE**************")
#plt.show()
