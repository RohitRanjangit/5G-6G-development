from scipy.io import loadmat
import scipy.fftpack
from matplotlib import pyplot as plt, image
import numpy as np
from matplotlib.pyplot import imread
from scipy.stats import norm



trainingData = loadmat('TrainingSamplesDCT_8_new.mat')

TrainsampleDCT_FG = np.array(trainingData['TrainsampleDCT_FG'])
TrainsampleDCT_BG = np.array(trainingData['TrainsampleDCT_BG'])

#calculate prior probabilities or state probabilities
priorBG = len(TrainsampleDCT_BG)/(len(TrainsampleDCT_BG) + len(TrainsampleDCT_FG))
priorFG = len(TrainsampleDCT_FG)/(len(TrainsampleDCT_BG) + len(TrainsampleDCT_FG))

prior = [priorBG, priorFG]
counts = [len(TrainsampleDCT_BG), len(TrainsampleDCT_FG)]

#plot observations of foreground(cheetah) and background(grass)

plt.bar(['back','fore'],counts)
plt.title('counts of observations')
plt.show()
plt.bar(['back','fore'],prior)
plt.title('prior probabilities')
plt.show()

#calculate mean of observations mu1 and mu2
meanFG = np.mean(np.array(TrainsampleDCT_BG),axis=0)
meanBG = np.mean(np.array(TrainsampleDCT_FG),axis=0)

#calculate covariance matrices of two classes
covBG = np.cov(np.stack(TrainsampleDCT_BG,axis=1))
covFG = np.cov(np.stack(TrainsampleDCT_FG,axis=1))

#calculate variance vector of two classes
varBG = np.var(TrainsampleDCT_BG,axis=0)
varFG = np.var(TrainsampleDCT_FG,axis=0)

#%maximum likehood calculations
#%%% foreground
LikeHood_muFG = meanFG
LikeHood_sigmaFG = np.mean(np.diag((TrainsampleDCT_FG - meanFG).dot((np.transpose(TrainsampleDCT_FG - meanFG)))))

#%maximum likehood calculations
# #%%% background
LikeHood_muBG = meanBG
LikeHood_sigmaBG = np.mean(np.diag((TrainsampleDCT_BG - meanBG).dot((np.transpose(TrainsampleDCT_BG - meanBG)))))

#plot of marginal densities
x_cheetah = [[] for _ in range(64)]
y_cheetah = [[] for _ in range(64)]
x_grass   = [[] for _ in range(64)]
y_grass   = [[] for _ in range(64)]

for k in range(4):
    fig, axs = plt.subplots(4,4)
    fig.suptitle('from dimension :' + str(16*k) + ' to ' + str(16*k+15))
    for j in range(16):
        i = 16*k+j
        #foreground
        mu_cheetah = meanFG[i]
        sigma_cheetah = np.sqrt(varFG[i])
        x_cheetah[i] = np.arange(mu_cheetah-5*sigma_cheetah,mu_cheetah+5*sigma_cheetah,sigma_cheetah/60)
        y_cheetah[i] = norm.pdf(x_cheetah[i], mu_cheetah,sigma_cheetah)
        axs[j//4,j%4].plot(x_cheetah[i],y_cheetah[i],'tab:green')

        #background
        mu_grass = meanBG[i]
        sigma_grass = np.sqrt(varBG[i])
        x_grass[i] = np.arange(mu_grass-5*sigma_grass,mu_grass+5*sigma_grass,sigma_grass/60)
        y_grass[i] = norm.pdf(x_grass[i], mu_grass,sigma_grass)
        axs[j//4,j%4].plot(x_grass[i],y_grass[i],'tab:red')
        axs[j//4,j%4].set_title('dimension' + str(i))
plt.show()

best_features = np.array([0,10,13,16,22,25,31,39])
worst_features = np.array([2 ,3, 4, 58, 59,61 ,62, 63])

fig,axs = plt.subplots(2,4)
fig.suptitle('best 8 feature graph')
for j,i in enumerate(best_features):
    mu_cheetah = meanFG[i]
    sigma_cheetah = np.sqrt(varFG[i])
    axs[j//4,j%4].plot(x_cheetah[i],y_cheetah[i],'tab:green')

    mu_grass = meanBG[i]
    sigma_grass = np.sqrt(varBG[i])
    axs[j//4,j%4].plot(x_grass[i],y_grass[i],'tab:red')
    axs[j//4,j%4].set_title('dimension' + str(i))
plt.show()
fig,axs = plt.subplots(2,4)
fig.suptitle('worst 8 feature graph')
for j,i in enumerate(worst_features):
    mu_cheetah = meanFG[i]
    sigma_cheetah = np.sqrt(varFG[i])
    axs[j//4,j%4].plot(x_cheetah[i],y_cheetah[i],'tab:green')

    mu_grass = meanBG[i]
    sigma_grass = np.sqrt(varBG[i])
    axs[j//4,j%4].plot(x_grass[i],y_grass[i],'tab:red')
    axs[j//4,j%4].set_title('dimension' + str(i))
plt.show()
# reading original image and mask image
img = image.imread("cheetah.bmp")
rgb_weights = [0.2989, 0.5870, 0.1140]
img = np.dot(img[...,:3], rgb_weights)/255
img_mask = image.imread('cheetah_mask.bmp')
zigzag = np.loadtxt("Zig-Zag Pattern.txt",dtype='i')
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

#getting dct matrix of scanned image
dct_mat = np.zeros((65224,64))
for i in range(248):
    for j in range(263):
        temp= np.array(dct2(img[i:(i+8),j:(j+8)]))
        temp_vector = np.zeros(64)
        for x in range(8):
            for y in range(8):
                temp_vector[zigzag[x][y]] = temp[x][y]
        dct_mat[(i)*263 + j] = temp_vector


###########################################################################################
## BDR using 64-dimensional guassian
mask1 = np.zeros((255,270))
alphaBG1 = np.log(np.power(2*np.pi,64)*np.linalg.det(covBG)) - 2*np.log(priorBG)
alphaFG1 = np.log(np.power(2*np.pi,64)*np.linalg.det(covFG)) - 2*np.log(priorFG)

trueFG1=0
trueBG1=0
missFG1=0
missBG1=0

for i in range(248):
    for j in range(263):
        count = (i)*263+j
        x = dct_mat[count]
        dx_muBG = (x-meanBG).dot((np.linalg.inv(covBG)).dot(np.transpose(x-meanBG)))
        dx_muFG = (x-meanFG).dot((np.linalg.inv(covFG)).dot(np.transpose(x-meanFG)))
        gBG = 1/(1+np.exp(dx_muBG - dx_muFG + alphaBG1 - alphaFG1))
        if(gBG <0.5) :
            mask1[i][j] =1
        if img_mask[i][j] == 1:
            trueFG1 = trueFG1+1
            if mask1[i][j] == 0:
                missFG1 = missFG1+1
        else:
            if mask1[i][j] == 1:
                missBG1 = missBG1 +1
            trueBG1 = trueBG1+1


plt.imshow(mask1)
plt.show()
falseness = (mask1!=img_mask).sum()/65224
# error_prob = priorBG*(missBG1/trueBG1) + priorFG*(missFG1/trueFG1)

###########################################################################################
# BDR using 8-dimensional guassian 
sampleBG8 = np.zeros((1053,8))
sampleFG8 = np.zeros((250,8))
best = best_features
for i in range(8):
    sampleBG8[:,i] = TrainsampleDCT_BG[:,best[i]]
    sampleFG8[:,i] = TrainsampleDCT_FG[:,best[i]]

meanBG8 = np.mean(sampleBG8,axis=0)
meanFG8 = np.mean(sampleFG8,axis=0)

covBG8 = np.cov(np.stack(sampleBG8,axis=1))
covFG8 = np.cov(np.stack(sampleFG8,axis=1))


mask2 = np.zeros((255,270))
alphaBG2 = np.log(np.power(2*np.pi,8)*np.linalg.det(covBG8)) - 2*np.log(priorBG)
alphaFG2 = np.log(np.power(2*np.pi,8)*np.linalg.det(covFG8)) - 2*np.log(priorFG)


trueFG2=0
trueBG2=0
missFG2=0
missBG2=0

for i in range(248):
    for j in range(263):
        count = (i)*263+j
        x = dct_mat[count]
        x_temp = np.zeros(8)
        for k in range(8):
            x_temp[k] = x[best[k]]
        x = x_temp
        dx_muBG8 = (x-meanBG8).dot((np.linalg.inv(covBG8)).dot(np.transpose(x-meanBG8)))
        dx_muFG8 = (x-meanFG8).dot((np.linalg.inv(covFG8)).dot(np.transpose(x-meanFG8)))
        gBG8 = 1/(1+np.exp(dx_muBG8 - dx_muFG8 + alphaBG2 - alphaFG2))
        if(gBG8 <0.5) :
            mask2[i][j] =1
        if img_mask[i][j] == 1:
            trueFG2 = trueFG2+1
            if mask2[i][j] == 0:
                missFG2 = missFG2+1
        else:
            if mask2[i][j] == 1:
                missBG2 = missBG2 +1
            trueBG2 = trueBG2+1


plt.imshow(mask2)
plt.show()
falseness2 = (mask2!=img_mask).sum()/65224
# error_prob = priorBG*(missBG1/trueBG1) + priorFG*(missFG1/trueFG1)
