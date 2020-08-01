from scipy.io import loadmat
import scipy.fftpack
from matplotlib import pyplot as plt, image
import numpy as np
from matplotlib.pyplot import imread
from scipy.stats import norm

data_sets = loadmat('TrainingSamplesDCT_subsets_8.mat')
alpha = loadmat('Alpha.mat')

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

for strategy in range(2):
    Prior_mat = {}
    if strategy ==1:
        Prior_mat = loadmat('Prior_1.mat')
    else:
        Prior_mat = loadmat('Prior_2.mat')
    for set_num in range(4):
        data_set_BG = []
        data_set_FG = []
        if set_num == 0:
            data_set_FG = data_sets['D1_FG']
            data_set_BG = data_sets['D1_BG']
        elif set_num == 1:
            data_set_FG = data_sets['D2_FG']
            data_set_BG = data_sets['D2_BG']
        elif set_num == 2:
            data_set_FG = data_sets['D3_FG']
            data_set_BG = data_sets['D3_BG']
        elif set_num == 3:
            data_set_FG = data_sets['D4_FG']
            data_set_BG = data_sets['D4_BG']

        ML_error = []
        BAYES_error = []
        MAP_error =[]

        for Alpha in alpha['alpha'][0]:
            ######### ML estimation #############
            rowBG = len(data_set_BG)
            rowFG = len(data_set_FG)

            priorBG = rowBG/(rowBG + rowFG)
            priorFG = rowFG/(rowBG + rowFG)

            #calculate mean of observations mu1 and mu2
            meanFG = np.mean(np.array(data_set_FG),axis=0)
            meanBG = np.mean(np.array(data_set_BG),axis=0)
            
            #calculate covariance matrices of two classes
            covBG = np.cov(np.stack(data_set_BG,axis=1))
            covFG = np.cov(np.stack(data_set_FG,axis=1))

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


            #plt.imshow(mask1)
            #plt.show()
            falseness = (mask1!=img_mask).sum()/65224
            ML_error += [falseness]

            ######### Bayes estimation #############
            rowBG = len(data_set_BG)
            rowFG = len(data_set_FG)

            priorBG = rowBG/(rowBG + rowFG)
            priorFG = rowFG/(rowBG + rowFG)

            #calculate mean of observations mu1 and mu2
            meanFG = np.mean(np.array(data_set_FG),axis=0)
            meanBG = np.mean(np.array(data_set_BG),axis=0)
            
            #calculate covariance matrices of two classes
            covBG = np.cov(np.stack(data_set_BG,axis=1))
            covFG = np.cov(np.stack(data_set_FG,axis=1))

            sumFG = np.sum(np.array(data_set_FG),axis=0)
            sumBG = np.sum(np.array(data_set_BG),axis=0)

            cov0 = np.zeros((64,64))
            for i in range(0,64):
                cov0[i][i] = Alpha*Prior_mat['W0'][i]

            #calculate parameters of X|T posterior
            ##BG
            meanN_BG = cov0.dot((np.linalg.inv(covBG + rowBG*cov0)).dot(np.transpose(sumBG))) + covBG.dot((np.linalg.inv(covBG + rowBG*cov0)).dot(np.transpose(Prior_mat['mu0_BG'][0])))
            meanN_BG = np.transpose(menaN_BG)
            covN_BG = covBG.dot((np.linalg.inv(covBG + rowBG*cov0)).dot(cov0))

            ##FG
            meanN_FG = cov0.dot((np.linalg.inv(covFG + rowFG*cov0)).dot(np.transpose(sumFG))) + covFG.dot((np.linalg.inv(covFG + rowFG*cov0)).dot(np.transpose(Prior_mat['mu0_FG'][0])))
            meanN_FG = np.transpose(menaN_FG)
            covN_FG = covFG.dot((np.linalg.inv(covFG + rowFG*cov0)).dot(cov0))

            mask1 = np.zeros((255,270))
            alphaBG1 = np.log(np.power(2*np.pi,64)*np.linalg.det(covBG + covN_BG)) - 2*np.log(priorBG)
            alphaFG1 = np.log(np.power(2*np.pi,64)*np.linalg.det(covFG + covN_FG)) - 2*np.log(priorFG)

            trueFG1=0
            trueBG1=0
            missFG1=0
            missBG1=0

            for i in range(248):
                for j in range(263):
                    count = (i)*263+j
                    x = dct_mat[count]
                    dx_muBG = (x-meanN_BG).dot((np.linalg.inv(covBG + covN_BG)).dot(np.transpose(x-meanN_BG)))
                    dx_muFG = (x-meanN_FG).dot((np.linalg.inv(covFG + covN_FG)).dot(np.transpose(x-meanN_FG)))
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


            #plt.imshow(mask1)
            #plt.show()
            falseness = (mask1!=img_mask).sum()/65224
            BAYES_error += [falseness]
            ######### MAP-estimation ###########

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
                    dx_muBG = (x-meanN_BG).dot((np.linalg.inv(covBG)).dot(np.transpose(x-meanN_BG)))
                    dx_muFG = (x-meanN_FG).dot((np.linalg.inv(covFG)).dot(np.transpose(x-meanN_FG)))
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


            #plt.imshow(mask1)
            #plt.show()
            falseness = (mask1!=img_mask).sum()/65224
            MAP_error += [falseness]
        #plot
        plt.plot(np.log(alpha),BAYES_error,np.log(alpha),ML_error,np.log(alpha),MAP_error)
        plt.show()