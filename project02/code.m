%load training data set
load('TrainingSamplesDCT_8_new.mat')

[rowBG,columnBG] = size(TrainsampleDCT_BG);
[rowFG,columnFG] = size(TrainsampleDCT_FG);

%calculate prior probabilities or state probabilities
priorBG = rowBG/(rowBG + rowFG);
priorFG = rowFG/(rowBG + rowFG);

prior = [priorBG, priorFG];
counts = [rowBG, rowFG];

%plot observations of foreground(cheetah) and background(grass)
figure;
bar(counts);
set(get(gca,'YLabel'),'String', 'No. of observation');
set(get(gca,'XLabel'),'String', 'Prior of grass and cheetah');

%plot prior probabilities of states
figure;
bar(prior);
set(get(gca,'YLabel'),'String', 'Probability');
set(get(gca,'XLabel'),'String', 'Prior of grass and cheetah');

%calculate mean of observations mu1 and mu2
meanFG = mean(TrainsampleDCT_BG);
meanBG = mean(TrainsampleDCT_FG);

%calculate covariance matrices of two classes
covBG = cov(TrainsampleDCT_BG);
covFG = cov(TrainsampleDCT_FG);

%calculate variance vector of two classes
varBG = var(TrainsampleDCT_BG);
varFG = var(TrainsampleDCT_FG);

%maximum likehood calculations
%%% foreground
LikeHood_muFG = meanFG;
LikeHood_sigmaFG = mean(diag((TrainsampleDCT_FG - meanFG)*(TrainsampleDCT_FG - meanFG)'));

%maximum likehood calculations
%%% background
LikeHood_muBG = meanBG;
LikeHood_sigmaBG = mean(diag((TrainsampleDCT_BG - meanBG)*(TrainsampleDCT_BG - meanBG)'));

%plot of marginal densities
x_cheetah = zeros(64,601);
y_cheetah = zeros(64,601);
x_grass   = zeros(64,601);
y_grass   = zeros(64,601);
for k = 0:3
    figure;
    for j = 1:16
        subplot(4,4,j);
        i = 16*k+j;
        %foreground
        mu_cheetah = meanFG(i);
        sigma_cheetah = sqrt(varFG(i));
        x_cheetah(i,:) = (mu_cheetah-5*sigma_cheetah):(sigma_cheetah/60):(mu_cheetah+5*sigma_cheetah);
        y_cheetah(i,:) = normpdf(x_cheetah(i,:),mu_cheetah,sigma_cheetah);

        %background
        mu_grass = meanBG(i);
        sigma_grass = sqrt(varBG(i));
        x_grass(i,:) = (mu_grass-5*sigma_grass):(sigma_grass/60):(mu_grass+5*sigma_grass);
        y_grass(i,:) = normpdf(x_grass(i,:),mu_grass,sigma_grass);

        plot(x_cheetah(i,:),y_cheetah(i,:),'-b',x_grass(i,:),y_grass(i,:),'-r');
        title(['dimension ',num2str(i)]);
    end
end

best_features = [1,11,14,17,23,26,32,40];
worst_features = [3 ,4, 5, 59, 60,62 ,63, 64];

%%%plot 8 best features
figure;
for j = 1:8
    i = best_features(j);
    subplot(2,4,j);
    plot(x_cheetah(i,:),y_cheetah(i,:),'-b',x_grass(i,:),y_grass(i,:),'-r');
    title(['dimension ',num2str(i)]);
end

%%% plot 8 worst features
figure;
for j = 1:8
    i = worst_features(j);
    subplot(2,4,j);
    plot(x_cheetah(i,:),y_cheetah(i,:),'-b',x_grass(i,:),y_grass(i,:),'-r');
    title(['dimension ',num2str(i)]);
end

%%% reading original image and mask image
[img, map] = imread('cheetah.bmp');
[img_mask, map_mask] = imread('cheetah_mask.bmp');
img = im2double(img);
img_mask = im2double(img_mask);
zigzag = readmatrix('Zig-Zag Pattern.txt');

%%getting dct matrix of scanned image
dct_mat = zeros(65224,64);
for i = 1:255-7
    for j = 1:270-7
        temp = img(i:i+7,j:j+7);
        temp = dct2(temp);
        temp_vector = zeros(1,64);
        for x= 1:8
            for y = 1:8
                temp_vector(zigzag(x,y)+1) = temp(x,y);
            end
        end
         dct_mat((i-1)*263 + j,:) = temp_vector;
    end
end
%%%% BDR using 64-dimensional guassian
mask1 = zeros(255,270);
alphaBG1 = log(((2*pi)^64)*det(covBG)) - 2*log(priorBG);
alphaFG1 = log(((2*pi)^64)*det(covFG)) - 2*log(priorFG);

trueFG1=0;
trueBG1=0;
missFG1=0;
missBG1=0;

for i =1:255-7
    for j = 1:270-7
        count = (i-1)*263+j;
        x = dct_mat(count,:);
        dx_muBG = (x-meanBG)*(inv(covBG))*(x-meanBG)';
        dx_muFG = (x-meanFG)*(inv(covFG))*(x-meanFG)';
        gBG = 1/(1+exp(dx_muBG - dx_muFG + alphaBG1 - alphaFG1));
        if(gBG <0.5) 
            mask1(i,j) =1;
        end
        if img_mask(i,j) == 1
              trueFG1 = trueFG1+1;
              if mask1(i,j) == 0
                  missFG1 = missFG1+1;
              end
        else
              if mask1(i,j) == 1
                  missBG1 = missBG1 +1;
              end
              trueBG1 = trueBG1+1;
        end
    end
end
figure;
imshow(mat2gray(mask1));
falseness = sum(sum(xor(mask1,img_mask)))/65224;
error_prob = priorBG*(missBG1/trueBG1) + priorFG*(missFG1/trueFG1);

%%%%%%%%%%%%%% BDR using 8-dimensional guassian %%%%%%%%%%%%
sampleBG8 = zeros(1053,8);
sampleFG8 = zeros(250,8);
best = best_features;
for i = 1:8
    sampleBG8(:,i) = TrainsampleDCT_BG(:,best(i));
    sampleFG8(:,i) = TrainsampleDCT_FG(:,best(i));
end

meanBG8 = mean(sampleBG8);
meanFG8 = mean(sampleFG8);

covBG8 = cov(sampleBG8);
covFG8 = cov(sampleFG8);


mask2 = zeros(255,270);
alphaBG2 = log(((2*pi)^8)*det(covBG8)) - 2*log(priorBG);
alphaFG2 = log(((2*pi)^8)*det(covFG8)) - 2*log(priorFG);

trueFG2=0;
trueBG2=0;
missFG2=0;
missBG2=0;

for i =1:255-7
    for j = 1:270-7
        count = (i-1)*263+j;
        x = dct_mat(count,:);
        x_temp = zeros(1,8);
        for k = 1:8
            x_temp(k) = x(best(k));
        end
        x = x_temp;
        dx_muBG8 = (x-meanBG8)*(inv(covBG8))*(x-meanBG8)';
        dx_muFG8 = (x-meanFG8)*(inv(covFG8))*(x-meanFG8)';
        gBG8 = 1/(1+exp(dx_muBG8 - dx_muFG8 + alphaBG2 - alphaFG2));
        if(gBG8 <0.5) 
            mask2(i,j) =1;
        end
        if img_mask(i,j) == 1
              trueFG2 = trueFG2+1;
              if mask2(i,j) == 0
                  missFG2 = missFG2+1;
              end
        else
              if mask2(i,j) == 1
                  missBG2 = missBG2 +1;
              end
              trueBG2 = trueBG2+1;
        end
    end
end
figure;
imshow(mat2gray(mask2));
falseness2 = sum(sum(xor(mask2,img_mask)))/65224;
error_prob2 = priorBG*(missBG2/trueBG2) + priorFG*(missFG2/trueFG2);


