load('TrainingSamplesDCT_subsets_8.mat');
load('Alpha.mat');

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

for strategy = 1:2
    if strategy ==1
        load('Prior_1.mat');
    else 
        load('Prior_2.mat');
    end
    for set = 1:4
        if set == 1
            data_set_FG = D1_FG;
            data_set_BG = D1_BG;
        elseif set == 2
            data_set_FG = D2_FG;
            data_set_BG = D2_BG;
        elseif set == 3
            data_set_FG = D3_FG;
            data_set_BG = D3_BG;
        elseif set == 4
            data_set_FG = D4_FG;
            data_set_BG = D4_BG;
        end
        ML_error = [];
        BAYES_error = [];
        MAP_error =[];
        for Alpha = alpha           
            %%%%%%% ML estimation %%%%%%%%%
            [rowBG, columnBG] = size(data_set_BG);
            [rowFG, columnFG] = size(data_set_FG);
            
            priorBG = rowBG/(rowBG + rowFG);
            priorFG = rowFG/(rowBG + rowFG);
            
            meanFG = mean(data_set_FG);
            meanBG = mean(data_set_BG);
            
            covFG = cov(data_set_FG);
            covBG = cov(data_set_BG);
            
            mask = zeros(255,270);
            alphaBG = log(((2*pi)^64)*det(covBG)) - 2*log(priorBG);
            alphaFG = log(((2*pi)^64)*det(covFG)) - 2*log(priorFG);
            
            trueFG=0;
            trueBG=0;
            missFG=0;
            missBG=0;
            
            for i =1:255-7
                for j = 1:270-7
                    count = (i-1)*263+j;
                    x = dct_mat(count,:);
                    dx_muBG = (x-meanBG)*(inv(covBG))*(x-meanBG)';
                    dx_muFG = (x-meanFG)*(inv(covFG))*(x-meanFG)';
                    gBG = 1/(1+exp(dx_muBG - dx_muFG + alphaBG - alphaFG));
                    if(gBG <0.5)
                        mask(i,j) =1;
                    end
                    if img_mask(i,j) == 1
                        trueFG = trueFG+1;
                        if mask(i,j) == 0
                            missFG = missFG+1;
                        end
                    else
                        if mask(i,j) == 1
                            missBG = missBG +1;
                        end
                        trueBG = trueBG+1;
                    end
                end
            end
%             figure;
%             imshow(mat2gray(mask));
            falseness = sum(sum(xor(mask,img_mask)))/65224;
            error_prob = priorBG*(missBG/trueBG) + priorFG*(missFG/trueFG);
            ML_error = [ML_error error_prob];
            %%%%%%%%%%%%%% BayesEstimation %%%%%%%%%%%%%%%%%%%
            sumFG = sum(data_set_FG);
            sumBG = sum(data_set_BG);
            
            covBG = cov(data_set_BG);
            covFG = cov(data_set_FG);
            
            cov0 = zeros(64,64);
            for i = 1:64
                cov0(i,i) = Alpha*W0(i);
            end
%             disp(cov0);
            %calculate parameters of X|T posterior
            %BG%
            meanN_BG = (cov0*(inv(covBG + rowBG*cov0))*sumBG' + covBG*(inv(covBG + rowBG*cov0))*mu0_BG')';
%             disp("covBG*(inv(covBG + rowBG*cov0))*mu0_BG')'",(covBG*(inv(covBG + rowBG*cov0))*mu0_BG')');
            covN_BG = covBG*(inv(covBG + rowBG*cov0))*cov0;
            %FG%
            meanN_FG = (cov0*(inv(covFG + rowFG*cov0))*sumFG' + covFG*(inv(covFG + rowFG*cov0))*mu0_FG')';
            covN_FG = covFG*(inv(covFG + rowFG*cov0))*cov0;
            
            %Classification Part %
            mask = zeros(255,270);
            alphaBG = log(((2*pi)^64)*det(covBG + covN_BG)) - 2*log(priorBG);
            alphaFG = log(((2*pi)^64)*det(covFG + covN_FG)) - 2*log(priorFG);
            
            trueFG=0;
            trueBG=0;
            missFG=0;
            missBG=0;
            
            for i =1:255-7
                for j = 1:270-7
                    count = (i-1)*263+j;
                    x = dct_mat(count,:);
                    dx_muBG = (x-meanN_BG)*(inv(covBG + covN_BG))*(x-meanN_BG)';
                    dx_muFG = (x-meanN_FG)*(inv(covFG + covN_FG))*(x-meanN_FG)';
                    gBG = 1/(1+exp(dx_muBG - dx_muFG + alphaBG - alphaFG));
                    if(gBG <0.5)
                        mask(i,j) =1;
                    end
                    if img_mask(i,j) == 1
                        trueFG = trueFG+1;
                        if mask(i,j) == 0
                            missFG = missFG+1;
                        end
                    else
                        if mask(i,j) == 1
                            missBG = missBG +1;
                        end
                        trueBG = trueBG+1;
                    end
                end
            end
%             figure;
%             imshow(mat2gray(mask));
            falseness = sum(sum(xor(mask,img_mask)))/65224;
            error_prob = priorBG*(missBG/trueBG) + priorFG*(missFG/trueFG);
            BAYES_error = [BAYES_error error_prob];
            
            %%%%%%%%%%%%%% BayesEstimation %%%%%%%%%%%%%%%%%%%
            
            
            %Classification Part %
            mask = zeros(255,270);
            alphaBG = log(((2*pi)^64)*det(covBG)) - 2*log(priorBG);
            alphaFG = log(((2*pi)^64)*det(covFG)) - 2*log(priorFG);
            
            trueFG=0;
            trueBG=0;
            missFG=0;
            missBG=0;
            
            for i =1:255-7
                for j = 1:270-7
                    count = (i-1)*263+j;
                    x = dct_mat(count,:);
                    dx_muBG = (x-meanN_BG)*(inv(covBG))*(x-meanN_BG)';
                    dx_muFG = (x-meanN_FG)*(inv(covFG))*(x-meanN_FG)';
                    gBG = 1/(1+exp(dx_muBG - dx_muFG + alphaBG - alphaFG));
                    if(gBG <0.5)
                        mask(i,j) =1;
                    end
                    if img_mask(i,j) == 1
                        trueFG = trueFG+1;
                        if mask(i,j) == 0
                            missFG = missFG+1;
                        end
                    else
                        if mask(i,j) == 1
                            missBG = missBG +1;
                        end
                        trueBG = trueBG+1;
                    end
                end
            end
%             figure;
%             imshow(mat2gray(mask));
            falseness = sum(sum(xor(mask,img_mask)))/65224;
            error_prob = priorBG*(missBG/trueBG) + priorFG*(missFG/trueFG);
            MAP_error = [MAP_error error_prob];
        end
        %plot
        figure;
        plot(log(alpha),BAYES_error,log(alpha),ML_error,log(alpha),MAP_error);
        legend('Predict','ML','MAP');
        title('PoE vs Alpha');
        xlabel('Alpha');
        ylabel('PoE');
    end
end