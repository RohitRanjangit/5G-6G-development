load('TrainingSamplesDCT_8.mat');

back_data_size = size(TrainsampleDCT_BG);
fore_data_size = size(TrainsampleDCT_FG);
P_Y_back = back_data_size(1)/(back_data_size(1) + fore_data_size(1));
P_Y_fore = fore_data_size(1)/(back_data_size(1) + fore_data_size(1));

[~,training_feature_back] = max(TrainsampleDCT_BG(:,2:64),[],2);
[~,training_feature_fore] = max(TrainsampleDCT_FG(:,2:64),[],2);

fore_histogram = histogram(training_feature_fore,0:64);
hold on
back_histogram = histogram(training_feature_back,0:64);

P_X_back = back_histogram.Values / back_data_size(1);
P_X_fore = fore_histogram.Values / fore_data_size(1);

[img, map] = imread('cheetah.bmp');

img2 = im2double(img);

[img_x_size,img_y_size] = size(img2);
zigzag = readmatrix('Zig-Zag Pattern.txt');
cheetah_map = zeros(img_x_size,img_y_size);
for i = 1:img_x_size-7
    for j = 1:img_y_size-7
        block8x8 = img2(i:i+7,j:j+7);
        dct = dct2(block8x8);
        dct = abs(dct);
        [ii,jj] = find(dct == max(max(dct)));
        dct(ii,jj) = -Inf;
        [pos_x,pos_y] = find(dct == max(max(dct)));
        Feature = zigzag(pos_x,pos_y);
        if (P_X_fore(Feature+1)/P_X_back(Feature+1)) > (P_Y_back/P_Y_fore)
            cheetah_map(i,j) = 1;
        end
    end
end

imshow(cheetah_map);


[A2,B2] = imread('cheetah_mask.bmp');
A2 = A2/255;
falseness = sum(sum(xor(A2, cheetah_map))) / (img_x_size*img_y_size);





