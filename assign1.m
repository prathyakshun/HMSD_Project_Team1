function [] = assign1(file, range)
range = str2double(range);
%path = 'Aqueduct_river_basins_TARIM.shp';
S = shaperead(file);
disp(S);
%range = input('Enter the resolution');
hold on;
len1 = length(min(S.X)-0.5 : range : max(S.X)+0.5);
len2 = length(min(S.Y)-0.5 : range : max(S.Y)+0.5);
arr1 = min(S.X)-0.5 : range : max(S.X)+0.5;
arr2 = min(S.Y)-0.5 : range : max(S.Y)+0.5;
mask = zeros(len1,len2);

% disp(length(arr1));
for i = 1:length(arr1)
%    disp(i)
    for j = 1:length(arr2)
        if inpolygon(arr1(i),arr2(j),S.X',S.Y') == 1
            mask(j,i) =  1;
        else
            mask(j,i) = 0;
        end
    end
end

figure(1);
imagesc(mask);
title('Mask of Tarim Basin')
colormap(gray);
axis equal;
figure(2);
plot(S.X',S.Y');
title('Tarim Basin')
axis equal;
hold off;
end