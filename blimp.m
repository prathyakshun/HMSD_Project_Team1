function [] = blimp(filepath, discharge_csv)

% Mask File -> b
% Ppt Data -> basin_daily_ppt.mat
% Discharge Location -> discharge/
% Discharge Values -> discharge/asin.mat
% Lat-long -> latlong_lim
% DEM File -> DEM_clipped.tif

%filepath='DEM_clipped.tif';
%Read GTiff
[image,geo]=geotiffread(filepath);
%Read DEM info
info=geotiffinfo(filepath);
[m,n] = size(image);
K = 0.045*ones(5); % 3x3 mean kernel
% image = conv2(image,K, 'same'); % Convolve keeping size of I
image = smoothdata(image);
hold on
figure(3)
title('Smoothened DEM');
surfl(image)
title('Smoothened DEM');
colormap(pink)    % change color map
shading interp    % interpolate colors across lines and faces
hold off

flow_output = get_flow_map(image);
%hold on 
%figure(1)
%imagesc(flow_output); 
%colorbar
%hold off
%------------------
% hold on
% surfl(image)
% colormap(pink)    % change color map
% shading interp    % interpolate colors across lines and faces
%-------Update Table ----
T = readtable(discharge_csv);
Update_T = getPos(T,info,757,912);
%-------Inverted Flow ----
inverted_map = getInvertedFlow(flow_output);
%hold on
%figure(2)
%imagesc(inverted_map); 
%colorbar
%hold off
%-------Labelling Flow ------
labelled_map = labelpieces(Update_T, inverted_map);
%disp('Abeeel')
%hold on
%figure(3)
%imagesc(labelled_map);
%colorbar
%hold off
disp(max(labelled_map(:)))
figure(2);
imshow('basins.png');
figure(1);
imshow('basin_delineated.png');
figure(4);
imshow('rainfall_runoff.png');
figure(5);
imshow('DEM.jpg');

function [flow_map]  = get_flow_map(image)
padded_image = conv2(image,[0,0,0;0,1,0;0,0,0]);
[m,n] = size(padded_image);
flow_map = zeros(size(padded_image));
for i = 2:m-1
    for j = 2:n-1
        if padded_image(i,j) == 0
            continue
        end
        vect = [padded_image(i-1,j), ...
            padded_image(i-1,j+1), ...
            padded_image(i,j+1),...
            padded_image(i+1,j+1), ...
            padded_image(i+1,j), ...
            padded_image(i+1,j-1), ...
            padded_image(i,j-1), ...
            padded_image(i-1,j-1)];
        vect = -vect + padded_image(i,j);
        [~, idx] = max(vect);
%         flow_map(i,j) = 2^(idx-1);
        flow_map(i,j) = idx;
    end
end
flow_map = flow_map(2:m-1,2:n-1);
end

function [accum_map] = get_accum_map(flow_map)
dir_array = [-1,0; ...
            -1,+1; ...
            0,+1;...
            +1,+1; ...
            +1,0; ...
            +1,-1; ...
            0,-1; ...
            -1,-1];
padded_image = conv2(flow_map,[0,0,0;0,1,0;0,0,0]);
accum_map = zeros(size(padded_image));
[m,n] = size(padded_image);    
for i = 2:m-1
    for j = 2:n-1
            if padded_image(i,j) == 0
            continue        
            end
       tmp = dir_array(padded_image(i,j),:) + [i,j];
       new_index = [i,j]+dir_array(padded_image(i,j),:) ; 
       accum_map(new_index(1),new_index(2)) = 1+accum_map(new_index(1),new_index(2)) + accum_map(i,j);
    end
end
accum_map = accum_map(2:m-1,2:n-1);
end

function [table] = getPos(table,info,rowNum,colNum)
    bb = info.BoundingBox;
    latmin = bb(1,2);
    latmax = bb(2,2);
    longmin = bb(1,1);
    longmax = bb(2,1);
    table.XPos = round(1+(((table.latitude-latmin)/(latmax - latmin))*rowNum));
    table.YPos = round(1+(((table.longitude-longmin)/(longmax - longmin))*colNum));
end

function [flow_map] = getInvertedFlow(flow_map)
    [m,n] = size(flow_map);
    for i = 1:m
        for j = 1:n
            if flow_map(i,j) == 0
                continue
            elseif flow_map(i,j) > 4
                flow_map(i,j) = 8 - flow_map(i,j);
            else
                flow_map(i,j) = 4 + flow_map(i,j);
            end 
        end
    end
end


function [sector_map] = recursive_label(labelled_map,sector_map,x1,y1,label) 
%     disp(strcat('Start','-',int2str(x1),'-',int2str(y1)))
    if labelled_map(x1,y1) == 0
%         disp(strcat('valIs0-soret','-',int2str(x1),'-',int2str(y1)))
        return
    end
    dir_array = [-1,0; ...
            -1,+1; ...
            0,+1;...
            +1,+1; ...
            +1,0; ...
            +1,-1; ...
            0,-1; ...
            -1,-1];
    for i = 1:8
        tmp = dir_array(i,:);
        new_index = [x1,y1]+tmp;
        if sector_map(new_index(1),new_index(2)) ~= 0
            continue
        end
        if labelled_map(new_index(1),new_index(2)) == i
            sector_map(new_index(1),new_index(2))  = label;
            sector_map = recursive_label(labelled_map,sector_map,new_index(1),new_index(2),label);
        end
    end
%     disp(strcat('Done','-',int2str(x1),'-',int2str(y1)))
  end
    
function [labelled_map] = labelpieces(T,inverted_flow_image)
    padded_image = conv2(inverted_flow_image,[0,0,0;0,1,0;0,0,0]);
    labelled_map = zeros(size(padded_image));
    [m,n] = size(labelled_map);
%     
    for i = 1:height(T)
        y1 = T.(5)(i) + 1;
        x1 = T.(4)(i) + 1;
        labelled_map(x1,y1) = i;
    end
    for i = 1:height(T)
        y1 = T.(5)(i) + 1;
        x1 = T.(4)(i) + 1;
        labelled_map = recursive_label(inverted_flow_image,labelled_map,x1,y1,i);
    end 
%     labelled_map = recursive_label(inverted_flow_image,labelled_map, 325,838,2);
    labelled_map = labelled_map(2:m-1,2:n-1);
end
end