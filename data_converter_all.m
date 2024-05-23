% Structure data and save in a format that can be loaded in python
% Based on https://github.com/johnson-ying/Code-for-Ying-et-al.-2023/blob/main/Figure_2/all%20cells.m
% Clone CMBHOME (https://github.com/hasselmonians/CMBHOME) directly under the main directory
% Download allcells_col20isgridness.mat from https://www.dropbox.com/s/mnb7ar3f6wls7ng/allcells_col20isgridness.mat?dl=0
% and place this code and the mat file inside Code-for-Ying-et-al.-2023/Figure_2/
% Run the following commands from inside Figure_2 to generate data
% addpath ../session_data
% addpath ../session_data2
% addpath ../CMBHOME
% import CMBHOME.*
% data_converter_all

clear
load('allcells_col20isgridness.mat')
load('emptycells.mat')
load('removecirclecells.mat')
load('1and4components.mat')

AnimalsToInclude = [12015,12040,12375,12378,12644,12646,12655,12656,12746,12748,12756,12757, 12758,12759,12784,12785,12786,12787,12788,12790,12791,12792,12794,...
    3530,3532,3534,3601,3630,3631,3683,3781,3782,3783,3784,3791,3792,3794,3795,3798,3799,3827,3828,3884,3885,3894,3895,3927,3928,3931,4012,4014,4015,4020,...
    4117,4118,4125,4574,4593,4598,4599,4623,4754,4756,4757,4847,4849,5035,5036];

Grids = All_DataArray;
Genotype = Grids(:,2);
Genotype2 = [];
%0 = J20, 1 = WT

for i=1:length(Genotype)
    if(length(Genotype{i,1})==3)
        Genotype2(i,1) = 0;
    else
        Genotype2(i,1) = 1;
    end
end

Age = cell2mat(Grids(:,3));

%picking out cells 
wty = Grids((Genotype2==1 & Age>=90 & Age <=135),:); 
finder = ismember(cell2mat(wty(:,1)),AnimalsToInclude);
wty = wty(finder,:);

wta = Grids((Genotype2==1 & Age>135 & Age <=210),:); 
finder = ismember(cell2mat(wta(:,1)),AnimalsToInclude);
wta = wta(finder,:);

j20y = Grids((Genotype2==0 & Age>=90 & Age <=135),:); 
finder = ismember(cell2mat(j20y(:,1)),AnimalsToInclude);
j20y = j20y(finder,:);

j20a = Grids((Genotype2==0 & Age>135 & Age <=210),:); 
finder = ismember(cell2mat(j20a(:,1)),AnimalsToInclude);
j20a = j20a(finder,:);


dir_lists = {wty, wta, j20y, j20a};
dir_names = ["wty", "wta", "j20y", "j20a"];
empty_cells = {wtyempty, wtaempty, j20yempty, j20aempty};
circl_cells = {wtycircle,wtacircle,j20ycircle,j20acircle};
compo_cells = {wtycompo, wtacompo, j20ycompo, j20acompo};

mkdir("../extracted_all")
for cat_id = 1:4
dir_list = dir_lists{cat_id};
dir_name = dir_names(cat_id);
remove_cells = [empty_cells{cat_id}; circl_cells{cat_id}];
compo = compo_cells{cat_id};
mkdir("../extracted_all", dir_name)
for celliter = 1:size(dir_list,1)
    if ismember(celliter,remove_cells)
        continue
    end
    if ~ismember(celliter,compo)
        continue
    end
    disp(celliter)
    
    clear root
    %load cell
    load(dir_list{celliter,4}(end-18:end));
    cel = [dir_list{celliter,5},dir_list{celliter,6}];
    spikes = root.spike(cel(1),cel(2));
    
    clear new_data
    new_data = struct();
    new_data.x = root.x;
    new_data.y = root.y;
    new_data.t = root.ts;
    new_data.spki = spikes.i;
    new_data.spkt = spikes.ts;
    new_data.animal = int64(dir_list{celliter,1});
    new_data.age = int64(dir_list{celliter,3});

    %[oc, xdim, ydim] = root.Occupancy();
    %xdim2 = linspace(xdim(1), xdim(end), 36);
    %ydim2 = linspace(ydim(1), ydim(end), 36);

    %[rate_map1, ~, ~, ~, ~] = root.RateMap(cel, 'xdim', xdim2, 'ydim', ydim2, 'binside', 0, 'std_smooth_kernel', 0);
    %[rate_map2, ~, ~, ~, ~] = root.RateMap(cel, 'xdim', xdim2, 'ydim', ydim2, 'binside', 4, 'std_smooth_kernel', 4);
    
    %figure
    %imagesc(rate_map1, [0 max(rate_map1,[],'all')])
    

    save(strcat("../extracted_all/",dir_name,"/",int2str(celliter),"-",dir_list{celliter,4}(end-18:end),"-",int2str(cel(1)),"-",int2str(cel(2))),"-struct","new_data")
end
end

