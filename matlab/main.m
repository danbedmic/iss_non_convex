

% Please make sure that the following repositories are installed and
% working
% https://github.com/tum-vision/prost
% https://github.com/tum-vision/sublabel_relax/cvpr2016


% Add the respective paths
% addpath YOUR_PATH_GOES_HERE/prost/matlab
% addpath YOUR_PATH_GOES_HERE/sublabel_relax/cvpr2016



energy = input(append("\n\nPlease enter the respective number", ...
                "\n 1: ROF", ...
                "\n 2: Stereo Matching", ...
                "\n\n\n"));

            
if energy == 1
    data = input(append("Please enter the respective number", ...
                "\n 1: Bregman iteration (1st row of Fig. 3)", ...
                "\n 2: Lifted Bregman iteration with arbitrary subgradient (2nd row of Fig. 3)", ...                
                "\n 3: Lifted Bregman iteration with transformed subgradient (3rd row of Fig. 3)", ...
                "\n\n\n"));
    rof(data)
            
elseif energy == 2
    data = input(append("Please enter the respective number", ...
                "\n 1: Bike (Fig. 1) [long runtime]", ...
                "\n 2: Umbrella (Fig. 6) [long runtime]", ...
                "\n 3: Backpack (Fig. 6) [long runtime]", ...
                "\n 4: Rainbow isotopric (Fig. 5)", ...
                "\n 5: Rainbow anisotopric (Fig. 5)", ...
                "\n\n\n"));
     stereo_matching(data)
else
    fprintf("Plase enter valid number.")
end
    

