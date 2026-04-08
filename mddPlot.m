function mddPlot(caseFolder, rasterBin, xsFile, hisFile, mapFiles, netFiles, exportVideo, exportImages, exportSTL, nameVideo, nameImages, nameSTL, rhoS, width, gridRes)
    arguments
        caseFolder {mustBeFolder} = '' 
        rasterBin  {mustBeText} = '*.tif'
        xsFile     {mustBeText} = '*.txt'
        hisFile    {mustBeText} = '*/*his.nc'  % Relative to 'caseFolder'
        mapFiles   {mustBeText} = '*/*map.nc'  % Relative to 'caseFolder'
        netFiles   {mustBeText} = '*net.nc'    % Relative to 'caseFolder'
        exportVideo logical = true
        exportImages logical = false
        exportSTL logical = false
        nameVideo {mustBeText} = 'Simulation_Summary.avi'
        nameImages {mustBeText} = 'images/step_%d.png'
        nameSTL {mustBeText} = 'Final_Bed_Surface.stl'
        rhoS double {mustBePositive} = 1600
        width double {mustBePositive} = 47.17
        gridRes double {mustBePositive} = 1.0
    end

% Validate inputs
rasterBin = singleFileGlob(rasterBin);
xsFile = singleFileGlob(xsFile);
hisFile = singleFileGlob(caseFolder, hisFile);
mapFiles = multiFileGlob(caseFolder, mapFiles);
netFiles = multiFileGlob(caseFolder, netFiles);

% maybe print some stuff here.
% fprintf('  History File: %s', hisFile);
% fprintf('%d Network Files\n', numel(netFiles));
% fprintf('%d Map Files\n', numel(mapFiles));

if numel(mapFiles) ~= numel(netFiles)
    warning("Mismatch in output files %d net, %d map", numel(netFiles), numel(mapFiles))
end

% Non-interactive detection
if isempty(getenv('DISPLAY')) || ~usejava('desktop')
    warning('No DISPLAY available: switching to headless plotting mode. Figures are invisible.');
    set(0, 'DefaultFigureVisible', 'off');
end

%% 2. LOAD TIME SERIES DATA (History File)
% -------------------------------------------------------------------------
fprintf('Loading history data...\n');
t = ncread(hisFile, 'time');
dt = t(2) - t(1);

% Qw: Discharge (m3/s) | Qs: Cumulative Bedload (kg)
Qw = ncread(hisFile, 'cross_section_discharge'); 
QsCum = ncread(hisFile, 'cross_section_bedload_sediment_transport');

% Convert cumulative mass to instantaneous volumetric flux (m3/m/s)
% diff() gets the mass per timestep, then divide by time, width, and density.
Qs = diff(QsCum, 1, 2) / dt / width / rhoS;

%% 3. LOAD MESH & DOMAIN STRUCTURE (Network File)
% -------------------------------------------------------------------------
fprintf('Parsing network and partition info...\n');
vInfo = ncinfo(netFiles{1});
numSteps = size(ncread(mapFiles{1}, 'time'), 1);
% Extract number of partitions from the idomain attribute

% Could add check to confirm number files same as in partition info.
% partitions = vInfo.Variables.mesh2d_netelem_domain

% Newer delft outputs use this updated schema.
links = ncread(netFiles{1}, 'NetElemLink')';
Xn = ncread(netFiles{1}, 'mesh2d_node_x');
Yn = ncread(netFiles{1}, 'mesh2d_node_y');
Zn = ncread(netFiles{1}, 'mesh2d_node_z');
TRI = ncread(netFiles{1}, 'NetElemNode')'; % Triangulation connectivity
idomainSize = size(links, 1);

%% 4. AGGREGATE MULTI-DOMAIN SPATIAL DATA
% -------------------------------------------------------------------------
% Pre-allocate global arrays for speed
X = zeros(idomainSize, 1);
Y = zeros(idomainSize, 1);
waterDepth = zeros(idomainSize, numSteps);     % Water depth
waterLevel = zeros(idomainSize, numSteps);    % Water level (s1)
medianGrainSize = zeros(idomainSize, numSteps);    % Median grain size (D50)

fprintf('Looping through partitions...\n');
for q = 1:numel(mapFiles)
    try
        %fname = sprintf('%s_%04d_map.nc', map_prefix, q);'
        fprintf('\r    %s\n', mapFiles{q})
        % Global mapping indices for this partition
        gIdx = int32(ncread(mapFiles{q}, 'mesh2d_flowelem_globalnr'));
        
        % Static Geometry
        X(gIdx) = ncread(mapFiles{q}, 'mesh2d_face_x');
        Y(gIdx) = ncread(mapFiles{q}, 'mesh2d_face_y');
        
        % Dynamic Results
        waterDepth(gIdx, :)  = ncread(mapFiles{q}, 'mesh2d_waterdepth');
        waterLevel(gIdx, :) = ncread(mapFiles{q}, 'mesh2d_s1');
        medianGrainSize(gIdx, :) = ncread(mapFiles{q}, 'mesh2d_dg');
    catch ME
        warning('Could not read %s', mapFiles{q})
    end
end

fprintf('Interpolate to regular grid.\n');

% Derived Bed Elevation (zBed = waterLevel - waterDepth)
zBed = waterLevel - waterDepth;

%% 5. GRID INTERPOLATION SETUP
% -------------------------------------------------------------------------
% Define a regular grid for raster-based analysis and plotting
xVec = floor(min(X)):gridRes:ceil(max(X));
yVec = floor(min(Y)):gridRes:ceil(max(Y));
[XX, YY] = meshgrid(xVec, yVec);

fprintf('Creating a mask for the active flow area using an alphaShape\n')
shp = alphaShape(X, Y);
inMask = inShape(shp, XX, YY);

%% 6. VISUALIZATION: DEPTH & DOD ANIMATION
% -------------------------------------------------------------------------
% Setup custom colormaps
% Red-Blue for DoD (Erosion/Deposition)
m1 = 128;
r = [linspace(0, 1, m1)'; ones(m1, 1)];
g = [linspace(0, 1, m1)'; linspace(1, 0, m1)'];
b = [ones(m1, 1); linspace(1, 0, m1)'];
cmapDod = [r g b];

% Load Cross-section locations
xs = load(xsFile);
xsOrder = [1:2:25, 29 31 33 37 35 39]; 

fig = figure('Position', [10 370 1280 976], 'Color', 'w');
tlo = tiledlayout(3, 4, 'TileSpacing', 'Compact');

for a = 2:numSteps
    % 1. Interpolate current data to regular grid for this timestep
    % 'nearest' is faster; 'linear' is smoother.
    bedInterp = griddata(X, Y, zBed(:, a), XX, YY, 'nearest');
    bedInterp(~inMask) = NaN;
    
    % Calculate DoD relative to initial timestep
    dodInstant = zBed(:, a) - zBed(:, 1);
    
    % Mapping back to Mesh Nodes for trisurf plotting
    ZnNodes = interp2(xVec, yVec, bedInterp, Xn, Yn);
    depthNodes = interp1(X, waterDepth(:, a), Xn); % Simple mapping to nodes
    
    % --- SUBPLOT 1: Water Depth ---
    ax1 = nexttile(1, [2, 2]);
    h1 = trisurf(TRI(:, 1:3), Xn, Yn, ZnNodes, depthNodes, 'EdgeColor', 'none');
    view(0, 90); axis equal; colorbar;
    colormap(ax1, parula); clim([0 2]);
    title(sprintf('Water Depth (h) | Time: %d hrs', round(t(a)/3600)));
    
    % --- SUBPLOT 2: Bed Level Difference (DoD) ---
    ax2 = nexttile(3, [2, 2]);
    % We plot dodInstant mapped to nodes
    dodNodes = interp1(X, dodInstant, Xn); 
    h2 = trisurf(TRI(:, 1:3), Xn, Yn, ZnNodes, dodNodes, 'EdgeColor', 'none');
    view(0, 90); axis equal; colorbar;
    colormap(ax2, cmapDod); clim([-1 1]);
    title('Morphological Change (DoD) [m]');

    % --- SUBPLOT 3: Longitudinal Mass Balance ---
    % (Requires mapping morphological change to spatial bins)
    % This logic can be expanded based on the 'binsraster' logic in original code
    
    drawnow;
    if exportVideo    
        v = VideoWriter(nameVideo);
        v.FrameRate = 10;
        open(v);
        writeVideo(v, getframe(gcf));
    end
    if exportImages    
        imagaName = sprintf(nameImages, a);
        imwrite(v, getframe(gcf));
    end
end

%% 7. STL EXPORT FOR 3D MODELING (Blender/Unity)
% -------------------------------------------------------------------------
% Centering coordinates to avoid large-coordinate jitter in 3D software
if exportSTL
    offsetX = floor(min(Xn));
    offsetY = floor(min(Yn));
    verts = [(Xn - offsetX), (Yn - offsetY), ZnNodes];
    TR = triangulation(TRI(:, 1:3), verts);
    stlwrite(TR, nameSTL);

    fprintf('Processing complete. STL exported.\n');
end

% matlab doesn't natively expand globs.
function [path] = singleFileGlob(varargin)
    pattern = fullfile(varargin{:});
    dirs = dir(pattern);
    path = fullfile(dirs(1).folder, dirs(1).name);
    if numel(dirs) > 1
        warning('Multiple files found matching %s; using %s', pattern, path);
    end
end

function [paths] = multiFileGlob(varargin)

    pattern = fullfile(varargin{:});
    dirs = dir(pattern);
    assert(~isempty(dirs), 'No files found matching %s', pattern);

    paths = sort(fullfile({dirs.folder}, {dirs.name}));
end
end