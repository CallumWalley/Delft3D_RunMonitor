function [hisFile, netFiles, mapFiles] = get_inputs(rootDir)
    % rootDir: main directory path
    arguments
        rootDir (1,:) char % or string
    end

    hisStruct = dir(fullfile(rootDir, '**', '*his.nc'));
    if numel(hisStruct) > 1
        error("too many history files found. %s", hisStruct.name)
    end
    hisFile = fullfile(hisStruct(1).folder, hisStruct(1).name);
    disp(hisFile);
    fprintf('  History File: %s', hisFile);




    netStruct = dir(fullfile(rootDir, '*net.nc'));
    netFiles = sort(fullfile({netStruct.folder}, {netStruct.name}));

    mapStruct = dir(fullfile(rootDir, '**', '*map.nc'));
    mapFiles = sort(fullfile({mapStruct.folder}, {mapStruct.name}));


    if numel(mapFiles) ~= numel(netFiles)
        warning("Mismatch in output files %d net, %d map", numel(netFiles), numel(mapFiles))
    end
    
    fprintf('%d Network Files\n', numel(netFiles));
    fprintf('%d Map Files\n', numel(mapFiles));
end