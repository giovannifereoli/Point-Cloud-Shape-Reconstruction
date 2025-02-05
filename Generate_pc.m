% Create pc from .obj
% TODO: add noise to the points, or random sample from obj.
clear; clc; close;

% Save as MAT
save_obj_as_pointcloud('Itokawa.obj', 'Itokawa_pc.mat'); 

% Cluster
num_clusters = 10;         % Number of clusters for K-Means
cluster_choice = 'largest'; % Options: 'largest' or specific index (e.g., 1)
extract_kmeans_cluster('Itokawa_pc.mat', 'Itokawa_cluster1.mat', num_clusters, cluster_choice);


%% Helper functions

function save_obj_as_pointcloud(obj_filename, output_filename)
    % Reads a .obj file, extracts vertex coordinates, and saves as nx3 point cloud.
    %
    % Inputs:
    %   obj_filename    - Path to the .obj file
    %   output_filename - Path to save the nx3 point cloud matrix (.txt or .mat)
    %
    % Example Usage:
    %   save_obj_as_pointcloud('model.obj', 'pointcloud.txt');

    % Open and read the .obj file
    fid = fopen(obj_filename, 'r');
    if fid == -1
        error('Cannot open file: %s', obj_filename);
    end
    
    % Initialize vertex list
    vertices = [];

    % Read file line by line
    while ~feof(fid)
        line = fgetl(fid);
        if startsWith(line, 'v ') % Only extract vertex lines
            data = sscanf(line(3:end), '%f %f %f'); % Read x, y, z
            vertices = [vertices; data']; % Append to vertex list
        end
    end
    
    % Close the file
    fclose(fid);

    % Save the point cloud
    if endsWith(output_filename, '.txt')
        writematrix(vertices, output_filename, 'Delimiter', ' ');
    elseif endsWith(output_filename, '.mat')
        save(output_filename, 'vertices');
    else
        error('Unsupported output format. Use .txt or .mat.');
    end
    
    fprintf('Point cloud saved to %s (%d points)\n', output_filename, size(vertices, 1));
end


function extract_kmeans_cluster(mat_filename, output_mat, num_clusters, cluster_choice)
    % Load the MAT file and detect point cloud variable
    data = load(mat_filename);
    varNames = fieldnames(data);

    % Find the 3D point cloud variable
    cut_pc = [];
    for i = 1:length(varNames)
        if ismatrix(data.(varNames{i})) && size(data.(varNames{i}),2) == 3
            cut_pc = data.(varNames{i});
            break;
        end
    end

    % Error if no valid point cloud is found
    if isempty(cut_pc)
        error('No valid point cloud found in %s. Check the variable names.', mat_filename);
    end

    % Apply K-Means clustering
    [idx, C] = kmeans(cut_pc, num_clusters, 'Replicates', 5);

    % Identify unique clusters
    unique_clusters = unique(idx);

    % Select cluster (by size or index)
    if strcmp(cluster_choice, 'largest')
        % Find the largest cluster
        cluster_sizes = arrayfun(@(x) sum(idx == x), unique_clusters);
        [~, max_idx] = max(cluster_sizes);
        chosen_cluster = unique_clusters(max_idx);
    else
        % Assume 'cluster_choice' is an index (1-based)
        chosen_cluster = unique_clusters(min(cluster_choice, numel(unique_clusters)));
    end

    % Extract the selected cluster
    extracted_pc = cut_pc(idx == chosen_cluster, :);

    % Save the extracted cluster
    save(output_mat, 'extracted_pc');
    disp(['Saved selected cluster to ', output_mat]);
end
