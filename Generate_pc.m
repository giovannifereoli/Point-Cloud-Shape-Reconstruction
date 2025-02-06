% Create pc from .obj
clear; clc; close;

% Save as MAT
generate_pc_measurement('Itokawa.obj', 'Itokawa_pc.mat', 0.001, 0.5); 

% Cluster
num_clusters = 10;         % Number of clusters for K-Means
cluster_choice = 'largest'; % Options: 'largest' or specific index (e.g., 1)
extract_kmeans_cluster('Itokawa_pc.mat', 'Itokawa_cluster1.mat', num_clusters, cluster_choice);


%% Helper functions

function generate_pc_measurement(obj_filename, output_filename, noise_level, downsample_ratio)
    % Reads a .obj file, extracts vertex coordinates, applies noise and downsampling, 
    % and saves as an nx3 point cloud.
    %
    % Inputs:
    %   obj_filename     - Path to the .obj file
    %   output_filename  - Path to save the nx3 point cloud matrix (.txt or .mat)
    %   noise_level      - Standard deviation of Gaussian noise (e.g., 0.001)
    %   downsample_ratio - Fraction of points to retain (0 < downsample_ratio <= 1)
    %
    % Example Usage:
    %   save_obj_as_pointcloud('model.obj', 'pointcloud.mat', 0.001, 0.5);

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

    % Add Gaussian Noise to the Point Cloud
    vertices_noisy = vertices + noise_level * randn(size(vertices));

    % Downsample Point Cloud Randomly
    num_points = size(vertices_noisy, 1);
    num_sample = round(downsample_ratio * num_points);
    idx_rand = randperm(num_points, num_sample); % Randomly select indices
    vertices_downsampled = vertices_noisy(idx_rand, :);

    % Plot the entire point cloud
    figure;
    scatter3(vertices_downsampled(:,1), vertices_downsampled(:,2), vertices_downsampled(:,3), 5, 'k.');
    title('Processed Point Cloud', 'Interpreter', 'latex', 'FontSize', 14);
    xlabel('$X$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('$Y$', 'Interpreter', 'latex', 'FontSize', 12);
    zlabel('$Z$', 'Interpreter', 'latex', 'FontSize', 12);
    grid on; axis equal; view(3);

    % Save the processed point cloud
    if endsWith(output_filename, '.txt')
        writematrix(vertices_downsampled, output_filename, 'Delimiter', ' ');
    elseif endsWith(output_filename, '.mat')
        save(output_filename, 'vertices_downsampled');
    else
        error('Unsupported output format. Use .txt or .mat.');
    end
    
    fprintf('Processed point cloud saved to %s (%d points)\n', output_filename, size(vertices_downsampled, 1));
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

    % Visualization: Plot all clusters with mean points
    figure;
    hold on;
    grid on;
    box on;
    colors = lines(num_clusters); % Use distinct colors for each cluster

    % Scatter plot of all clusters
    for i = 1:num_clusters
        cluster_points = cut_pc(idx == i, :);
        scatter3(cluster_points(:,1), cluster_points(:,2), cluster_points(:,3), 10, colors(i,:), 'filled');
    end

    % Plot centroids with larger markers
    scatter3(C(:,1), C(:,2), C(:,3), 100, 'k', 'filled', 'd'); % Centroids in black diamonds

    % Labels and formatting
    title('Point Cloud Clustering with K-Means', 'Interpreter', 'latex', 'FontSize', 14);
    xlabel('$X$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('$Y$', 'Interpreter', 'latex', 'FontSize', 12);
    zlabel('$Z$', 'Interpreter', 'latex', 'FontSize', 12);
    axis equal;
    legend(arrayfun(@(x) sprintf('Cluster %d', x), 1:num_clusters, 'UniformOutput', false), 'Location', 'best');
    hold off; grid on; view(3);
end
