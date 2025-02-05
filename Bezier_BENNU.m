%      The function is based on the following publication:
%      Lifton, J., Liu, T. & McBride, J. 'Non-Linear Least Squares Fitting
%      of Bézier Surfaces to Unstructured Point Clouds'. AIMS Mathematics,
%      2020. 6(4) 3142-3159.
clear; clc; close;

%% Load Point Cloud
pc_mat = read_pointcloud('BENNU_preTag_pc.mat');  % Load nx3 point cloud

% Flip Y and Z
x = [pc_mat(:,1), pc_mat(:,3), pc_mat(:,2)]; 

% Add Gaussian Noise to the Point Cloud
noise_level = 0.1; % Adjust noise intensity as needed
x_noisy = x + noise_level * randn(size(x));

% Downsample Point Cloud Randomly
downsample_ratio = 1; 
num_points = size(x_noisy, 1);
num_sample = round(downsample_ratio * num_points);
idx_rand = randperm(num_points, num_sample); % Randomly select indices
x = x_noisy(idx_rand, :);

% Visualize Raw Point Cloud 
figure(20); 
hold on;
grid on;
box on;
scatter3(x(:,1), x(:,2), x(:,3), 5, 'k.');
title('Loaded Point Cloud', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('$X$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$Y$', 'Interpreter', 'latex', 'FontSize', 12);
zlabel('$Z$', 'Interpreter', 'latex', 'FontSize', 12);
view(3);
axis equal;
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
xx = get(gca, 'XLim');
yy = get(gca, 'YLim');
zz = get(gca, 'ZLim');
xlim(xx);
ylim(yy);
zlim(zz);

%% Training

% Determine Fitting Parameters
n = 12;                     % Bézier surface order (adjust as needed)
p = round(sqrt(size(x,1))); % Estimate grid size based on point count
q = 200;                   % Number of evaluation points

% Generate Normalized Knot Grid
[u, v] = meshgrid(linspace(0,1,q)); % Evaluation grid

% Best-fitting surface
tic
[i,r1] = bsfit(x,n);                            % LLS fitting
toc
tic
[j,r2] = bsfit(x,n,[],[],'plot',1);             % NLLS fitting
toc

% Evaluate surface fit
[a,b,c] = bsval(i,u,v);                         % LLS evaluation
[d,e,f] = bsval(j,u,v);                         % NLLS evaluation

%% Plots

% Plot LLS results
figure(1);
hold on;
grid on;
box on;
scatter3(x(:,1), x(:,2), x(:,3), 5, 'k.');
surf(a, b, c, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
title('LLS Bezier Surface Fitting', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('$X$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$Y$', 'Interpreter', 'latex', 'FontSize', 12);
zlabel('$Z$', 'Interpreter', 'latex', 'FontSize', 12);
view(3);
axis equal;
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
xx = get(gca, 'XLim');
yy = get(gca, 'YLim');
zz = get(gca, 'ZLim');
xlim(xx);
ylim(yy);
zlim(zz);

% Plot NLS results
figure(2);
hold on;
grid on;
box on;
scatter3(x(:,1), x(:,2), x(:,3), 5, 'k.');
surf(d, e, f, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
title('NLLS Bezier Surface Fitting', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('$X$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$Y$', 'Interpreter', 'latex', 'FontSize', 12);
zlabel('$Z$', 'Interpreter', 'latex', 'FontSize', 12);
view(3);
axis equal;
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
xx = get(gca, 'XLim');
yy = get(gca, 'YLim');
zz = get(gca, 'ZLim');
xlim(xx);
ylim(yy);
zlim(zz);

%% Helper functions

% Generate rough surface
function pointcloud = read_pointcloud(file_name)
    if endsWith(file_name, '.txt')
        pointcloud = readmatrix(file_name);
    elseif endsWith(file_name, '.mat')
        data = load(file_name);
        pointcloud = data.vertices;
    else
        error('Unsupported file format. Use .txt or .mat.');
    end

    % Check if the loaded data is valid
    if size(pointcloud, 2) ~= 3
        error('Invalid point cloud format. Expected an nx3 matrix.');
    end

    fprintf('Loaded point cloud from %s (%d points)\n', file_name, size(pointcloud, 1));
end

function [idx,res,u,v,val] = bsfit(x,n,u,v,varargin)
%         'tol'       Convergence tolerance during NLS iterations
%         'iter'      Maximum number of Newton-Raphson iterations
%         'coef'      Relaxation coefficients of knot permutation
%         'bico'      Binomial coefficient vector of degree 'N+1'
%         'nlls'      No-input-data query for NLLS fitting method
%         'plot'      Create plot with iteration convergence logs

    % Get surface size and order
    m = size(x,1);
    if nargin<2 || isempty(n)
        n = 4;
    end
    
    % Get pairs of parameter-values
    if nargin>4
        flag = true;
        [varargin{:}] = convertStringsToChars(varargin{:});
    else
        flag = false;
    end
    
    % Set optional parameters
    arg = {'tol','iter','coef','bico','nlls','plot'};
    def = {min(1e-4,sum(range(x))/(6*m)), 2, 0.5,0,0,0};
    [tol,imax,rlx,b,~,fig] = internal.stats.parseArgs(arg,def,varargin{:});
    
    % Initialise U knots
    if nargin<3 || isempty(u)
        u = x(:,1);
        u = u-min(u);
        u = u/max(u);
    else
        u = double(u(:));
        u = u-min(u);
        u = u/max(u);
    end
    
    % Initialise V knots
    if nargin<4 || isempty(v)
        v = x(:,2);
        v = v-min(v);
        v = v/max(v);
    else
        v = double(v(:));
        v = v-min(v);
        v = v/max(v);
    end
    
    % Initialise parameters
    p = n+1;
    q = p:-1:1;
    r = 0:n;
    s = n-r;
    
    % Initialise binomial coefficients
    if b==0
        b = bctp(p);
    else
        b = reshape(b.*b',1,[]);
    end
    
    % Initialise basis functions
    y = 1-u;
    z = 1-v;
    c(:,p) = 0*u+1;
    d = c;
    e = c;
    f = d;
    
    % Create basis functions
    for i = n:-1:1
        c(:,i) = c(:,i+1).*u;
        d(:,i) = d(:,i+1).*v;
        e(:,i) = e(:,i+1).*y;
        f(:,i) = f(:,i+1).*z;
    end
    
    % Assemble Bernstein matrix
    a = b.*reshape(repmat(c(:,q).*e,p,1),m,[]).*repmat(d(:,q).*f,1,p);
    
    % Solve linear system
    idx = a'*a\(a'*x);
    res = a*idx-x;
    
    % Exit criterion for LLS case
    if ~flag
        if nargout>1
            res = sum(sqrt(sum(res.*res,2)))/m;
            if nargout>4
                val = a*idx;
            end
        end
        return
    end
    
    % Initialise convergence parameters
    t = 0;
    iter = 0;
    if fig~=0
        sse = zeros(imax+1,1);
        sse(1) = sum(res.*res,'all');
    end
    
    % Newton-Raphson iterations
    while norm(t-res)/max(1,norm(res))>tol
        
        % Initialise basis functions
        g = c*0;
        g(:,2) = 1;
        h = g;
        j = g;
        k = g;
        
        % Create basis functions
        for i = 3:p
            g(:,i) = g(:,i-1).*u;
            h(:,i) = h(:,i-1).*v;
            j(:,i) = j(:,i-1).*y;
            k(:,i) = k(:,i-1).*z;
        end
        
        % Calculate U-derivative
        du = b.*reshape(repmat(r.*g.*e-c(:,q).*s.*j(:,q),p,1),m,[]).*...
             repmat(d(:,q).*f,1,p)*idx;
        
        % Calculate V-derivative
        dv = b.*reshape(repmat(c(:,q).*e,p,1),m,[]).*repmat(r.*h.*f-...
             s.*d(:,q).*k(:,q),1,p)*idx;
        
        % Produce U-permutation
        u = u-rlx*sum(du.*res,2)./sum(du.*du,2);
        u = u-min(u);
        u = u/max(u);
        
        % Produce V-permutation
        v = v-rlx*sum(dv.*res,2)./sum(dv.*dv,2);
        v = v-min(v);
        v = v/max(v);
        
        % Initialise basis functions
        y = 1-u;
        z = 1-v;
        c(:,p) = 0*u+1;
        d = c;
        e = c;
        f = d;
        
        % Create basis functions
        for i = n:-1:1
            c(:,i) = c(:,i+1).*u;
            d(:,i) = d(:,i+1).*v;
            e(:,i) = e(:,i+1).*y;
            f(:,i) = f(:,i+1).*z;
        end
        
        % Assemble Bernstein matrix
        a = b.*reshape(repmat(c(:,q).*e,p,1),m,[]).*repmat(d(:,q).*f,1,p);
        
        % Solve linear system
        idx = a'*a\(a'*x);
        t = res;
        res = a*idx-x;
        
        % Track convergence logs
        if fig~=0
            sse(iter+1) = sum(res.*res,'all');
        end
        
        % Exit criterion
        iter = iter+1;
        if iter==imax
            break
        end
    end
    
    % Prepare outputs
    if nargout>1
        res = sum(sqrt(sum(res.*res,2)))/m;
        if nargout>4
            val = a*idx;
        end
    end
    
    % Generate residual plot
    if fig~=0
        figure(10);
        semilogy(sse(sse>0), 'LineWidth', 1.5, 'Color', 'b'); % Log scale for better visualization
        xlabel('Iteration Number', 'Interpreter', 'latex', 'FontSize', 12);
        ylabel('Sum of Squared Error (SSE)', 'Interpreter', 'latex', 'FontSize', 12);
        title('NLLS Convergence Log', 'Interpreter', 'latex', 'FontSize', 14);
        grid on;
        box on;
        set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'XColor', 'k', 'YColor', 'k', ...
        'YScale', 'log'); % Log scale on the Y-axis for better convergence visualization
        xlim([1 length(sse(sse>0))]);
        ylim([min(sse(sse>0))*0.9, max(sse(sse>0))*1.1]);
    end
end

% Binomial coefficient tensor product
function b = bctp(n)
    % Iterative sums
    if n<30
        b(n,n) = 1;
        for i = 1:n
            j = i-1;
            b(i) = 1;
            b(i,i) = 1;
            for k = 2:j
                b(i,k) = b(j,k-1)+b(j,k);
            end
        end
        b = b(n,:);
        
    % Symmetric products
    else
        m = n-1;
        b(n) = 1;
        b(m) = m;
        for i = 3:ceil(0.5*n)
            k = m;
            for j = 2:i-1
                k = k*(n-j)/j;
            end
            b(n-j) = k;
        end
        b(1:i) = b(n:-1:n-j);
    end
    
    % Linear form
    b = reshape(b.*b',1,[]);
end

function [x,y,z] = bsval(x,u,v,b)
    % Initialise dimensions
    m = size(x,1);
    n = sqrt(m);
    
    % Parse knot vectors
    if nargin < 3 || isempty(u) || isempty(v)
        [u,v] = meshgrid(0:1/99:1, 0:1/99:1);
        u = u(:);
        v = v(:);
        p = 1e4;
    else
        u = double(u(:));
        v = double(v(:));
        p = length(u);
        
        % Correct outlying values
        if any(u<0) || any(u>1)
            u = u-min(u);
            u = u/max(u); 
        end
        if any(v<0) || any(v>1)
            v = v-min(v);
            v = v/max(v); 
        end
    end
    
    % Parse binomial coefficient
    if nargin<4 || isempty(b)
        b = bctp(n);
    else
        b = reshape(b.*b',1,[]);
    end
    
    % Prepare basis functions
    uu = 1-u;
    vv = 1-v;
    u1(:,n) = 0*u+1;
    v1 = u1;
    u2 = u1;
    v2 = u1;
    
    % Create basis functions
    for i = n-1:-1:1
        u1(:,i) = u1(:,i+1).*u;
        v1(:,i) = v1(:,i+1).*v;
        u2(:,i) = u2(:,i+1).*uu;
        v2(:,i) = v2(:,i+1).*vv;
    end
    
    % Assemble Bernstein matrix
    x = b.*reshape(repmat(u1(:,n:-1:1).*u2,n,1),p,m).*...
        repmat(v1(:,n:-1:1).*v2,1,n)*x;
    
    % Format outputs
    if nargout>1
        n = sqrt(length(u));
        if n==fix(n)
            z = reshape(x(:,3),n,n);
            y = reshape(x(:,2),n,n);
            x = reshape(x(:,1),n,n);
        else
            z = x(:,3);
            y = x(:,2);
            x = x(:,1);
        end
    end
end
