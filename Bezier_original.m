% Simulation parameters
n = 6;                                          % Surface order
p = 40;                                         % 'p x p' data points
q = 200;                                        % 'q x q' query points

% Generate gridded data
x = rsrf(p);                                    % Create data to fit
[u,v] = meshgrid(0:1/(q-1):1, 0:1/(q-1):1);     % Evaluation grid

% Indice noise in data
x = x+(rand(p*p,3)-0.5)/p;                      % Noise in [x,y,z]

% Best-fitting surface
tic
[i,r1] = bsfit(x,n);                            % LLS fitting
toc
tic
[j,r2] = bsfit(x,n,[],[],'plot',1);             % NLLS fitting
toc
fplt

% Evaluate surface fit
[a,b,c] = bsval(i,u,v);                         % LLS evaluation
[d,e,f] = bsval(j,u,v);                         % NLLS evaluation

% Plot LLS results
figure
hold on
plot3(x(:,1),x(:,2),x(:,3), 'k.','markersize',3)
surf(a,b,c, 'edgecolor','none','facealpha',0.8)
title 'LLS Bezier Surface Fitting'
view(3)
fplt
xx = get(gca,'xlim');
yy = get(gca,'ylim');
zz = get(gca,'zlim');

% Plot NLS results
figure
hold on
plot3(x(:,1),x(:,2),x(:,3), 'k.','markersize',3)
surf(d,e,f, 'edgecolor','none','facealpha',0.8)
title 'NLLS Bezier Surface Fitting'
xlim(xx)
ylim(yy)
zlim(zz)
view(3)
fplt

% Generate rough surface
function x = rsrf(n)
    [x,y] = meshgrid(1:n, 1:n);
    z = ifft2(exp(-(min(x-1, n-x+1).^2 + min(y-1, n-y+1).^2)/2).*...
        fft2(randn(n)));
    z = z(:)-min(z(:));
    x = [(x(:)-1)/(n-1), (y(:)-1)/(n-1), z/max(z)];
end

% Format plots
function fplt()
    % Format axes
    h = findall(gca, 'Type','axes');
    set(h, 'box','on','Color','w')
    set(h, 'TickDir','in')
    set(h, 'TickLabelInterpreter','LaTeX')
    % Format text
    h = findall(gca, 'Type','Text');
    set(h, 'Interpreter','LaTeX')
    set(h, 'FontSize',12)
    % Format title
    h = get(gca, 'Title');
    set(h, 'FontSize',14)
end

function [idx,res,u,v,val] = bsfit(x,n,u,v,varargin)
%     [Y,R,U,V,Z] = BSFIT(X,N,U,V,'PAR1',VAL1,...,'PARN',VALN) Returns the
%     best-fitting Bézier surface control points 'Y', residuals 'R', knots
%     'U' and 'V' and surface evaluation 'Z' to the data set 'X' given the
%     surface order 'N' and, optionally, the knot vectors 'U' and 'V', and
%     a series of parameter-value combinations:
% 
%         'tol'       Convergence tolerance during NLS iterations
%         'iter'      Maximum number of Newton-Raphson iterations
%         'coef'      Relaxation coefficients of knot permutation
%         'bico'      Binomial coefficient vector of degree 'N+1'
%         'nlls'      No-input-data query for NLLS fitting method
%         'plot'      Create plot with iteration convergence logs
%     
%     If no combinations of parameter-values are parsed, a LLS approach is
%     used by default. The function is based on the following publication:
%      Lifton, J., Liu, T. & McBride, J. 'Non-Linear Least Squares Fitting
%      of Bézier Surfaces to Unstructured Point Clouds'. AIMS Mathematics,
%      2020. 6(4) 3142-3159.
%
%     See also: BSVAL, BPFIT, BPVAL, PCFIT, PCVAL, BSPL Toolbox.
    
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
    def = {min(1e-4,sum(range(x))/(6*m)),1e2,0.5,0,0,0};
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
        figure
        plot(sse(sse>0))
        xlabel 'Iteration Number'
        ylabel 'Sum of Squared Error'
        title 'NLLS convergence log'
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
%     [X,Y,Z] = BSVAL(P,U,V,B) Returns the Bézier surface evaluation 'X',
%     'Y' and 'Z' that results from the control points 'P' and the arrays
%     of knots 'U' and 'V'. An optional parameter 'B' can parse the value
%     of the binomial coefficients of order 'N+1', where 'N' is the order
%     of the surface.
% 
%     The function is based on the following research article:
% 
%     Lifton, J., Liu, T. & McBride, J. 'Non-Linear Least Squares Fitting
%     of Bézier Surfaces to Unstructured Point Clouds'. AIMS Mathematics,
%     2020. 6(4) 3142-3159.
%
%     See also: BSVAL, BPFIT, BPVAL, PCFIT, PCVAL, BSPL Toolbox.
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

% 
% function val = de_casteljau_surface(control_points, u, v)
%     % Evaluates the Bézier surface at given u, v using De Casteljau’s algorithm
%     % control_points: (n+1) x (m+1) x 3 matrix of control points
%     % u, v: Parametric coordinates (vectors)
% 
%     n = size(control_points, 1) - 1;
%     m = size(control_points, 2) - 1;
% 
%     val = zeros(length(u), 3); % Output surface points
% 
%     for k = 1:length(u)
%         u_k = u(k);
%         v_k = v(k);
% 
%         % Apply De Casteljau's algorithm in u-direction
%         temp = control_points; % Copy control points
%         for r = 1:n
%             for i = 1:(n-r+1)
%                 temp(i, :, :) = (1 - u_k) * temp(i, :, :) + u_k * temp(i+1, :, :);
%             end
%         end
%         temp = squeeze(temp(1, :, :)); % Reduce to 2D
% 
%         % Apply De Casteljau's algorithm in v-direction
%         for r = 1:m
%             for j = 1:(m-r+1)
%                 temp(j, :) = (1 - v_k) * temp(j, :) + v_k * temp(j+1, :);
%             end
%         end
% 
%         val(k, :) = temp(1, :); % Store computed surface point
%     end
% end
