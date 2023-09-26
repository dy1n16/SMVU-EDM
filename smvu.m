function [X, D, Info] = smvu(Delta, W, L, r, nu, pars)
%%
% This code for Supervised Maximum Variance Unfolding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    min_D Tr(DJLJ) + \nu \sum_{i,j} w_{ij} ( \sqrt{D_{ij}} - delta_{ij} )^2 
%    s.t.     A_{ij} \le D_{ij} \le B_{ij}            (Q1)
%
%    where  Delta -- dissimilarity matrix
%           W     -- weight matrix, needing W_ij > 0 if delta_ij > 0 
%           L     -- lable kernel matrix
%           r     -- embedding dimension
%           nu    -- balance parameter
%           J     -- centering matrix J = I - 1/n
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%     Delta:  the n x n dissimilarity matrix 
%         W:  n x n weight matrix
%         L:  label kernel matrix
%         r:  Embedding dimension  (e.g. r = 2 or 3)
%        nu:  balance parameter
%
%      pars:  parameters and other informations
%             pars.rtol  -- relative tolerance
%             pars.etol  -- tolerance for eigenvalues
%             pars.rho   -- penalty parameter
%             pars.A     -- lower bounds
%             pars.B     -- upper bound matrix
%             pars.D0    -- initial point
%             pars.itmax -- maximum iteration number (default: 1000)
%%%******************************************************
% NOTE: Make sure choose \nu such that JLJ + \nu *W >= 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                      

% Output:
%     X:  rxn matrix of the final embedding points (points in columns)
%     D:  nxn (squared) Euclidean distance matrix by the final embedding points 
%     Info (optional output)
%         Info.t -- total time
%         Info.f -- histoty of objective functions
%         Info.It    -- Number of iterations taken
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Send your comments and suggestions to              
%           hdqi@soton.ac.uk                           
%
% Warning: Accuracy may not be guaranteed!!!!!    
%          First version: 31.08.2021
%          Last updated:  31.08.2021        
%
%%%%%%%%%  Hou-Duo Qi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t0 = cputime;
n = size(Delta, 1);
deltaMax = max(Delta(:));

% if nargin < 7  % set up default parameters for pars
%     pars.rtol = 1.0e-1; pars.etol = 1.0e-2; pars.itmax = 1000;
%     pars.A = zeros(n); pars.B = sqrt(n)*deltaMax*ones(n);
%     pars.D0 = Delta.^2;
% end

if isfield(pars, 'rtol'); rtol = pars.rtol; else; rtol = 1.0e-1; end
if isfield(pars, 'etol'); etol = pars.etol; else; etol = 1.0e-4; end
if isfield(pars, 'itmax'); itmax = pars.itmax; else; itmax = 1000; end
if isfield(pars, 'A'); A = pars.A; else; A = zeros(n); end
if isfield(pars, 'B'); B = pars.B; else; B = sqrt(n)*deltaMax*ones(n); end
if isfield(pars, 'D0'); D = pars.D0; D0Max = max(pars.D0(:)); B = n*D0Max*ones(n);else; D = Delta.^2; end
%if isfield(pars, 'rho'); rho = pars.rho; else; [~,PD,~] = ProjKr(-D, r); rho = 10*norm(D+PD, 'fro');end

% compute S matrix --- centralzing L
    sumL = sum(L, 2)/n;
    S = L - (ones(n,1)*sumL' + sumL*ones(1, n)) + sum(sumL)/n;
    H = S + nu*W; % combined weights
    Delta_hat = nu * (( W .* Delta)./H); % WD = Delta .* W;
    Delta_hat(Delta_hat == inf) = 0;     %
    HD = H.*Delta_hat;
    [pidx, zidx] = get_positive_index(HD); 
%
D = min( max(D, A ), B);   % make use the initial D0 is feasible

D = min( max(D, A ), B);   % make use the initial D0 is feasible
TH = find(D>0);
df = norm(S(TH) + nu*sqrt(W(TH)).*(1-Delta(TH)./sqrt(D(TH))),'fro');
if isfield(pars, 'rho'); rho = pars.rho; 
else
    [~,PD,~] = ProjKr(-D, r); 
    rho = norm(S(TH) + nu*sqrt(W(TH)).*(1-Delta(TH)./sqrt(PD(TH))),'fro');
end

fprintf('df: %.3e rho: %.3e\n',df, rho);

% compute Frho, g, P at D
 [~, P, f, g] = frho(D, Delta, S, W, nu, r, rho); % P is the projection of -D on \K^n_+(r)
 g2 = g^2;
 ErrEig0 = (g2)/norm(JXJ(D), 'fro')^2;
 fold = f; 
 
% count iteration numbers
  it_1 = 0;
  it_2 = 0;
 
fprintf('Start to run smvu \n');
fprintf('\n-----------------------------------------------------------------\n'); 

iter = 1;
while (iter <= itmax) 
    % uppdate D
    if g == 0
       D = zeros(n);
       for i = 1:(n-1)
           for j = (i+1):n
               D(i,j) = mappingT(H(i,j), 2*nu*Delta(i,j)*W(i,j), rho, D0(i,j), A(i,j), B(i,j));
           end
       end
       D = D + D';
       it_1 = it_1 + 1;
       step_type = 1;
    else
       rhok = g/rho;
       D = dcroot_mvu(-P - rhok*H, rhok*HD, pidx, zidx);
       D  = min( max(D, A), B);
       it_2 = it_2 + 1;
       step_type = 2;
    end
    
    % compute Frho, g, P at D
    [X, P, f, g, lambda] = frho(D, Delta, S, W, nu, r, rho); % P is the projection of -D on \K^n_+(r)
    g2 = g^2; 
    ErrEig = (g2)/norm(JXJ(D), 'fro')^2;
    
    TH = find(D>0);
    df = norm(S(TH) + nu*sqrt(W(TH)).*(1-Delta(TH)./sqrt(D(TH))),'fro');
    rho_t = 1.1*df;
    if rho_t >= rho 
     rho = rho_t;
    end
    Fprog = abs( fold - f )/( 1+abs(fold));
    fprintf('Iter: %3d  ErrEig: %.3e  Fprog: %.3e  Frho: %.3e  gD: %.3e df: %.3e rho: %.3e rhok: %.3e Steptype: %3d \n', ...
        iter, ErrEig, Fprog, f, g, df, rho, rhok, step_type);
%     if (( ErrEig < etol ) || (abs(ErrEig0 - ErrEig) < 1.0e-4)) && (Fprog < rtol) 
%             break; 
%     end
    if (( ErrEig < etol ) || (abs(ErrEig0 - ErrEig) < etol)) && (Fprog < rtol) 
            break; 
    end
    
    

    %[ErrEig, etol, Fprog, rtol]
    ErrEig0 = ErrEig;
    fold = f;  
    iter = iter + 1;
end
%     [U,E]      = eig(JXJ(-D)/2);
%     % toc(tt)
%     Eig        = sort(real(diag(E)),'descend');
%     Er         = real((Eig(1:r)).^0.5); 
%     X      = (sparse(diag(Er))*real(U(:,1:r))');

Info.t = cputime - t0;
Info.rho = rho;
Info.It = iter;
Info.It1 = it_1;
Info.It2 = it_2;
Info.lambda = lambda;


% generate the final embedding
%  X = cMDS(D, r);

%%  Service Functions ------------------------------------------------------------------------
function  JXJ = JXJ( X )
% Compute J*X*J where J = I-ee'/n;
    nX   = size(X,1);
    Xe   = sum(X, 2);  
    eXe  = sum(Xe);    
    JXJ  = repmat(Xe,1,nX);
    JXJt = repmat(Xe',nX,1);
    JXJ  = -(JXJ + JXJt)/nX;
    JXJ  = JXJ + X + eXe/nX^2;
return

%% ------------------------------------------------------------------------
function [X0, Z0, lambda] = ProjKr(A, r)
% X0: embedding points of (-A) with A being close to EDM and being symmetric
%     points in columns (nxr matrix)
% Z0: projection of A on cone K_+^n(r)  
  JAJ    =  JXJ(A);
  [V0, lambda]= eigs((JAJ+JAJ')/2, r, 'LA');
  lambda = diag(lambda);
  lambda = sqrt(max(0, lambda));
% V0     = real(V0);
  X0 = V0*diag(lambda);
  Z0 = X0*X0' + A - JAJ;
  X0 = X0';
return

%% compute objective fD, gD, and the projection of -D onto \K^n_+(r) 
function [X, P, f, g, lambda] = frho(D, Delta, S, W, nu, r, rho)
    [X, P, lambda] = ProjKr(-D, r);
    g = norm(D+P, 'fro');
    f = sum(sum(D.*S)) + nu*norm(sqrt(W).*(sqrt(D)-Delta), 'fro').^2 + rho*g;
return

%% Mapping T
function [x] = mappingT(alpha, beta, gamma, eta, a, b)
    if (alpha-gamma) <= 0
        t1 = eta;
    else
        t1 = 0.25*(beta/(alpha-gamma))^2;
        t1 = max(a, min(eta, t1));
    end
    
    t2 = 0.25*(beta/(alpha+gamma))^2;
    t2 = max(eta, min(b, t2));
    
    f1 = (alpha-gamma)*t1 - beta*sqrt(t1) + gamma*eta;
    f2 = (alpha+gamma)*t2 - beta*sqrt(t2) - gamma*eta;
    
    if f1 < f2
        x = t1;
    else
        x = t2;
    end
return