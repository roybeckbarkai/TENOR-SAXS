function [g_rat,RG2,Pxn,res,actualPSF]=MG_extract(Pxn,q_mat_x,q_mat_y,I_mat,signum,RG2,use_r3,use_g3,WAVELENGTH,QRG_MAX,DEADPIX_FACTOR)
%% Find the M and G coeffs and R_g for a given intensity map
%Pxn is the 4 pixel numbers defining the 2 assymetric PSFs
% [g_rat,RG2,Pxn,res,actualPSF]=MG_extract(Pxn,q_mat_x,q_mat_y,I_mat,signum,RG2,use_r3,use_g3,WAVELENGTH,QRG_MAX,DEADPIX_FACTOR)
% input: meshgrid of q on the detector, x- and y-values in q_mat_x and
% signum= default 4; % number of stds in the gaussian filter (PSF)
% q_mat_y (nm-1), respectively, and intensity matrix I_mat
% use_r3 says the fit for M is cubic or quadratic
% use_g3 says the fit for G is cubic or quadratic
% WAVELENGTH (nm) is needed to make the q->angle conversion for a correct
% filtering of the PSF (independent of location in detector).
% QRG_MAX (default 0.79) is the upper Guinier edge: the analysis window
% only includes pixels with q*sqrt(R_g_nofilt) < QRG_MAX.
% DEADPIX_FACTOR (default 2) is the detector-border margin, in units of
% max(Pxn) pixels, excluded from the fit on each side.

% result matrix g_rat shows :
% 1: g1/g0
% 2-3: 95% confidence interval for that
% 4: m2/m1
% 5-6: 95% confidence interval for that
% 7: g2/g1
% 8: estimated Guinier radius- theoretically r_g^2*(1+V). the average of the
% fits for the 2 PSFs
% 9-10: 95% confidence interval for g2/g1
% RG2 is also the estimated square Guinier radius squared- theoretically r_g^2*(1+V). for
% the unfiltered I_mat

% default fit for M is quadratic or cubic

gridded_filtering=0; % do not assume the qx and qy are meshgrids, smooth according to solid angle PSF

if nargin < 7 || isempty(use_r3)
    use_r3 = false;  % was true; every documented entry point already forces
end                  % quadratic (init_TENOR_params.m/the Python port), so
                     % only a direct low-arg-count call hit the cubic default

if nargin < 8 || isempty(use_g3)
    use_g3 = false;  % was true; see use_r3 above
end

if nargin < 9 || isempty(WAVELENGTH)
    WAVELENGTH = 0.1; %nm
end

if nargin < 10 || isempty(QRG_MAX)
    QRG_MAX = 0.79;  % upper Guinier edge, q*Rg_apparent < QRG_MAX
end

if nargin < 11 || isempty(DEADPIX_FACTOR)
    DEADPIX_FACTOR = 2;  % border margin in units of max(Pxn) pixels
end



qvr=hypot(q_mat_x,q_mat_y);
try 
    RG2=RG2(1);
    if ~isfinite(RG2) || RG2<=0
        RG2=-3*best_origin_quad_b_faster_bins(qvr.^2, log(I_mat)); %r_g^2*(1+V)
    end
catch
    RG2=-3*best_origin_quad_b_faster_bins(qvr.^2, log(I_mat)); %r_g^2*(1+V)
end
if ~exist('signum','var')
    signum=4; % number of stds in the gaussian filter (PSF)
end
try
    Pxn=Pxn(1:4);
    if (max((Pxn-1)/2~=uint8((Pxn-1)/2)) || ... %Pix num should be odd
            (isempty(setdiff(Pxn(1:2),Pxn(3:4)))) || ... % two pixn couples should be different
            ((Pxn(1) == Pxn(2)) && (Pxn(1)*Pxn(2)~=1)) || ...
            ((Pxn(3) == Pxn(4)) && (Pxn(3)*Pxn(4)~=1)) ) % none of the couples should be symmetric (except for [1 1])
        clear Pxn
    end
    % end
catch
    clear Pxn
end
I_mat=single(I_mat);
qrng=[5*min(hypot(q_mat_x(:),q_mat_y(:))) max(hypot(q_mat_x(:),q_mat_y(:)))]; % in 1/nm
% qv=linspace(-1/r_g,1/r_g,101);
qvx=q_mat_x;
qvy=q_mat_y;
qvr=sqrt(qvx.^2+qvy.^2);
maxq=max(qvr(:));
g_rat=[];
pxx=5;
pxy=1;
% PSF pixel nums pxx and pxy must be odd!
pxx=pxx+2*(pxx==pxy);
if exist("Pxn","var")
    pxx=Pxn(3);
    pxy=Pxn(4);
end

%         H=ones(pxx,pxy)';
%         H=exp(-linspace(-2,2,pxy).^2/2)'*exp(-linspace(-2,2,pxx).^2/2); %gaussian
H=exp(-linspace(-signum,signum,pxy).^2/2)'*exp(-linspace(-signum,signum,pxx).^2/2); %gaussian
H=single(H/sum(H(:))); %PSF shape
if gridded_filtering
    F2=filter2(H,I_mat,"same"); %applying the PSF
else
    F2=filter2_ungridded( ...
        H, I_mat, qvx, qvy, 1, [], [], 0, WAVELENGTH); %applying the PSF ungridded
end
temp=polyfit (1:pxx,log(H(round(pxy/2),:)),2);temp=sqrt(-1/2/temp(1));
actualPSF(3)=temp;
temp=polyfit (1:pxy,log(H(:,round(pxx/2))),2);temp=sqrt(-1/2/temp(1));
actualPSF(4)=temp;
% % Zero-pad kernel to image size once
% KH = fft2( H, size(I_mat,1), size(I_mat,2) );   % H = gy(:) * gx(:).'
% FI = fft2(I_mat);
% F2  = real(ifft2(FI .* KH));
% gx = exp(-linspace(-signum,signum,pxx).^2/2);  gx = gx / sum(gx);
% gy = exp(-linspace(-signum,signum,pxy).^2/2);  gy = gy / sum(gy);
% F2=imfilter(imfilter(I_mat, gy(:),'replicate','same'), ...
%                     gx(:).','replicate','same');


% sigma_pixels_x = (pxx-1)/(2*signum);  % since your grid spans ±signum
% sigma_pixels_y = (pxy-1)/(2*signum);
%
% F2 = imgaussfilt(I_mat, [sigma_pixels_y, sigma_pixels_x], ...
%                 'FilterSize', [pxy pxx], ...
%                 'Padding', 'replicate');
opx=[pxx pxy];
% pxx=pxx+2;
% pxy=pxy+8;

pxx=1;pxy=9;
if exist("Pxn","var")
    pxx=Pxn(1);
    pxy=Pxn(2);
end

H=exp(-linspace(-signum,signum,pxy).^2/2)'*exp(-linspace(-signum,signum,pxx).^2/2); %gaussian
H=single(H/sum(H(:))); %PSF shape
if gridded_filtering
    F=filter2(H,I_mat,"same"); %applying the PSF
else
    F=filter2_ungridded( ...
        H, I_mat, qvx, qvy, 1, [], [], 0, WAVELENGTH); %applying the PSF ungridded
end
temp=polyfit (1:pxx,log(H(round(pxy/2),:)),2);temp=sqrt(-1/2/temp(1));
actualPSF(1)=temp;
temp=polyfit (1:pxy,log(H(:,round(pxx/2))),2);temp=sqrt(-1/2/temp(1));
actualPSF(2)=temp;

dqpix=mean(diff((qvy(:,1)))); %q diff between pixels, assuming qy and qx spacing are the same
actualPSF=actualPSF*dqpix; % converting the actualPSF from pixels to q (1/nm)
% sigma_pixels_x = (pxx-1)/(2*signum);  % since your grid spans ±signum
% sigma_pixels_y = (pxy-1)/(2*signum);
%
% F = imgaussfilt(I_mat, [sigma_pixels_y, sigma_pixels_x], ...
%                 'FilterSize', [pxy pxx], ...
%                 'Padding', 'replicate');

% % Zero-pad kernel to image size once
% KH = fft2( H, size(I_mat,1), size(I_mat,2) );   % H = gy(:) * gx(:).'
% FI = fft2(I_mat);
% F  = real(ifft2(FI .* KH));
% gx = exp(-linspace(-signum,signum,pxx).^2/2);  gx = gx / sum(gx);
% gy = exp(-linspace(-signum,signum,pxy).^2/2);  gy = gy / sum(gy);
% F=imfilter(imfilter(I_mat, gy(:),'replicate','same'), ...
%                     gx(:).','replicate','same');
%% extract the guinier radius (r_g^2(1+V))
if 0,
    rng=find(qvr<dqpix*45& qvr>0*qrng(1)/3);
    [~,temp]=sort(qvr(rng));
    rng=rng(temp); %sorted by q
    npoly=2;
    [temp,S]=polyfit(qvr(rng).^2,log(F(rng)),npoly);
    [R, df] = deal(S.R, S.df);


    [temp2,S]=polyfit(qvr(rng).^2,log(F2(rng)),npoly);
    [temp3,S]=polyfit(qvr(rng).^2,log(I_mat(rng)),npoly);
    covp = (S.normr^2 / df) * inv(R)*inv(R)';   % covariance matrix
    alpha = 0.05;                     % 95% CI
    tval = 1.96;
    se_slope = sqrt(covp(end-1,end-1));   % std error of slope coeff
    CI_slope = [temp3(end-1) - tval*se_slope, temp3(end-1) + tval*se_slope];
end
%
%         figure(8)
%         plot(qvr(rng).^2,log(F(rng)),'o',qvr(rng).^2,log(F2(rng)),'o',qvr(rng).^2,polyval(temp,qvr(rng).^2),'-',qvr(rng).^2,polyval(temp2,qvr(rng).^2),'-')
%         hold on
%         plot(qvr(rng).^2,log(I_mat(rng)),'.',qvr(rng).^2,polyval(temp3,qvr(rng).^2),'-')
%         hold off
%         title(['guinier plot (lnI vs q^2) ' num2str(sqrt(-3*[temp(2) temp2(2) temp3(2)]))])
%         disp(['guinier plot (lnI vs q^2) ' num2str(sqrt(-3*[temp(end-1) temp2(end-1) temp3(end-1)]))])


% R_g_est=-3/2*(temp(end-1)+temp2(end-1));  %r_g^2*(1+V)
R_g_est=RG2;
R_g_nofilt=RG2;  %r_g^2*(1+V)
Pxn=[ pxx pxy opx];
% deadpix=sum(Pxn);%max(size(H))*2;
deadpix=DEADPIX_FACTOR*max(Pxn);%max(size(H))*2;  % was hard-coded 2*max(Pxn)
% disp((max(Pxn)/signum*dqpix*R_g_est)^2*2/3)


% DEAD CODE (kept for provenance only): the three assignments below were
% overwritten immediately and never took effect. The live window is the
% single QRG_MAX assignment that follows them.
% qrng=([0*deadpix*dqpix min((maxq-deadpix*dqpix),1.4765/sqrt(R_g_nofilt))]);
% qrng=([0*deadpix*dqpix min((maxq-deadpix*dqpix),1.35/sqrt(R_g_nofilt))]);
% qrng=([0*deadpix*dqpix min((maxq-deadpix*dqpix),0.9/sqrt(R_g_nofilt))]);
% --- LIVE analysis window -------------------------------------------------
% Relevant range avoiding the detector boundaries when filtering. A lower
% upper limit is more accurate for a real form-factor (with 3rd order phi''').
qrng=([0*deadpix*dqpix min((maxq-deadpix*dqpix),QRG_MAX/sqrt(R_g_nofilt))]);  % was hard-coded 0.79



if diff(qrng)<0 % not enough pixels
    error('not enough pixels- consider using a smaller slit')
    qrng(2)=0.9/sqrt(R_g_nofilt);
end

% flattening w/o averaging
qvt=atan2(qvy,qvx);

G2f=log(F2(:)./F(:));

% now we fit the ln difference to a function of the form
% g_0+g_1*Q+g_2*Q^2+cos(2theta)*Q*(m_0+m_1*Q+m_2*Q^2)
% where Q=q^2*r_g^2
rng=find(qvr<qrng(2)& qvr>qrng(1));
% res = fit_I_r_theta_ratios(qvr(rng).^2*r_g^2,qvt(rng),G2f(rng));
% res = fit_I_r_theta_ratios(qvr(rng).^2,qvt(rng),G2f(rng));
warning('error', 'MATLAB:singularMatrix');  % treat that warning as error
warning('error', 'MATLAB:nearlySingularMatrix');  % treat that warning as error
RG2=R_g_est;
try
    % res = fit_I_r_theta_ratios_weighted_centered(qvr(rng).^2,qvt(rng),G2f(rng),0*ones(size(rng))+1*sqrt(I_mat(rng)));
%     res = fit_I_r_theta_ratios_weighted_centered(qvr(rng).^2,qvt(rng),double(G2f(rng)),double(0*ones(size(rng))+1*sqrt(I_mat(rng))),use_r3,use_g3);
    res = fit_I_r_theta_ratios_weighted_centered(qvr(rng).^2,qvt(rng),double(G2f(rng)),double(0*ones(size(rng))+(I_mat(rng))),use_r3,use_g3);  % july 2026 weighting by the intensity not its square
    g_coeff=res.p(1:3);
    m_coeff=res.p(4:5);
    g_rat=double([g_coeff(2)/g_coeff(1), res.CI95_G(1), res.g100_ratio, m_coeff(2)/m_coeff(1), res.CI95_M(1), res.g210_ratio,g_coeff(3)/g_coeff(2), res.g100_CI95(1) res.g21_CI95(1) res.g210_CI95(1)]');
%replacing m with m210
    g_rat=double([g_coeff(2)/g_coeff(1), res.CI95_G(1), res.g100_ratio, res.m210_ratio, res.m210_CI95(1), res.g210_ratio,g_coeff(3)/g_coeff(2), res.g100_CI95(1) res.g21_CI95(1) res.g210_CI95(1)]');

    %     g_rat=double([g_coeff(2)/g_coeff(1), res.CI95_G(1), res.CI95_G(2)*1, m_coeff(2)/m_coeff(1), res.CI95_M(1), res.CI95_M(2)*1,g_coeff(3)/g_coeff(2), R_g_est*1 res.g21_CI95]');
catch
    g_rat=nan*(ones(10,1));
end
RG2=double(R_g_nofilt);
%res.p is the coefficient vector [g0 g1 g2 m0 m1 m2]

if 0
    figure(1)
    r_g=1;
    rng=find(qvr<qrng(2)& qvr>qrng(1));
    p=res.p;
    r0 = linspace(qrng(1),qrng(2),50).^2*r_g^2;
    tm=qvr(rng).^2*r_g^2;
    Gfit = p(1) + p(2)*r0 + p(3)*r0.^2;
    Mfit = p(4) + p(5)*r0 ;
    if length(p)>5
    Mfit = Mfit + p(6)*r0.^2;
    plot3(cos(2*qvt(rng)),qvr(rng).^2*r_g^2,G2f(rng),'+',cos(2*qvt(rng)),tm,p(1) + p(2)*tm + p(3)*tm.^2+tm.*(p(4) + p(5)*tm + p(6)*tm.^2).*cos(2*qvt(rng)),'go',zeros(size(r0)),r0,Gfit,'b',ones(size(r0)),r0,Gfit+r0.*Mfit,'r',-ones(size(r0)),r0,Gfit-r0.*Mfit,'r')
    else
    plot3(cos(2*qvt(rng)),qvr(rng).^2*r_g^2,G2f(rng),'k.',cos(2*qvt(rng)),tm,p(1) + p(2)*tm + p(3)*tm.^2+tm.*(p(4) + p(5)*tm ).*cos(2*qvt(rng)),'go',zeros(size(r0)),r0,Gfit,'b',ones(size(r0)),r0,Gfit+r0.*Mfit,'r',-ones(size(r0)),r0,Gfit-r0.*Mfit,'r')
    end
    xlabel('cos(2\chi)')
    ylabel('Q')
    legend('data','full fit','G(Q) fit','G(Q)\pmM(Q) fit')
    grid on
end


% for M=[0 2]
%     figure(2+M)
%     for i=1:inu
%         h2=plot(ax,g_rat(1+1.5*M,((i-1)*iV+1):(i*iV)),'v:','displayname',['\phi''''=' num2str(nulist(i))]);
%
%         for tmp=1:length(h2)
%             %             h2(tmp).Annotation.LegendInformation.IconDisplayStyle = 'off';
%         end
%         for tmp=1:length(h2)
%             h2(tmp).Color=h2(1).Color;
%         end
%         hold on
%         nu=nulist(i);
%
%         xlabel ('V')
%         if ~M
%             ylabel('g_1/R_0^2/g_0')
%         else
%             ylabel('m_2/R_0^2/m_1')
%
%         end
%         hold on
%     end
%     if  flag_show_legend,
%         for i=[2 4]
%             figure (i)
%             h=legend;
%             %             set(h,'location','bestoutside')
%             set(h,'location','best')
%         end
%     end
% end
end
%%
function R = fit_I_r_theta_ratios_weighted_centered(r, th, I, weight, use_r3, use_g3)
% Weighted fit with per-ratio precision grades based on CI95 width
% Model:
%   I(r,th) = G(r) + M(r)*cos(2*th)
% G(r) = g0 + g1*r + g2*r^2 (optional + g3*r^3 if use_g3)
% M(r) = m0*r + m1*r^2 (optional + m2*r^3 if use_r3)
%
%   use_r3: include r^3 .* c2 in M-block (default true)
%   use_g3: include rc.^3 in G-block (default false)

% This version:
%   * centers r in the G(r) block to reduce collinearity (safe — model class unchanged)
%   * uses a QR-based weighted least squares (stable; avoids XtWX normal equations)
%   * maps centered coefficients back to the original parameterization for outputs

% defaults
if nargin < 5 || isempty(use_r3), use_r3 = true; end
if nargin < 6 || isempty(use_g3), use_g3 = false; end


% -------- setup / guards -------------------------------------------
warning('error', 'MATLAB:singularMatrix');        % keep as hard errors
warning('error', 'MATLAB:nearlySingularMatrix');

r   = r(:);    th = th(:);    I = I(:);    weight = weight(:);
c2  = cos(2*th);
valid = isfinite(I) &  isfinite(r) & isfinite(th) & isfinite(weight);
if ~any(valid), 
    error('No valid samples.'); 
end

r = r(valid);  I = I(valid);  c2 = c2(valid);  w = weight(valid);

% -------- weighted centering for G(r) only -------------------------
% IMPORTANT: We center r in the *G-block* (1, r, r^2) only.
% Centering the c2* block would introduce a pure c2 column (i.e., change model class).
w_sum = sum(w);
mu_r  = sum(w .* r) / w_sum;      % weighted mean (more appropriate than unweighted here)
rc    = r - mu_r;                 % centered r for G

% % -------- build design (mixed basis: centered G, original M) -------
% % G-part (centered): [1, rc, rc.^2]
% % M-part (original): [r.*c2, r.^2.*c2, r.^3.*c2]
% if use_r3
%     % 6-column model
%     X = [ ones(size(r)), rc, rc.^2, r.*c2, r.^2.*c2, r.^3.*c2 ];
% else
%     % 5-column model
%     X = [ ones(size(r)), rc, rc.^2, r.*c2, r.^2.*c2 ];  %only second order in M
% end
% Build design matrix: G-block first, M-block after
if use_g3
    Gcols = [ ones(size(r)), rc, rc.^2, rc.^3 ];   % 4 cols
else
    Gcols = [ ones(size(r)), rc, rc.^2 ];          % 3 cols
end

if use_r3
    Mcols = [ r.*c2, r.^2.*c2, r.^3.*c2 ];         % 3 cols
else
    Mcols = [ r.*c2, r.^2.*c2 ];                   % 2 cols
end

X = [Gcols, Mcols];
y = I;

% -------- stable weighted least squares via QR ---------------------
% Solve argmin || sqrt(w).*(X*p_c - y) ||_2
sw = sqrt(w);
Xw = X .* sw;
yw = y .* sw;

% Economy QR; handles near-collinearity without forming XtWX
[Q,Rq] = qr(Xw, 0);
% If R is rank-deficient, use SVD least-squares as a fallback
rankX = rank(Rq);
k = size(X,2);
if rankX < k
    % SVD fallback (stable minimum-norm LS solution)
    [U,S,V] = svd(Xw, 'econ');
    s = diag(S);
    tol = max(size(Xw)) * eps(s(1));
    sInv = diag( 1 ./ s .* (s > tol) );
    p_c = V * sInv * (U' * yw);
else
    p_c = Rq \ (Q' * yw);
end

% -------- residuals / variance & covariance (stable via Rq) ---------
yhat = X * p_c;
res  = y - yhat;
N    = size(X,1);
SSE  = sum(w .* (res.^2));
dof  = max(N - k, 1);
s2   = SSE / dof;

% Covariance of p_c using Rq (no explicit inverse of XtWX)
% cov(p_c) = s2 * inv(Rq)' * inv(Rq)
Rinv  = Rq \ eye(k);              % solves Rq*Rinv = I
covPc = s2 * (Rinv * Rinv.');

% % -------- map back to original parameterization --------------------
% % Our p_c is for [1, rc, rc^2, r*c2, r^2*c2, r^3*c2].
% % We need p (original) for [1, r, r^2, r*c2, r^2*c2, r^3*c2].
% % For G block (first 3), with r = rc + mu_r:
% %   g2 = g2c
% %   g1 = g1c - 2*mu_r*g2c
% %   g0 = g0c - mu_r*g1 - mu_r^2*g2
% % M block (last 3) is already in original r-powers.
% T = eye(6);
% T(1,1) = 1;       T(1,2) = -mu_r;     T(1,3) =  mu_r^2;
% T(2,1) = 0;       T(2,2) =  1;        T(2,3) = -2*mu_r;
% T(3,1) = 0;       T(3,2) =  0;        T(3,3) =  1;
% % lower-right 3x3 is identity
% 
% T=T(1:rankX,1:rankX);  %cut T in case we take only 2 M coefficients


% Build transform T mapping centered G-block back to original r-powers
% We'll map G block [1, rc, rc^2, (rc^3)] to [1, r, r^2, (r^3)].
% For rc^3 present: r = rc + mu: r^3 = rc^3 + 3*mu*rc^2 + 3*mu^2*rc + mu^3
% Build Tfull for maximum size 7 (4 G + 3 M). We'll then truncate.
Tfull = eye(7);
% G block mapping (rows = target [1 r r^2 r^3], cols = [1 rc rc^2 rc^3])
% Fill first 4x4:
Tfull(1,1) = 1;    Tfull(1,2) = -mu_r;      Tfull(1,3) =  mu_r^2;       Tfull(1,4) = -mu_r^3;
Tfull(2,1) = 0;    Tfull(2,2) = 1;           Tfull(2,3) = -2*mu_r;      Tfull(2,4) = 3*mu_r^2;
Tfull(3,1) = 0;    Tfull(3,2) = 0;           Tfull(3,3) = 1;            Tfull(3,4) = -3*mu_r;
Tfull(4,1) = 0;    Tfull(4,2) = 0;           Tfull(4,3) = 0;            Tfull(4,4) = 1;
% M block columns are already in original r powers; they map identity in their positions.
% Determine how many columns total:
kG = size(Gcols,2);
kM = size(Mcols,2);
kTot = kG + kM;

% Build final T of size kTot x kTot by taking top kTot rows and cols from Tfull,
% but place the identity for M-block at the right position.
T = zeros(kTot);
% Insert G mapping: target G rows are 1:kG map from centered G cols 1:kG
T(1:kG,1:kG) = Tfull(1:kG,1:kG);
% Insert identity for M-block
T(kG+1:end, kG+1:end) = eye(kM);


p = T * p_c;

% Covariance transform: covP = T * covPc * T'
covP = T * covPc * T.';

% -------- ratios & SEs (same as before, but using covP) -----------
g0=p(1); g1=p(2); g2=p(3);
m0=p(4); m1=p(5); 
try m2=p(6);
catch
    m2=nan;
end
if kG >= 4, g3 = p(4); else g3 = NaN; end

g_ratio = g1 / g0;
m_ratio = m1 / m0;

dg = zeros(1,rankX); dg(1) = -g1/g0^2; dg(2) = 1/g0;
dm = zeros(1,rankX); dm(4) = -m1/m0^2; dm(5) = 1/m0;

% Guards against tiny negatives from roundoff
se_g = sqrt(max(dg * covP * dg.', 0));
se_m = sqrt(max(dm * covP * dm.', 0));

z95  = 1.95996398454005;
g_CI = g_ratio + z95*se_g*[-1 1];
m_CI = m_ratio + z95*se_m*[-1 1];

g_hw = z95*se_g;    m_hw = z95*se_m;
g_w  = 2*g_hw;      m_w  = 2*m_hw;

grade_g = 1 ./ (1 + g_w);
grade_m = 1 ./ (1 + m_w);

% -------- weighted R^2 ---------------------------------------------
ybar_w = sum(w .* y) / sum(w);
SSTw   = sum(w .* (y - ybar_w).^2);
R2w    = 1 - SSE / SSTw;

% -------- pack outputs (same schema) -------------------------------
R.p        = p.';                    % [g0 g1 g2 m0 m1 m2] in ORIGINAL basis
R.g_ratio  = g_ratio;   R.m_ratio  = m_ratio;
R.g_CI95   = g_CI;      R.m_CI95   = m_CI;
R.g_SE     = se_g;      R.m_SE     = se_m;
R.g_CI95W  = g_w;       R.m_CI95W  = m_w;
R.g_CI95HW = g_hw;      R.m_CI95HW = m_hw;
R.grade_g  = grade_g;   R.grade_m  = grade_m;
R.R2w      = R2w;
R.Rq= Rq;
R.covP=covP;

R.ratioG   = R.g_ratio;  R.ratioM  = R.m_ratio;
R.CI95_G   = R.g_CI95;   R.CI95_M  = R.m_CI95;

% -------- optional: 95% CI for g2/g1 (unchanged) -------------------
g1 = p(2); g2 = p(3);
if abs(g1) < eps
    g21_ratio = NaN; g21_SE = NaN; g21_CI = [NaN NaN];
    g21_hw = NaN; g21_w = NaN;
else
    g21_ratio = g2 / g1;
    d21 = zeros(1,rankX); d21(2) = -g2 / (g1^2); d21(3) = 1 / g1;
    q = d21 * covP * d21.'; q = max(q, 0);
    g21_SE = sqrt(q);
    z95 = 1.95996398454005;
    g21_CI = g21_ratio + z95 * g21_SE * [-1 1];
    g21_hw = z95 * g21_SE; g21_w = 2 * g21_hw;
end

R.g21_ratio  = g21_ratio;
R.g21_SE     = g21_SE;
R.g21_CI95   = g21_CI;
R.g21_CI95HW = g21_hw;
R.g21_CI95W  = g21_w;

% -------- optional: new ratios g100 = g1/g0^2 and g210 = g2/(g1*g0) -----------
g0 = p(1); g1 = p(2); g2 = p(3);

% g100 = g1 / g0^2
if abs(g0) < eps
    g100_ratio = NaN; g100_SE = NaN; g100_CI = [NaN NaN];
    g100_hw = NaN; g100_w = NaN;
else
    g100_ratio = g1 / g0^2;
    d100 = zeros(1,rankX); 
    d100(1) = -2*g1 / (g0^3);  % derivative w.r.t g0
    d100(2) = 1 / (g0^2);      % derivative w.r.t g1
    q = d100 * covP * d100.'; q = max(q,0);
    g100_SE = sqrt(q);
    g100_hw = z95 * g100_SE; g100_w = 2 * g100_hw;
    g100_CI = g100_ratio + z95 * g100_SE * [-1 1];
end

% g210 = g2 / (g1 * g0)
if abs(g0*g1) < eps
    g210_ratio = NaN; g210_SE = NaN; g210_CI = [NaN NaN];
    g210_hw = NaN; g210_w = NaN;
else
    g210_ratio = g2 / (g1 * g0);
    d210 = zeros(1,rankX);
    d210(1) = -g2 / (g1 * g0^2);  % derivative w.r.t g0
    d210(2) = -g2 / (g0 * g1^2);  % derivative w.r.t g1
    d210(3) = 1 / (g0 * g1);      % derivative w.r.t g2
    q = d210 * covP * d210.'; q = max(q,0);
    g210_SE = sqrt(q);
    g210_hw = z95 * g210_SE; g210_w = 2 * g210_hw;
    g210_CI = g210_ratio + z95 * g210_SE * [-1 1];
end

% m210 = m2 / (m1 * g0)
if abs(g0*m1) < eps
    m210_ratio = NaN; m210_SE = NaN; m210_CI = [NaN NaN];
    m210_hw = NaN; m210_w = NaN;
else
    m210_ratio = m1 / (m0 * g0);
    d210 = zeros(1,rankX);
    d210(1) = -m1 / (m0 * g0^2);  % derivative w.r.t g0
    d210(2) = -m1 / (g0 * m0^2);  % derivative w.r.t g1
    d210(3) = 1 / (g0 * m0);      % derivative w.r.t g2
    q = d210 * covP * d210.'; q = max(q,0);
    m210_SE = sqrt(q);
    m210_hw = z95 * m210_SE; m210_w = 2 * m210_hw;
    m210_CI = m210_ratio + z95 * m210_SE * [-1 1];
end

% -------- pack into output struct --------------------------------------
R.g100_ratio    = g100_ratio;
R.g100_SE       = g100_SE;
R.g100_CI95     = g100_CI;
R.g100_CI95HW   = g100_hw;
R.g100_CI95W    = g100_w;
R.g3 = g3;

R.g210_ratio    = g210_ratio;
R.g210_SE       = g210_SE;
R.g210_CI95     = g210_CI;
R.g210_CI95HW   = g210_hw;
R.g210_CI95W    = g210_w;

R.m210_ratio    = m210_ratio;
R.m210_SE       = m210_SE;
R.m210_CI95     = m210_CI;
R.m210_CI95HW   = m210_hw;
R.m210_CI95W    = m210_w;


% -------- metadata (could be useful for debugging) -----------------
R.mu_r   = mu_r;          % weighted centering used
R.rankX  = rankX;
R.SSE    = SSE; R.SSTw = SSTw; R.s2 = s2;

end

%%

function I_filtered = filter2_ungridded( ...
    H, I, X, Y, allow_shortcut, dx_h, dy_h, rel_cutoff, WAVELENGTH)
%FILTER2_UNGRIDDED Filter detector data using a center-q Gaussian PSF.
%
% =========================================================================
% PURPOSE
% =========================================================================
%
% This function filters detector data when:
%
%   1. H is a bi-Gaussian PSF represented in q-pixel units at the
%      detector center.
%
%   2. X and Y contain the qx and qy coordinates of every detector pixel.
%
%   3. The physical PSF is fixed in scattering solid angle rather than
%      fixed in q-space.
%
%   4. The mapping from detector pixels to q-space is slightly distorted
%      and may contain a coupled, nonseparable two-dimensional aberration.
%
% The function avoids interp2. Instead, it:
%
%   - maps q coordinates to center-equivalent angular coordinates;
%   - calculates the solid angle represented by each detector pixel;
%   - approximates the nonuniform-coordinate filtering using a first-order
%     Gaussian derivative expansion;
%   - processes invalid values with numerator/denominator normalization;
%   - uses a separable Gaussian kernel for speed.
%
% =========================================================================
% INPUT INTERFACE
% =========================================================================
%
% The input interface is unchanged from the original implementation:
%
%   I_filtered = filter2_ungridded(H, I, X, Y)
%
%   I_filtered = filter2_ungridded( ...
%       H, I, X, Y, allow_shortcut)
%
%   I_filtered = filter2_ungridded( ...
%       H, I, X, Y, allow_shortcut, dx_h, dy_h)
%
%   I_filtered = filter2_ungridded( ...
%       H, I, X, Y, allow_shortcut, dx_h, dy_h, rel_cutoff)
%
% H
%   Two-dimensional bi-Gaussian PSF.
%
%   H is sampled in center-equivalent q coordinates. In other words,
%   dx_h and dy_h describe the q-pixel pitch corresponding to the angular
%   PSF at the detector center.
%
% I
%   Detector intensity or counts per equal-area detector pixel.
%
% X, Y
%   qx and qy coordinates of each detector pixel.
%
% allow_shortcut
%   If true, a regular grid with matching kernel pitch uses filter2 or
%   normalized conv2 directly.
%
% dx_h, dy_h
% dx_h and dy_h are the center-equivalent q spacing per H sample.
%
% rel_cutoff
%   Estimated relative error permitted from truncating the one-dimensional
%   Gaussian tails. The support estimate depends on gradients in I.
%
% =========================================================================
% PHYSICAL COORDINATE TRANSFORMATION
% =========================================================================
%
% The relation between q and the full scattering angle alpha = 2*theta is:
%
%       q = (4*pi/lambda)*sin(alpha/2)
%
% Define:
%
%       k = 2*pi/lambda
%
% Then:
%
%       q = 2*k*sin(alpha/2)
%
% At the detector center:
%
%       dq/dalpha = k
%
% Therefore, define center-equivalent angular coordinates:
%
%       QX = k*alpha*qx/q
%       QY = k*alpha*qy/q
%
% These coordinates:
%
%   - have the same units as q;
%   - equal q to first order near the beam center;
%   - allow H to remain expressed in center q-pixel units;
%   - represent a fixed angular PSF as a nominal shift-invariant kernel.
%
% %

% =========================================================================
% SOLID-ANGLE NORMALIZATION AND NUMERATOR/DENOMINATOR MEANING
% =========================================================================
%
% This section assumes that I contains integrated detector counts per
% equal-area physical detector pixel.
%
% -------------------------------------------------------------------------
% 1. What a detector pixel measures
% -------------------------------------------------------------------------
%
% Let:
%
%       J_omega(s) = scattering intensity density per unit solid angle
%                    at source pixel s
%
%       DeltaOmega(s) = solid angle represented by source pixel s
%
% Ignoring detector efficiency and other calibration factors, the measured
% value in detector pixel s is:
%
%       I(s) = J_omega(s) * DeltaOmega(s)
%
% Therefore, the intensity density per unit solid angle is:
%
%       J_omega(s) = I(s) / DeltaOmega(s)
%
% Since the function works with a relative solid-angle map, define:
%
%       solid_angle_ratio(s)
%           = DeltaOmega(s) / DeltaOmega_0
%
% where DeltaOmega_0 is a nominal reference solid angle, chosen here as the
% median valid detector-pixel solid angle.
%
% The intensity expressed per nominal solid angle is then:
%
%       I_omega_0(s)
%           = J_omega(s) * DeltaOmega_0
%
%           = I(s) * DeltaOmega_0 / DeltaOmega(s)
%
%           = I(s) / solid_angle_ratio(s)
%
% Thus, an explicit point-by-point solid-angle correction would be:
%
%       I_omega_0 = I ./ solid_angle_ratio
%
% The function does not form this corrected array explicitly because the
% correction cancels with the quadrature weight in the convolution
% numerator, as shown below.
%
% -------------------------------------------------------------------------
% 2. Filtering a continuous angular intensity
% -------------------------------------------------------------------------
%
% Let H(p,s) be the angular PSF weight connecting source pixel s to output
% pixel p. A normalized angular-space filter is approximated by:
%
%                            sum_s H(p,s) J_omega(s) DeltaOmega(s)
%       J_filtered(p) = ------------------------------------------------
%                                  sum_s H(p,s) DeltaOmega(s)
%
% The numerator is the angular integral of the intensity weighted by the
% PSF. The denominator is the angular integral of the PSF itself over the
% available detector samples.
%
% The denominator is necessary because:
%
%   - detector pixels may represent different solid angles;
%   - some intensity samples may be NaN or Inf;
%   - the PSF may extend beyond the available detector region;
%   - the coordinate transformation makes the effective sampling
%     nonuniform.
%
% -------------------------------------------------------------------------
% 3. Why the numerator uses the original measured I
% -------------------------------------------------------------------------
%
% Substitute the detector measurement:
%
%       I(s) = J_omega(s) * DeltaOmega(s)
%
% into the numerator:
%
%       J_omega(s) * DeltaOmega(s) = I(s)
%
% Therefore:
%
%       sum_s H(p,s) J_omega(s) DeltaOmega(s)
%
% is exactly:
%
%       sum_s H(p,s) I(s)
%
% This is why the numerator source is the original measured detector array:
%
%       numerator_source = I
%
% There is no missing solid-angle correction in the numerator. The factor
% 1/DeltaOmega required to convert counts to intensity density is canceled
% by the DeltaOmega quadrature weight used to approximate the angular
% integral.
%
% Equivalently, using relative solid-angle units:
%
%       [I(s) / solid_angle_ratio(s)] ...
%           * solid_angle_ratio(s)
%
%       = I(s)
%
% Forming the division and multiplication explicitly would produce the same
% numerator but would require two unnecessary full-array operations.
%
% -------------------------------------------------------------------------
% 4. Why the denominator uses solid_angle_ratio
% -------------------------------------------------------------------------
%
% The denominator represents the PSF-weighted solid angle contributing to
% each output pixel:
%
%       sum_s H(p,s) DeltaOmega(s)
%
% Divide numerator and denominator by the constant reference solid angle
% DeltaOmega_0:
%
%       sum_s H(p,s) I(s)
%       ---------------------------------------------
%       sum_s H(p,s) [DeltaOmega(s)/DeltaOmega_0]
%
% The denominator source is therefore:
%
%       denominator_source = solid_angle_ratio
%
% where:
%
%       solid_angle_ratio = DeltaOmega / DeltaOmega_0
%
% The final result is:
%
%                            filtered numerator_source
%       I_filtered = ------------------------------------------
%                          filtered denominator_source
%
% or, in code:
%
%       numerator = F(numerator_source)
%       denominator = F(denominator_source)
%       I_filtered = numerator ./ denominator
%
% Here F denotes the same first-order, coordinate-corrected PSF operator
% applied to both arrays.
%
% The output is intensity scaled to the nominal solid angle DeltaOmega_0:
%
%       I_filtered(p) = J_filtered(p) * DeltaOmega_0
%
% Consequently, the output remains on a scale comparable to counts per
% nominal central detector pixel. It is not an absolute intensity per
% steradian unless it is divided by the absolute DeltaOmega_0.
%
% -------------------------------------------------------------------------
% 5. Treatment of invalid intensity pixels
% -------------------------------------------------------------------------
%
% A NaN or Inf in I means that no usable intensity measurement exists at
% that detector pixel. It must not be interpreted as a measured intensity
% of zero.
%
% For an invalid intensity pixel:
%
%       numerator_source(s)   = 0
%       denominator_source(s) = 0
%
% The zero in numerator_source is only a computational placeholder. The
% matching zero in denominator_source removes that pixel's statistical and
% angular weight from the normalized result.
%
% Thus, the implemented sources are:
%
%       numerator_source = I
%       numerator_source(~valid_mask) = 0
%
%       denominator_source = solid_angle_ratio
%       denominator_source(~valid_mask) = 0
%
% Both sources are filtered with the same physical-coordinate operator:
%
%       numerator   = F(numerator_source)
%       denominator = F(denominator_source)
%
% followed by:
%
%       I_filtered = numerator ./ denominator
%
% This is normalized convolution. Missing samples do not create dark
% depressions around NaNs because their values and their weights are both
% removed.
%
% -------------------------------------------------------------------------
% 6. Constant-intensity example
% -------------------------------------------------------------------------
%
% Suppose the true angular intensity density is constant:
%
%       J_omega(s) = C
%
% The measured detector counts vary with pixel solid angle:
%
%       I(s) = C * DeltaOmega(s)
%
% Using relative solid-angle units:
%
%       I(s) = C * DeltaOmega_0 * solid_angle_ratio(s)
%
% The numerator becomes:
%
%       numerator(p)
%           = C * DeltaOmega_0 ...
%             * sum_s H(p,s) solid_angle_ratio(s)
%
% The denominator is:
%
%       denominator(p)
%           = sum_s H(p,s) solid_angle_ratio(s)
%
% Their ratio is:
%
%       I_filtered(p) = C * DeltaOmega_0
%
% Therefore, a spatially constant intensity per solid angle remains
% constant after filtering even though the uncorrected detector counts vary
% from pixel to pixel.
%
% -------------------------------------------------------------------------
% 7. If I is already corrected for solid angle
% -------------------------------------------------------------------------
%
% If the input I has already been converted to intensity per nominal solid
% angle:
%
%       I(s) = I_raw(s) / solid_angle_ratio(s)
%
% then the cancellation described above no longer applies directly.
%
% In that case, the numerator quadrature source should be:
%
%       numerator_source = I .* solid_angle_ratio
%
% while:
%
%       denominator_source = solid_angle_ratio
%
% The current implementation assumes that I contains uncorrected integrated
% detector-pixel counts. Supplying an already solid-angle-corrected I without
% changing numerator_source would apply the physical convention
% incorrectly.
%
% -------------------------------------------------------------------------
% 8. Absolute intensity per steradian
% -------------------------------------------------------------------------
%
% The function normalizes the solid-angle map by DeltaOmega_0:
%
%       solid_angle_ratio = DeltaOmega / DeltaOmega_0
%
% Consequently, I_filtered is expressed per nominal solid angle rather than
% per steradian.
%
% If the absolute DeltaOmega_0 is known and the detector counts have the
% necessary flux, exposure, efficiency, and transmission calibrations, an
% absolute per-steradian result can be formed as:
%
%       I_filtered_per_steradian = I_filtered / DeltaOmega_0
%
% Those additional experimental calibration factors are outside the scope
% of this filtering function.
%
% =========================================================================

% %
% =========================================================================
% NaN HANDLING
% =========================================================================
%
% Invalid intensity pixels are never treated as measured zeros.
%
% Internally:
%
%       numerator_source(invalid)   = 0
%       denominator_source(invalid) = 0
%
% The same coordinate-dependent derivative-expanded filter is applied to
% both sources, and the final result is:
%
%       I_filtered = filtered_numerator ./ filtered_denominator
%
% Thus, invalid pixels have zero statistical/area weight and do not lower
% nearby values as zero-valued observations would.
%
% Pixels for which the filtered denominator is too small are set to NaN.
%
% =========================================================================
% APPROXIMATION
% =========================================================================
%
% Let epsilon be the difference between the actual center-equivalent
% coordinate and its nominal regular-grid coordinate:
%
%       Q_actual = Q_nominal + epsilon
%
% For source pixel s and destination pixel p:
%
%       H(DQ + epsilon_s - epsilon_p)
%
% is approximated by:
%
%       H(DQ)
%       + (epsilon_x_s - epsilon_x_p)*dH/dQx
%       + (epsilon_y_s - epsilon_y_p)*dH/dQy
%
% This is first-order accurate in the coordinate displacement.
%
% =========================================================================

    %% ====================================================================
    % Configuration parameters
    % =====================================================================

    DEFAULT_ALLOW_SHORTCUT = true;
    DEFAULT_REL_CUTOFF     = 1e-3;

    % Grid-classification tolerances.
    GRID_REGULAR_TOL       = 1e-3;
    GRID_SCALE_TOL         = 1e-2;

    % Kernel-validation tolerances.
    KERNEL_SEPARABLE_TOL   = 1e-5;
    GAUSSIAN_FIT_TOL       = 1e-4;

    % Numerical safety thresholds.
    WEIGHT_FLOOR           = 1e-12;
    DENOMINATOR_FLOOR      = 1e-12;
    SOLID_ANGLE_FLOOR      = 1e-6;

    % Conservative multiplier used by gradient-dependent support selection.
    GRADIENT_SAFETY_FACTOR = 1.25;

    % ---------------------------------------------------------------------
    % Experimental wavelength
    % ---------------------------------------------------------------------
    %
    % WAVELENGTH must use units reciprocal to the q units:
    %
    %   q in Angstrom^-1  -> wavelength in Angstrom
    %   q in nm^-1        -> wavelength in nm
    %
    % Replace this value with the experimental wavelength.
    if nargin<9 || isempty(WAVELENGTH)
        WAVELENGTH             = single(0.1);
    else
        WAVELENGTH             = single(WAVELENGTH);
    end        

    % Enable q-to-center-equivalent-angular-coordinate conversion.
    USE_Q_TO_ANGLE         = true;

    % Calculate and use the local detector-pixel solid angle.
    APPLY_SOLID_ANGLE      = true;

    % Normalize around NaNs and Inf values.
    NORMALIZE_INVALID      = true;

    % If true, renormalize the partial PSF at detector boundaries.
    %
    % If false:
    %   - clean regular-grid shortcut matches filter2(...,'same');
    %   - the ungridded normalized path treats outside-detector samples as
    %     nominal valid zero-padding.
    NORMALIZE_EDGES        = false;

    % Warning threshold for the first-order coordinate correction.
    %
    % This is expressed as a fraction of Gaussian sigma.
    MAX_FIRST_ORDER_SHIFT  = 0.25;

    %% ====================================================================
    % Optional input defaults and validation
    % =====================================================================

    if nargin < 5 || isempty(allow_shortcut)
        allow_shortcut = DEFAULT_ALLOW_SHORTCUT;
    end

    if nargin < 8 || isempty(rel_cutoff)
        rel_cutoff = DEFAULT_REL_CUTOFF;
    end

    if ~isscalar(rel_cutoff) || ...
       ~isfinite(rel_cutoff) || ...
       rel_cutoff < 0

        error('rel_cutoff must be a finite nonnegative scalar.');
    end

    if ~isscalar(WAVELENGTH) || ...
       ~isfinite(WAVELENGTH) || ...
       WAVELENGTH <= 0

        error('WAVELENGTH must be a finite positive scalar.');
    end

    %% ====================================================================
    % Convert large arrays to single precision
    % =====================================================================

    H = single(H);
    I = single(I);
    X = single(X);
    Y = single(Y);

    [rows, cols] = size(I);

    if ~isequal(size(X), size(I)) || ...
       ~isequal(size(Y), size(I))

        error('I, X, and Y must have identical sizes.');
    end

    if isempty(H)
        error('H must be nonempty.');
    end

    %% ====================================================================
    % Normalize the PSF
    % =====================================================================

    H_sum = sum(H(:), 'native');

    if ~isfinite(H_sum) || ...
       abs(H_sum) <= single(WEIGHT_FLOOR)

        error('H must have a finite nonzero sum.');
    end

    H = H ./ H_sum;

    %% ====================================================================
    % Convert q coordinates to center-equivalent angular coordinates
    % =====================================================================

    coordinate_valid = isfinite(X) & isfinite(Y);

    if USE_Q_TO_ANGLE
        q_magnitude = hypot(X, Y);

        asin_argument = ...
            WAVELENGTH .* q_magnitude ./ single(4*pi);

        physical_q = coordinate_valid & ...
                     asin_argument >= 0 & ...
                     asin_argument < 1;

        if any(coordinate_valid(:)) && ...
           all(physical_q(coordinate_valid))

            angular_mode = true;

            % Full scattering angle alpha = 2*theta.
            alpha = 2 .* asin(asin_argument);

            % k = 2*pi/lambda = dq/dalpha at the beam center.
            center_q_per_radian = ...
                single(2*pi) ./ WAVELENGTH;

            direction_x = zeros(rows, cols, 'single');
            direction_y = zeros(rows, cols, 'single');

            nonzero_q = ...
                coordinate_valid & ...
                q_magnitude > eps('single');

            direction_x(nonzero_q) = ...
                X(nonzero_q) ./ q_magnitude(nonzero_q);

            direction_y(nonzero_q) = ...
                Y(nonzero_q) ./ q_magnitude(nonzero_q);

            % Center-equivalent angular coordinates in q units.
            QX = center_q_per_radian .* alpha .* direction_x;
            QY = center_q_per_radian .* alpha .* direction_y;

            % The beam center maps exactly to zero.
            center_pixels = coordinate_valid & ~nonzero_q;

            QX(center_pixels) = 0;
            QY(center_pixels) = 0;

            QX(~coordinate_valid) = NaN;
            QY(~coordinate_valid) = NaN;
        else
            % The supplied benchmark uses synthetic coordinates that do not
            % necessarily satisfy the physical q relation. Fall back to
            % treating X and Y directly as filtering coordinates.
            angular_mode = false;

            QX = X;
            QY = Y;

            alpha = zeros(rows, cols, 'single');

            warning('filter2_ungridded:NonphysicalQ', ...
                ['Some q values do not satisfy lambda*q/(4*pi) < 1. ', ...
                 'For this call, X and Y are treated directly as ', ...
                 'filter coordinates and solid-angle conversion is ', ...
                 'disabled.']);
        end
    else
        angular_mode = false;

        QX = X;
        QY = Y;

        alpha = zeros(rows, cols, 'single');
    end

%% ====================================================================
% Estimate center-reference detector spacing
% =====================================================================
%
% H is sampled in center-equivalent q units at the detector center.
% Therefore, dqx_grid and dqy_grid should represent the detector spacing
% near q = 0, rather than the mean or maximum spacing over the complete
% aberrated detector.
%
% QX and QY are the center-equivalent angular coordinates expressed in
% q units. Near q = 0:
%
%       QX approximately equals X
%       QY approximately equals Y
%
% The following procedure:
%
%   1. Locates the detector pixel nearest q = 0 using X and Y.
%   2. Extracts a small region around that pixel.
%   3. Uses the median local spacing in QX and QY.
%   4. Falls back to the complete detector if the central region does not
%      contain enough valid coordinate differences.
%   5. Uses the complete spacing matrices to determine whether the entire
%      detector is sufficiently regular for the filter2 shortcut.

% Number of detector pixels included on each side of the pixel nearest
% q = 0. A value of 2 gives a region of at most 5-by-5 pixels.
CENTER_HALF_WIDTH = 2;

% ---------------------------------------------------------------------
% Locate the detector pixel nearest q = 0
% ---------------------------------------------------------------------

q_squared = X.^2 + Y.^2;

% Invalid coordinate pairs must not be selected as the detector center.
q_squared(~isfinite(X) | ~isfinite(Y)) = Inf;

[minimum_q_squared, center_linear_index] = min(q_squared(:));

if isempty(minimum_q_squared) || ~isfinite(minimum_q_squared)
    error('No finite X,Y coordinate pair is available.');
end

[center_row, center_col] = ...
    ind2sub([rows, cols], center_linear_index);

% ---------------------------------------------------------------------
% Define a small region around q = 0
% ---------------------------------------------------------------------

center_row_first = ...
    max(1, center_row - CENTER_HALF_WIDTH);

center_row_last = ...
    min(rows, center_row + CENTER_HALF_WIDTH);

center_col_first = ...
    max(1, center_col - CENTER_HALF_WIDTH);

center_col_last = ...
    min(cols, center_col + CENTER_HALF_WIDTH);

% ---------------------------------------------------------------------
% Calculate complete-detector coordinate increments
% ---------------------------------------------------------------------
%
% dqx_matrix contains changes in QX between consecutive columns.
% dqy_matrix contains changes in QY between consecutive rows.
%
% These complete matrices are used later to test whether the detector is
% sufficiently regular for the direct filter2 shortcut.

dqx_matrix = diff(QX, 1, 2);
dqy_matrix = diff(QY, 1, 1);

% Cross-coordinate increments are also calculated.
%
% A grid can have nearly constant dQX/dcolumn and dQY/drow while still
% containing shear or rotation through:
%
%       dQY/dcolumn
%       dQX/drow
%
% These terms should be included when deciding whether the ordinary
% filter2 shortcut is physically valid.

dqy_dcol_matrix = diff(QY, 1, 2);
dqx_drow_matrix = diff(QX, 1, 1);

% ---------------------------------------------------------------------
% Calculate coordinate increments near q = 0
% ---------------------------------------------------------------------

QX_center_region = ...
    QX(center_row_first:center_row_last, ...
       center_col_first:center_col_last);

QY_center_region = ...
    QY(center_row_first:center_row_last, ...
       center_col_first:center_col_last);

dqx_center = diff(QX_center_region, 1, 2);
dqy_center = diff(QY_center_region, 1, 1);

% Also estimate central cross-coordinate increments. These are used only
% for the regular-grid classification.
dqy_dcol_center = diff(QY_center_region, 1, 2);
dqx_drow_center = diff(QX_center_region, 1, 1);

% ---------------------------------------------------------------------
% Remove invalid and zero coordinate increments
% ---------------------------------------------------------------------
%
% Zero increments are not valid estimates of detector pitch. They may
% occur if the coordinate map contains repeated values or if the selected
% central region is too small in one dimension.

valid_dqx_center = ...
    dqx_center(isfinite(dqx_center) & dqx_center ~= 0);

valid_dqy_center = ...
    dqy_center(isfinite(dqy_center) & dqy_center ~= 0);

valid_dqy_dcol_center = ...
    dqy_dcol_center(isfinite(dqy_dcol_center));

valid_dqx_drow_center = ...
    dqx_drow_center(isfinite(dqx_drow_center));

% ---------------------------------------------------------------------
% Estimate signed center-reference detector pitches
% ---------------------------------------------------------------------
%
% The median is used instead of the maximum because it is resistant to a
% small number of noisy or geometrically abnormal coordinate differences.
%
% The signs are retained because they describe the orientation of each
% detector coordinate axis.

if ~isempty(valid_dqx_center)
    dqx_grid = median(valid_dqx_center);
else
    % Fallback to the complete detector if the central region contains no
    % usable horizontal coordinate differences.
    valid_dqx_all = ...
        dqx_matrix(isfinite(dqx_matrix) & dqx_matrix ~= 0);

    if isempty(valid_dqx_all)
        error(['Could not determine a finite nonzero horizontal ', ...
               'center-reference coordinate spacing.']);
    end

    dqx_grid = median(valid_dqx_all);

    warning('filter2_ungridded:CentralDqxUnavailable', ...
        ['No valid horizontal spacing was found in the central region. ', ...
         'The median spacing over the complete detector is being used.']);
end

if ~isempty(valid_dqy_center)
    dqy_grid = median(valid_dqy_center);
else
    % Fallback to the complete detector if the central region contains no
    % usable vertical coordinate differences.
    valid_dqy_all = ...
        dqy_matrix(isfinite(dqy_matrix) & dqy_matrix ~= 0);

    if isempty(valid_dqy_all)
        error(['Could not determine a finite nonzero vertical ', ...
               'center-reference coordinate spacing.']);
    end

    dqy_grid = median(valid_dqy_all);

    warning('filter2_ungridded:CentralDqyUnavailable', ...
        ['No valid vertical spacing was found in the central region. ', ...
         'The median spacing over the complete detector is being used.']);
end

% Convert the scalar pitches back to single precision. Depending on the
% MATLAB release, median may otherwise preserve or promote the input type.
dqx_grid = single(dqx_grid);
dqy_grid = single(dqy_grid);

if ~isfinite(dqx_grid) || dqx_grid == 0
    error(['Could not determine a finite nonzero horizontal ', ...
           'center-reference coordinate spacing.']);
end

if ~isfinite(dqy_grid) || dqy_grid == 0
    error(['Could not determine a finite nonzero vertical ', ...
           'center-reference coordinate spacing.']);
end

% ---------------------------------------------------------------------
% Estimate central cross-coordinate increments
% ---------------------------------------------------------------------
%
% For an ideal Cartesian detector grid:
%
%       dQY/dcolumn = 0
%       dQX/drow    = 0
%
% Small nonzero values can result from numerical roundoff or slight
% detector rotation. Large values indicate shear, rotation, or coupled
% two-dimensional distortion, for which the direct filter2 shortcut should
% not be used.

if isempty(valid_dqy_dcol_center)
    dqy_dcol_grid = single(0);
else
    dqy_dcol_grid = ...
        single(median(valid_dqy_dcol_center));
end

if isempty(valid_dqx_drow_center)
    dqx_drow_grid = single(0);
else
    dqx_drow_grid = ...
        single(median(valid_dqx_drow_center));
end

% ---------------------------------------------------------------------
% Determine the sampling pitch of H
% ---------------------------------------------------------------------
%
% dx_h_used and dy_h_used describe the physical center-equivalent q
% spacing between adjacent samples of H.
%
% If dx_h or dy_h is omitted, the function assumes that one PSF sample
% corresponds to one detector pixel at q = 0.
%
% Keeping the inputs is important when H is oversampled or undersampled
% relative to the detector.

if nargin < 6 || isempty(dx_h)
    dx_h_used = dqx_grid;
else
    dx_h_used = single(dx_h);
end

if nargin < 7 || isempty(dy_h)
    dy_h_used = dqy_grid;
else
    dy_h_used = single(dy_h);
end

if ~isscalar(dx_h_used) || ...
   ~isfinite(dx_h_used) || ...
   dx_h_used == 0

    error('dx_h must be a finite nonzero scalar.');
end

if ~isscalar(dy_h_used) || ...
   ~isfinite(dy_h_used) || ...
   dy_h_used == 0

    error('dy_h must be a finite nonzero scalar.');
end

% ---------------------------------------------------------------------
% Determine whether the complete coordinate grid is regular
% ---------------------------------------------------------------------
%
% The nominal direct increments are:
%
%       dQX/dcolumn = dqx_grid
%       dQY/drow    = dqy_grid
%
% The nominal cross increments are the central values:
%
%       dQY/dcolumn = dqy_dcol_grid
%       dQX/drow    = dqx_drow_grid
%
% All four coordinate derivatives are tested. This prevents a sheared or
% rotated two-dimensional grid from being incorrectly classified as a
% standard Cartesian grid.

horizontal_grid_error = ...
    max(abs(dqx_matrix(:) - dqx_grid), [], 'omitnan') ...
    / max(abs(dqx_grid), eps('single'));

vertical_grid_error = ...
    max(abs(dqy_matrix(:) - dqy_grid), [], 'omitnan') ...
    / max(abs(dqy_grid), eps('single'));

horizontal_cross_error = ...
    max(abs(dqy_dcol_matrix(:) - dqy_dcol_grid), ...
        [], 'omitnan') ...
    / max(abs(dqy_grid), eps('single'));

vertical_cross_error = ...
    max(abs(dqx_drow_matrix(:) - dqx_drow_grid), ...
        [], 'omitnan') ...
    / max(abs(dqx_grid), eps('single'));

% The ordinary filter2 shortcut additionally requires negligible central
% cross-coupling. A globally affine but rotated or sheared grid is regular
% mathematically, but H would need to be transformed before ordinary
% row-column filter2 could represent the physical PSF exactly.

central_cross_error = max( ...
    abs(dqy_dcol_grid) / max(abs(dqy_grid), eps('single')), ...
    abs(dqx_drow_grid) / max(abs(dqx_grid), eps('single')));

is_regular = ...
    horizontal_grid_error <= GRID_REGULAR_TOL && ...
    vertical_grid_error <= GRID_REGULAR_TOL && ...
    horizontal_cross_error <= GRID_REGULAR_TOL && ...
    vertical_cross_error <= GRID_REGULAR_TOL && ...
    central_cross_error <= GRID_REGULAR_TOL;

% ---------------------------------------------------------------------
% Determine whether H has the same sampling pitch as the central detector
% ---------------------------------------------------------------------
%
% Pitch magnitudes are compared because a reversed coordinate axis does not
% alter a symmetric centered Gaussian PSF.
%
% If H may later be asymmetric, axis signs should also be handled by
% reversing the appropriate kernel dimension.

horizontal_scale_error = ...
    abs(abs(dqx_grid) - abs(dx_h_used)) ...
    / max(abs(dx_h_used), eps('single'));

vertical_scale_error = ...
    abs(abs(dqy_grid) - abs(dy_h_used)) ...
    / max(abs(dy_h_used), eps('single'));

scale_match = ...
    horizontal_scale_error <= GRID_SCALE_TOL && ...
    vertical_scale_error <= GRID_SCALE_TOL;

%
%

    scale_match = ...
        abs(dqx_grid - dx_h_used) ...
            / max(abs(dx_h_used), eps('single')) ...
            <= GRID_SCALE_TOL && ...
        abs(dqy_grid - dy_h_used) ...
            / max(abs(dy_h_used), eps('single')) ...
            <= GRID_SCALE_TOL;

    %% ====================================================================
    % Calculate relative solid angle represented by each detector pixel
    % =====================================================================

    if APPLY_SOLID_ANGLE && angular_mode

        % Derivatives with respect to detector column and row indices.
        [dQX_dc, dQX_dr] = index_derivatives(QX);
        [dQY_dc, dQY_dr] = index_derivatives(QY);

        % Area in the center-equivalent angular coordinate plane.
        coordinate_area = abs( ...
            dQX_dc .* dQY_dr - ...
            dQX_dr .* dQY_dc);

        % Spherical area factor:
        %
        %   dOmega proportional to sin(alpha)/alpha * dQX*dQY
        %
        % The omitted proportionality factor 1/k^2 is spatially constant
        % and cancels after normalization.
        sphere_factor = ones(rows, cols, 'single');

        nonzero_alpha = ...
            isfinite(alpha) & ...
            abs(alpha) > sqrt(eps('single'));

        sphere_factor(nonzero_alpha) = ...
            sin(alpha(nonzero_alpha)) ...
            ./ alpha(nonzero_alpha);

        solid_angle = sphere_factor .* coordinate_area;
        solid_angle(~coordinate_valid) = NaN;

        solid_angle_values = solid_angle( ...
            coordinate_valid & ...
            isfinite(solid_angle) & ...
            solid_angle > 0);

        if isempty(solid_angle_values)
            error('Could not calculate a valid solid-angle map.');
        end

        % Median is more robust than the mean if a few coordinate pixels
        % are invalid or locally poorly conditioned.
        nominal_solid_angle = ...
            median(solid_angle_values, 'omitnan');

        if ~isfinite(nominal_solid_angle) || ...
           nominal_solid_angle <= 0

            error('Could not determine a nominal solid angle.');
        end

        solid_angle_ratio = ...
            solid_angle ./ nominal_solid_angle;

        solid_angle_ratio( ...
            ~isfinite(solid_angle_ratio) | ...
            solid_angle_ratio < SOLID_ANGLE_FLOOR) = NaN;
    else
        solid_angle_ratio = ...
            ones(rows, cols, 'single');
    end

    %% ====================================================================
    % Construct the complete validity mask
    % =====================================================================

    % Intensity validity is intentionally separate from coordinate validity.
    %
    % A coordinate can be geometrically valid while its intensity is NaN.
    % In that situation its denominator weight must be set to zero, because
    % it represents a missing observation, not a zero-valued observation.
    intensity_valid = isfinite(I);

    geometry_valid = ...
        isfinite(QX) & ...
        isfinite(QY) & ...
        isfinite(solid_angle_ratio) & ...
        solid_angle_ratio > 0;

    valid_mask = intensity_valid & geometry_valid;

    has_invalid = ~all(valid_mask(:));

    %% ====================================================================
    % Regular-grid shortcut
    % =====================================================================

    if allow_shortcut && is_regular && scale_match

        solid_angle_values = ...
            solid_angle_ratio(valid_mask);

        if isempty(solid_angle_values)
            I_filtered = nan(rows, cols, 'single');
            return;
        end

        solid_angle_is_uniform = ...
            max(abs(solid_angle_values - 1), ...
                [], 'all', 'omitnan') ...
            <= GRID_REGULAR_TOL;

        % Exact filter2 shortcut is permitted only when:
        %
        %   - no intensity or geometry values are invalid;
        %   - no varying solid-angle correction is required;
        %   - filter2-like boundary attenuation is requested.
        if ~has_invalid && ...
           solid_angle_is_uniform && ...
           ~NORMALIZE_EDGES

            I_filtered = filter2(H, I, 'same');
            return;
        end

        % Internal zero replacement is safe because invalid pixels receive
        % exactly zero denominator weight.
        numerator_source = I;
        numerator_source(~valid_mask) = 0;

        if APPLY_SOLID_ANGLE
            denominator_source = solid_angle_ratio;
        else
            denominator_source = ones(rows, cols, 'single');
        end

        % This is the critical NaN correction. An invalid intensity pixel
        % gets zero denominator weight and therefore cannot bias nearby
        % values as a measured zero.
        denominator_source(~valid_mask) = 0;

        H_convolution = rot90(H, 2);

        numerator = conv2( ...
            numerator_source, H_convolution, 'same');

        denominator = conv2( ...
            denominator_source, H_convolution, 'same');

        if ~NORMALIZE_EDGES
            % Treat outside-detector samples as nominal valid zero padding.
            %
            % Internal NaNs are still ignored because no correction is
            % added at their locations.
            inside_kernel_weight = conv2( ...
                ones(rows, cols, 'single'), ...
                H_convolution, 'same');

            denominator = ...
                denominator + 1 - inside_kernel_weight;
        end

        I_filtered = numerator ./ denominator;

        invalid_output = ...
            ~isfinite(denominator) | ...
            denominator <= DENOMINATOR_FLOOR;

        I_filtered(invalid_output) = NaN;
        return;
    end

    %% ====================================================================
    % Extract separable rank-1 factors from H
    % =====================================================================

    [kernel_rows, kernel_cols] = size(H);

    row_center = ceil(kernel_rows / 2);
    col_center = ceil(kernel_cols / 2);

    % The kernel is small relative to the detector image. Using double
    % precision for the SVD improves validation and factor accuracy without
    % substantially increasing memory use.
    [U, S, V] = svd(double(H), 'econ');

    singular_values = diag(S);
    total_singular_energy = norm(singular_values);

    if numel(singular_values) > 1
        residual_singular_energy = ...
            norm(singular_values(2:end));
    else
        residual_singular_energy = 0;
    end

    separability_error = ...
        residual_singular_energy ...
        / max(total_singular_energy, eps);

    if separability_error > KERNEL_SEPARABLE_TOL
        error(['H is not sufficiently rank-1 separable. ', ...
               'Relative rank-1 residual: %.3e.'], ...
               separability_error);
    end

    square_root_singular_value = sqrt(S(1,1));

    gy = single( ...
        U(:,1) .* square_root_singular_value);

    gx = single( ...
        V(:,1) .* square_root_singular_value);

    % The signs of SVD vectors are arbitrary. Choose positive-sum factors.
    if sum(gy, 'native') < 0
        gy = -gy;
        gx = -gx;
    end

    gy_sum = sum(gy, 'native');
    gx_sum = sum(gx, 'native');

    if abs(gy_sum) <= WEIGHT_FLOOR || ...
       abs(gx_sum) <= WEIGHT_FLOOR

        error('Separable PSF factors have a near-zero sum.');
    end

    gy = gy ./ gy_sum;
    gx = gx ./ gx_sum;

    %% ====================================================================
    % Center-reference q coordinates of the PSF samples
    % =====================================================================

    qy_kernel = single( ...
        ((1:kernel_rows) - row_center).' ...
        .* double(dy_h_used));

    qx_kernel = single( ...
        ((1:kernel_cols) - col_center).' ...
        .* double(dx_h_used));

    %% ====================================================================
    % Select I-dependent Gaussian support
    % =====================================================================

    if rel_cutoff > 0
        % Split the approximate relative error budget between the two
        % separable passes.
        pass_cutoff = rel_cutoff / 2;

        [row_first, row_last] = gradient_support_1d( ...
            gy, qy_kernel, I, valid_mask, 1, ...
            pass_cutoff, GRADIENT_SAFETY_FACTOR);

        [col_first, col_last] = gradient_support_1d( ...
            gx, qx_kernel, I, valid_mask, 2, ...
            pass_cutoff, GRADIENT_SAFETY_FACTOR);
    else
        % A zero cutoff disables support truncation.
        row_first = 1;
        row_last = kernel_rows;

        col_first = 1;
        col_last = kernel_cols;
    end

    gy = gy(row_first:row_last);
    gx = gx(col_first:col_last);

    qy_kernel = qy_kernel(row_first:row_last);
    qx_kernel = qx_kernel(col_first:col_last);

    % Preserve the response to a constant field after truncation.
    gy = gy ./ sum(gy, 'native');
    gx = gx ./ sum(gx, 'native');

%% ====================================================================
% Construct derivative filters for the retained Gaussian support
% =====================================================================
%
% A retained support containing only one coefficient is a delta kernel.
% No Gaussian fit is possible or necessary in that dimension:
%
%   g = 1
%   dg/dq = 0
%
% Consequently, coordinate-displacement corrections in that dimension
% vanish. This is also the correct behavior because a one-sample kernel
% has no neighboring support over which its weight can shift.

if numel(gx) == 1
    % Horizontal delta kernel.
    px = single([0, 0, 0]);
    accepted_x = true;

    gx = single(1);
    dgx = single(0);

    sigma_x = single(Inf);
else
    [px, accepted_x] = fit_log_gaussian( ...
        qx_kernel, gx, ...
        WEIGHT_FLOOR, GAUSSIAN_FIT_TOL);

    if accepted_x
        % If:
        %
        %   log(g(q)) = p1*q^2 + p2*q + p3
        %
        % then:
        %
        %   dg/dq = (2*p1*q + p2)*g
        dgx = ...
            (2 .* px(1) .* qx_kernel + px(2)) .* gx;

        % A derivative filter must have zero DC response. Remove any
        % residual caused by finite support or floating-point roundoff.
        dgx = dgx - sum(dgx, 'native') .* gx;

        sigma_x = gaussian_sigma_from_fit(px);
    else
        sigma_x = single(Inf);
    end
end

if numel(gy) == 1
    % Vertical delta kernel.
    py = single([0, 0, 0]);
    accepted_y = true;

    gy = single(1);
    dgy = single(0);

    sigma_y = single(Inf);
else
    [py, accepted_y] = fit_log_gaussian( ...
        qy_kernel, gy, ...
        WEIGHT_FLOOR, GAUSSIAN_FIT_TOL);

    if accepted_y
        dgy = ...
            (2 .* py(1) .* qy_kernel + py(2)) .* gy;

        dgy = dgy - sum(dgy, 'native') .* gy;

        sigma_y = gaussian_sigma_from_fit(py);
    else
        sigma_y = single(Inf);
    end
end

if ~accepted_x || ~accepted_y
    error(['The retained PSF support is not sufficiently Gaussian. ', ...
           'Horizontal support: %d samples. ', ...
           'Vertical support: %d samples.'], ...
           numel(gx), numel(gy));
end
    %% ====================================================================
    % Construct nominal regular center-equivalent coordinate grid
    % =====================================================================

    column_index = single(0:cols-1);
    row_index = single((0:rows-1).');

    qx_origin_samples = ...
        QX - repmat(column_index .* dqx_grid, rows, 1);

    qy_origin_samples = ...
        QY - repmat(row_index .* dqy_grid, 1, cols);

    qx_origin = median( ...
        qx_origin_samples(:), 'omitnan');

    qy_origin = median( ...
        qy_origin_samples(:), 'omitnan');

    QX_nominal = ...
        qx_origin + ...
        repmat(column_index .* dqx_grid, rows, 1);

    QY_nominal = ...
        qy_origin + ...
        repmat(row_index .* dqy_grid, 1, cols);

    % Full coupled two-dimensional coordinate displacement.
    epsilon_x = QX - QX_nominal;
    epsilon_y = QY - QY_nominal;

    epsilon_x(~geometry_valid) = 0;
    epsilon_y(~geometry_valid) = 0;

    %% ====================================================================
    % Warn if the first-order approximation may be insufficient
    % =====================================================================

% sigma_x and sigma_y were determined while constructing the derivative
% filters. A one-sample delta kernel has sigma = Inf here, so it produces
% zero normalized displacement and does not trigger a false warning.

max_shift_x = ...
    max(abs(epsilon_x(:)), [], 'omitnan') ...
    / max(sigma_x, eps('single'));

max_shift_y = ...
    max(abs(epsilon_y(:)), [], 'omitnan') ...
    / max(sigma_y, eps('single'));

    if max(max_shift_x, max_shift_y) > MAX_FIRST_ORDER_SHIFT
        warning('filter2_ungridded:FirstOrderShift', ...
            ['Maximum coordinate displacement exceeds %.2f of the ', ...
             'Gaussian sigma. First-order accuracy may be insufficient. ', ...
             'Normalized shifts: X = %.3f, Y = %.3f.'], ...
             MAX_FIRST_ORDER_SHIFT, ...
             max_shift_x, max_shift_y);
    end

    %% ====================================================================
    % Build NaN-safe numerator and denominator sources
    % =====================================================================

    % NUMERATOR:
    %
    % If I contains counts per equal detector pixel, the conversion from
    % counts to intensity density per solid angle is proportional to:
    %
    %   I_omega = I / solid_angle_ratio
    %
    % Physical quadrature then multiplies I_omega by solid_angle_ratio.
    % Those factors cancel, so the numerator source is the original I.
    numerator_source = I;
    numerator_source(~valid_mask) = 0;

    % DENOMINATOR:
    %
    % Each valid observation receives its represented solid-angle weight.
    %
    % Critically, an intensity NaN receives zero denominator weight. It is
    % therefore omitted rather than interpreted as zero intensity.
    if APPLY_SOLID_ANGLE
        denominator_source = solid_angle_ratio;
    else
        denominator_source = ones(rows, cols, 'single');
    end

    denominator_source(~valid_mask) = 0;

    %% ====================================================================
    % First-order derivative-expanded numerator
    % =====================================================================

    % Zeroth-order nominal filtering.
    numerator_0 = separable_filter2( ...
        gy, gx, numerator_source);

    % Source-position x correction:
    %
    %   dH/dx * (I * epsilon_x)
    numerator_x_source = separable_filter2( ...
        gy, dgx, numerator_source .* epsilon_x);

    % Destination-position x correction:
    %
    %   epsilon_x * [dH/dx * I]
    numerator_x_destination = ...
        epsilon_x .* separable_filter2( ...
            gy, dgx, numerator_source);

    % Source-position y correction.
    numerator_y_source = separable_filter2( ...
        dgy, gx, numerator_source .* epsilon_y);

    % Destination-position y correction.
    numerator_y_destination = ...
        epsilon_y .* separable_filter2( ...
            dgy, gx, numerator_source);

    numerator = ...
        numerator_0 ...
        + numerator_x_source ...
        - numerator_x_destination ...
        + numerator_y_source ...
        - numerator_y_destination;

    %% ====================================================================
    % First-order derivative-expanded denominator
    % =====================================================================

    % The denominator must use exactly the same geometric operator as the
    % numerator. Otherwise, NaNs would cause position-dependent bias.
    denominator_0 = separable_filter2( ...
        gy, gx, denominator_source);

    denominator_x_source = separable_filter2( ...
        gy, dgx, denominator_source .* epsilon_x);

    denominator_x_destination = ...
        epsilon_x .* separable_filter2( ...
            gy, dgx, denominator_source);

    denominator_y_source = separable_filter2( ...
        dgy, gx, denominator_source .* epsilon_y);

    denominator_y_destination = ...
        epsilon_y .* separable_filter2( ...
            dgy, gx, denominator_source);

    denominator = ...
        denominator_0 ...
        + denominator_x_source ...
        - denominator_x_destination ...
        + denominator_y_source ...
        - denominator_y_destination;

    %% ====================================================================
    % Boundary convention
    % =====================================================================

    if ~NORMALIZE_EDGES
        % Restore the missing nominal kernel weight outside the detector.
        %
        % This approximates filter2(...,'same') zero-padding. The correction
        % is applied only outside the detector, not at internal NaNs.
        %
        % Internal NaNs remain omitted because their denominator source is
        % zero and no missing-weight correction is added at their locations.
        inside_kernel_weight = separable_filter2( ...
            gy, gx, ones(rows, cols, 'single'));

        denominator = ...
            denominator + 1 - inside_kernel_weight;
    end

    %% ====================================================================
    % Final normalized result
    % =====================================================================

    I_filtered = numerator ./ denominator;

    invalid_output = ...
        ~isfinite(denominator) | ...
        denominator <= DENOMINATOR_FLOOR;

    I_filtered(invalid_output) = NaN;
end


function output = separable_filter2(gy, gx, input)
%SEPARABLE_FILTER2 Apply a rank-1 kernel with filter2 orientation.
%
% This function is equivalent to:
%
%   filter2(gy*gx.', input, 'same')
%
% MATLAB filter2 performs two-dimensional correlation, while conv2 performs
% convolution. Consequently, the one-dimensional factors are reversed
% before conv2 is called.

    temporary = conv2( ...
        input, flipud(gy), 'same');

    output = conv2( ...
        temporary, fliplr(gx.'), 'same');
end


function [dA_dc, dA_dr] = index_derivatives(A)
%INDEX_DERIVATIVES Derivatives with respect to detector indices.
%
% dA_dc is the derivative with respect to detector column.
% dA_dr is the derivative with respect to detector row.
%
% Central differences are used in the interior. One-sided differences are
% used at the first and last rows or columns.
%
% NaNs propagate through differences involving invalid coordinates. Those
% locations are subsequently removed from the geometry-valid mask.

    [rows, cols] = size(A);

    dA_dc = zeros(rows, cols, 'single');
    dA_dr = zeros(rows, cols, 'single');

    if cols > 1
        dA_dc(:,1) = ...
            A(:,2) - A(:,1);

        dA_dc(:,end) = ...
            A(:,end) - A(:,end-1);
    end

    if cols > 2
        dA_dc(:,2:end-1) = ...
            0.5 .* (A(:,3:end) - A(:,1:end-2));
    end

    if rows > 1
        dA_dr(1,:) = ...
            A(2,:) - A(1,:);

        dA_dr(end,:) = ...
            A(end,:) - A(end-1,:);
    end

    if rows > 2
        dA_dr(2:end-1,:) = ...
            0.5 .* (A(3:end,:) - A(1:end-2,:));
    end
end


function [p, accepted] = fit_log_gaussian( ...
    coordinate, profile, weight_floor, fit_tolerance)
%FIT_LOG_GAUSSIAN Fit a one-dimensional Gaussian in log space.
%
% A Gaussian profile satisfies:
%
%   profile(q) = A*exp(a*q^2 + b*q)
%
% Therefore:
%
%   log(profile(q)) = a*q^2 + b*q + log(A)
%
% The function fits:
%
%   log(profile) = p(1)*q^2 + p(2)*q + p(3)
%
% The fitted profile is normalized and compared with the supplied profile.
% The fit is accepted only if the maximum relative profile error is below
% fit_tolerance.
%
% Double precision is used only for this small polynomial fit.

%
% At least three retained samples are required to determine the three
% coefficients of a quadratic polynomial.
%
% A one-sample profile must be handled by the calling function as a delta
% kernel. A two-sample profile also cannot uniquely determine a quadratic
% Gaussian model.

    coordinate_double = double(coordinate(:));
    profile_double = double(profile(:));

    % A quadratic fit requires at least three distinct samples.
    if numel(profile_double) < 3 || ...
       numel(unique(coordinate_double(isfinite(coordinate_double)))) < 3

        p = single([0, 0, 0]);
        accepted = false;
        return;
    end

    profile_scale = max(abs(profile_double));

    valid = ...
        isfinite(coordinate_double) & ...
        isfinite(profile_double) & ...
        profile_double > 0 & ...
        profile_double >= ...
            weight_floor .* max(profile_scale, eps);

    if nnz(valid) < 3
        p = single([0, 0, 0]);
        accepted = false;
        return;
    end

    p_double = polyfit( ...
        coordinate_double(valid), ...
        log(profile_double(valid)), 2);

    fitted_log_profile = ...
        (p_double(1) .* coordinate_double + p_double(2)) ...
        .* coordinate_double + p_double(3);

    fitted_profile = exp(fitted_log_profile);

    fitted_sum = sum(fitted_profile);
    reference_sum = sum(profile_double);

    if ~isfinite(fitted_sum) || fitted_sum <= 0 || ...
       ~isfinite(reference_sum) || reference_sum == 0

        p = single([0, 0, 0]);
        accepted = false;
        return;
    end

    fitted_profile = fitted_profile ./ fitted_sum;
    reference_profile = profile_double ./ reference_sum;

    relative_fit_error = ...
        max(abs(fitted_profile - reference_profile)) ...
        / max(max(abs(reference_profile)), eps);

    accepted = ...
        isfinite(relative_fit_error) && ...
        relative_fit_error <= fit_tolerance;

    p = single(p_double);
end
function sigma = gaussian_sigma_from_fit(p)
%GAUSSIAN_SIGMA_FROM_FIT Recover Gaussian sigma from a log-quadratic fit.
%
% For:
%
%   g(q) = exp(-q^2/(2*sigma^2))
%
% the quadratic log coefficient is:
%
%   p(1) = -1/(2*sigma^2)

    if p(1) < 0
        sigma = sqrt( ...
            -1 ./ (2 .* p(1)));
    else
        sigma = single(Inf);
    end
end


function [first_index, last_index] = gradient_support_1d( ...
    weights, offsets, I, valid_mask, dimension, ...
    rel_cutoff, safety_factor)
%GRADIENT_SUPPORT_1D Select an image-dependent centered kernel support.
%
% The old amplitude-only rule:
%
%   abs(weight) >= rel_cutoff*max(abs(weight))
%
% does not account for the image being filtered.
%
% This function instead estimates the effect of each omitted coefficient:
%
%   estimated contribution =
%       abs(weight) * estimated image change at that offset
%
% The change is bounded by:
%
%   gradient_bound * abs(offset)
%
% and capped at:
%
%   2*signal_scale
%
% The support is expanded from the kernel center until the cumulative
% estimated contribution of all omitted coefficients is below rel_cutoff.
%
% NaNs are excluded from both the signal-scale and gradient estimates.

    weights = single(weights(:));
    offsets = single(offsets(:));

    coefficient_count = numel(weights);
    center_index = ceil(coefficient_count / 2);

    first_index = 1;
    last_index = coefficient_count;

    valid_values = abs(I(valid_mask));

    if isempty(valid_values)
        return;
    end

    signal_scale = ...
        max(valid_values, [], 'all', 'omitnan');

    if ~isfinite(signal_scale)
        return;
    end

    if signal_scale <= eps('single')
        % A zero-valued valid image has no varying contribution.
        first_index = center_index;
        last_index = center_index;
        return;
    end

    coordinate_step = ...
        positive_median_step(offsets);

    if dimension == 1
        intensity_difference = diff(I, 1, 1);

        valid_pairs = ...
            valid_mask(1:end-1,:) & ...
            valid_mask(2:end,:);
    elseif dimension == 2
        intensity_difference = diff(I, 1, 2);

        valid_pairs = ...
            valid_mask(:,1:end-1) & ...
            valid_mask(:,2:end);
    else
        error('dimension must be 1 or 2.');
    end

    valid_differences = ...
        abs(intensity_difference(valid_pairs));

    if isempty(valid_differences)
        gradient_bound = single(0);
    else
        gradient_bound = ...
            max(valid_differences, [], 'all', 'omitnan') ...
            / max(coordinate_step, eps('single'));
    end

    gradient_bound = ...
        single(safety_factor) .* gradient_bound;

    estimated_change = min( ...
        2 .* signal_scale, ...
        gradient_bound .* abs(offsets));

    estimated_contribution = ...
        abs(weights) .* estimated_change ./ signal_scale;

    omitted_contribution = ...
        sum(estimated_contribution, 'native') ...
        - estimated_contribution(center_index);

    left_index = center_index;
    right_index = center_index;

    while omitted_contribution > rel_cutoff && ...
          (left_index > 1 || ...
           right_index < coefficient_count)

        left_contribution = single(-Inf);
        right_contribution = single(-Inf);

        if left_index > 1
            left_contribution = ...
                estimated_contribution(left_index - 1);
        end

        if right_index < coefficient_count
            right_contribution = ...
                estimated_contribution(right_index + 1);
        end

        % Add the adjacent side with the largest currently omitted
        % contribution.
        if left_contribution >= right_contribution
            left_index = left_index - 1;

            omitted_contribution = ...
                omitted_contribution ...
                - estimated_contribution(left_index);
        else
            right_index = right_index + 1;

            omitted_contribution = ...
                omitted_contribution ...
                - estimated_contribution(right_index);
        end
    end

    half_width = min( ...
        center_index - first_index, ...
        last_index - center_index);

    first_index = center_index - half_width;
    last_index  = center_index + half_width;

end


function step = positive_median_step(offsets)
%POSITIVE_MEDIAN_STEP Return a representative positive coordinate spacing.
%
% Zero, negative, NaN, and Inf differences are excluded.

    coordinate_differences = ...
        abs(diff(offsets(:)));

    coordinate_differences = ...
        coordinate_differences( ...
            isfinite(coordinate_differences) & ...
            coordinate_differences > 0);

    if isempty(coordinate_differences)
        step = single(1);
    else
        step = median(coordinate_differences);
    end
end

function [best_b, best_b_CI, best_xmax, best_y_at_xmax_CI, best_coef, best_coef_CI, info] = ...
         best_origin_quad_b_faster_bins(x, y, varargin)

%% 1. Parse Inputs
opts.nBins = 200;
opts.B = 100; 
opts.alpha = 0.05;
opts.min_pts = 10;
opts.tol_pct = 5;
opts.abs_tol = 1e-3;
opts.stop_on_tol = true;
opts.threshold = -1/6;

if ~isempty(varargin)
    for ii = 1:2:numel(varargin)
        opts.(varargin{ii}) = varargin{ii+1};
    end
end

% Standardize and Sort
x = double(x(:)); y = double(y(:));
mask = x >= 0 & ~isnan(x) & ~isnan(y) & ~isinf(x) & ~isinf(y);
x = x(mask); y = y(mask);
[x, ord] = sort(x); y = y(ord);

%% 2. Pre-calculate Cumulative Sums (The Speed Secret)
% We need components for X'X and X'y where X = [1, x, x^2]
s1  = cumsum(ones(size(x)));
sx  = cumsum(x);
sx2 = cumsum(x.^2);
sx3 = cumsum(x.^3);
sx4 = cumsum(x.^4);
sy  = cumsum(y);
sxy = cumsum(x.*y);
sx2y= cumsum(x.^2 .* y);
sy2 = cumsum(y.^2); % For fast residual calculation

% Determine bin indices
nBins = opts.nBins;
p_edges = linspace(0, 1, nBins + 1);
edges = quantile_fast(x, p_edges);
[~, bin_idx] = histc(edges(2:end), x); 
bin_idx(bin_idx == 0) = numel(x); % Handle edge case

%% 3. Analytical Search Loop (Ultra Fast)
nCuts = numel(bin_idx);
z_val = 1.96; % Approx for 95% CI without stats toolbox
best_k = 1;
stopped_early = false;

% Pre-allocate info storage
all_b = NaN(nCuts, 1);
all_y_low = NaN(nCuts, 1);

for k = 1:nCuts
    idx = bin_idx(k);
    if idx < opts.min_pts, continue; end
    
    % Construct X'X and X'y from cumsums in O(1)
    XtX = [s1(idx),  sx(idx),  sx2(idx);
           sx(idx),  sx2(idx), sx3(idx);
           sx2(idx), sx3(idx), sx4(idx)];
    
    XtY = [sy(idx); sxy(idx); sx2y(idx)];
    
    % Solve OLS
    beta = XtX \ XtY;
    
    % Analytical Error Estimation
    % MSE = (SumY^2 - beta'*X'Y) / (n - p)
    mse = (sy2(idx) - beta' * XtY) / (idx - 3);
    if mse < 0, mse = 0; end % Numerical safety
    
    % Covariance matrix
    invXtX = XtX \ eye(3);
    se = sqrt(diag(invXtX) * mse);
    
    % Predicted Y at x_max and its CI
    xk = x(idx);
    xvec = [1; xk; xk^2];
    y_val = xvec' * beta;
    y_se = sqrt((xvec' * invXtX * xvec) * mse);
    y_low = y_val - z_val * y_se;
    
    % Reliability check for b (slope)
    b_est = beta(2);
    b_se = se(2);
    rel_err = (z_val * b_se / max(abs(b_est), opts.abs_tol)) * 100;
    
    all_b(k) = b_est;
    all_y_low(k) = y_low;
    
    % Logic: Stop if we hit the curvature threshold or stability
    if opts.stop_on_tol && (y_low < opts.threshold) && (rel_err <= opts.tol_pct)
        best_k = k;
        stopped_early = true;
        break;
    end
    best_k = k;
end

%% 4. Final High-Fidelity Bootstrap (Run only ONCE)
% Now we only do the expensive resampling on the selected "best" window
final_idx = bin_idx(best_k);
xn = x(1:final_idx); yn = y(1:final_idx);
X = [ones(final_idx,1), xn, xn.^2];
XtX = X'*X;
beta = XtX \ (X'*yn);
M = XtX \ X';
resid = yn - X*beta;

% Vectorized Bootstrap
B = opts.B;
R = randi(final_idx, final_idx, B);
beta_b = beta + M * (resid(R) - mean(resid));

% Final CIs
lower_p = 100 * (opts.alpha/2);
upper_p = 100 * (1 - opts.alpha/2);

best_xmax = x(final_idx);
best_coef = beta;
best_coef_CI = [percentile_fast(beta_b(1,:), lower_p, upper_p);
                percentile_fast(beta_b(2,:), lower_p, upper_p);
                percentile_fast(beta_b(3,:), lower_p, upper_p)];
            
y_at_b = [1, best_xmax, best_xmax^2] * beta_b;
best_y_at_xmax_CI = percentile_fast(y_at_b, lower_p, upper_p);
best_b = beta(2);
best_b_CI = best_coef_CI(2,:);

% Info struct
info.stopped_early = stopped_early;
info.best_idx = final_idx;
end

%% Fast Helper Functions
function q = percentile_fast(v, p_low, p_high)
    v = sort(v); n = numel(v);
    idx = [p_low, p_high] / 100 * (n-1) + 1;
    i1 = floor(idx); i2 = min(i1+1, n); f = idx - i1;
    q = v(i1).*(1-f) + v(i2).*f;
end

function q = quantile_fast(v, p)
    v = sort(v); n = numel(v);
    r = 1 + (n-1).*p;
    i1 = floor(r); i2 = min(i1+1, n); f = r - i1;
    q = v(i1).*(1-f) + v(i2).*f;
end