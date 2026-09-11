function Results = TENOR_protocol(I_mat,qx,qy,inst,sim,ens)
%TENOR_PROTOCOL Toolbox-free TENOR-SAXS weighted-variance extraction.
%
% Results = TENOR_protocol(I_mat,qx,qy,inst,sim,ens)
% Initialization of the setup may be done by first running: 
% [inst, sim, ens, dnames] = init_TENOR_params(); 
% Then changing them according to the actual known instrumental and
% ensemble parameters, and definitition of the simulation parameters.
% The protocol:
%   1. Validates the scattering map and parameter structures.
%   2. Calls MG_extract using sim.use_r3/use_m3 and sim.use_g3.
%   3. Extracts Rg, Yg100, Yg210, Ym210, Jg10, Jg21, and Jm.
%   4. Inverts their analytical V-dependence over sim.VRange.
%   5. Propagates the full polynomial-fit covariance to each observable and
%      then to deltaV through the local analytical slope dY/dV.
%   6. Combines selected valid V estimates using sim.strategy.
%
% IMPORTANT
%   V is the SCATTERING-WEIGHTED relative variance
%
%       V = Var_w(Rg) / mean_w(Rg)^2.
%
%   It is not generally the number-weighted variance. Values V < 0 are
%   permitted only as modest analytical extrapolation diagnostics and are
%   not physical negative variances.
%
% REQUIRED EXTERNAL FUNCTION
%   MG_extract and its project dependencies must be on the MATLAB path.
%
% INPUT STRUCTURE FIELDS
%   inst.lambda or inst.WAVELENGTH
%       Wavelength in the length unit reciprocal to q. Default: 0.1.
%
%   sim.Pxn
%       [nx1 ny1 nx2 ny2], four odd Gaussian-support sizes.
%       Default: [87 85 125 123].
%
%   sim.signum
%       Gaussian support in standard deviations. Default: 4.
%
%   sim.RG2
%       Optional apparent Rg squared. Empty lets MG_extract estimate it.
%
%   sim.use_r3 or sim.use_m3
%       Include the q^6 M coefficient m3. Default: false.
%
%   sim.use_g3
%       Include the q^6 G coefficient g3. Default: false.
%
%   sim.VRange
%       Analytical lookup domain. Default: [-0.05 0.35].
%
%   sim.VGridN
%       Number of analytical lookup points. Default: 4001.
%
%   sim.observables
%       Any subset of:
%       {'Yg100','Yg210','Ym210','Jg10','Jg21','Jm'}.
%
%   sim.strategy
%       'inverseVariance' (default), 'bestSingle', 'mean', 'median', or
%       'robust'.
%
%   sim.minSlope
%       Minimum acceptable absolute analytical slope. Default: 1e-8.
%
%   sim.QRG_MAX
%       Upper Guinier edge, q*Rg_apparent < QRG_MAX. Default: 0.79.
%
%   sim.DEADPIX_FACTOR
%       Detector-border margin, in units of max(Pxn) pixels. Default: 2.
%
%   ens.nu, ens.phi2, or ens.phi_double_prime
%       Monodisperse form-factor curvature phi''.
%
% COEFFICIENT LAYOUT
%   use_g3=false, use_r3=false: [g0 g1 g2 m1 m2]
%   use_g3=false, use_r3=true : [g0 g1 g2 m1 m2 m3]
%   use_g3=true,  use_r3=false: [g0 g1 g2 g3 m1 m2]
%   use_g3=true,  use_r3=true : [g0 g1 g2 g3 m1 m2 m3]
%
% SIGN AND PSF CONVENTION
%   MG_extract fits log(F2/F). actualPSF is ordered
%   [sigma_x1 sigma_y1 sigma_x2 sigma_y2], so this protocol uses PSF2-PSF1.
%   With
%
%       log(F2/F) = A_G G + A_M M cos(2 chi),
%
%   A_G = DeltaSigma2/2 and A_M = DeltaAnisotropy2/2.
%
%   All three fitted Y ratios contain one uncancelled inverse factor A_G.
%   Therefore Yg100, Yg210, and Ym210 are all multiplied by A_G. The A_M
%   scale cancels between m2 and m1 in Ym210. The J observables cancel their
%   PSF scale completely.

narginchk(6,6);
check_inputs(I_mat,qx,qy,inst,sim,ens);
c = configuration(inst,sim,ens);

[~,RG2,Pxn,res,actualPSF] = MG_extract( ...
    c.Pxn,qx,qy,I_mat,c.signum,c.RG2,c.use_r3,c.use_g3,c.lambda, ...
    c.QRG_MAX,c.DEADPIX_FACTOR);

if ~isstruct(res) || ~isfield(res,'p') || ~isfield(res,'covP')
    error('TENOR:FitFailed', ...
        'MG_extract must return a fit structure containing p and covP.');
end

p = double(res.p(:).');
C = double(res.covP);
expectedCount = 5 + double(c.use_g3) + double(c.use_r3);
if numel(p) ~= expectedCount
    error('TENOR:FitLayout', ...
        'Expected %d coefficients but MG_extract returned %d.', ...
        expectedCount,numel(p));
end
if ~isequal(size(C),[expectedCount expectedCount]) || any(~isfinite(C(:)))
    error('TENOR:Covariance', ...
        'res.covP must be a finite %d-by-%d matrix.', ...
        expectedCount,expectedCount);
end
C = 0.5*(C+C.');

% Layout-aware coefficient extraction.
g0 = p(1);
g1 = p(2);
g2 = p(3);
mOffset = 3 + double(c.use_g3);
m1 = p(mOffset+1);
m2 = p(mOffset+2);

require_nonzero(g0,'g0');
require_nonzero(g1,'g1');
require_nonzero(m1,'m1');
if ~isscalar(RG2) || ~isfinite(RG2) || RG2 <= 0
    error('TENOR:RG2','MG_extract returned an invalid apparent Rg squared.');
end

% actualPSF=[sigma_x1 sigma_y1 sigma_x2 sigma_y2] in q units.
psf = double(actualPSF(:).');
if numel(psf) ~= 4 || any(~isfinite(psf)) || any(psf < 0)
    error('TENOR:PSF','MG_extract returned an invalid actualPSF.');
end

% MG_extract fits log(F2/F), hence the signed difference is PSF2-PSF1.
dsx2 = psf(3)^2 - psf(1)^2;
dsy2 = psf(4)^2 - psf(2)^2;
DeltaSigma2 = 0.5*(dsx2+dsy2);
DeltaAnisotropy2 = 0.5*(dsx2-dsy2);
AG = 0.5*DeltaSigma2;
AM = 0.5*DeltaAnisotropy2;
require_nonzero(AG,'DeltaSigma2/2');

% Fitted coefficient ratios. The M scale cancels in m2/m1; the remaining
% scale in Ym210 is the G scale from g0, hence AG is used for every Y.
rawYg100 = g1/g0^2;
rawYg210 = g2/(g1*g0);
rawYm210 = m2/(m1*g0);

Yg100 = AG*rawYg100;
Yg210 = AG*rawYg210;
Ym210 = AG*rawYm210;

Jg10 = (g1/g0)/RG2;
Jg21 = (g2/g1)/RG2;
Jm = (m2/m1)/RG2;

names = {'Yg100','Yg210','Ym210','Jg10','Jg21','Jm'};
observed = [Yg100 Yg210 Ym210 Jg10 Jg21 Jm];

% Delta-method propagation from the complete coefficient covariance.
D = observable_gradients(p,RG2,AG,c.use_g3);
observableSE = nan(size(observed));
for k = 1:numel(names)
    d = D.(names{k});
    variance = d*C*d.';
    observableSE(k) = sqrt(max(real(variance),0));
end

% Analytical lookup and local slope propagation.
Vgrid = linspace(c.VRange(1),c.VRange(2),c.VGridN);
calibration = analytical_theory(Vgrid,c.nu);
Vest = nan(size(observed));
deltaV = nan(size(observed));
slope = nan(size(observed));
status = cell(size(observed));

for k = 1:numel(names)
    [Vest(k),slope(k),status{k}] = invert_lookup( ...
        Vgrid,calibration.(names{k}),observed(k),c.minSlope);
    if isfinite(Vest(k)) && isfinite(slope(k)) && ...
            abs(slope(k)) >= c.minSlope
        deltaV(k) = observableSE(k)/abs(slope(k));
    end
end

requestedLower = cellfun(@lower,c.observables,'UniformOutput',false);
nameLower = cellfun(@lower,names,'UniformOutput',false);
selected = ismember(nameLower,requestedLower);
usable = selected & isfinite(Vest) & isfinite(deltaV) & deltaV > 0;
[best,bestSE,used] = combine_estimates(Vest,deltaV,usable,c.strategy);

% Pack results.
Results = struct();
Results.Rg = sqrt(RG2);
Results.RG2 = RG2;
Results.Yg100 = Yg100;
Results.Yg210 = Yg210;
Results.Ym210 = Ym210;
Results.Jg10 = Jg10;
Results.Jg21 = Jg21;
Results.Jm = Jm;
Results.DeltaSigma2 = DeltaSigma2;
Results.DeltaAnisotropy2 = DeltaAnisotropy2;
Results.AG = AG;
Results.AM = AM;
Results.actualPSF = psf;
Results.Pxn = Pxn;
Results.p = p;
Results.covP = C;
Results.fit = res;
Results.fitCoefficientLayout = coefficient_layout(c.use_g3,c.use_r3);
Results.use_r3 = c.use_r3;
Results.use_m3 = c.use_r3;
Results.use_g3 = c.use_g3;
Results.nu = c.nu;
Results.phiDoublePrime = c.nu;
Results.VRange = c.VRange;
Results.VGrid = Vgrid;
Results.weightedVariance = true;
Results.varianceDefinition = ...
    'V = Var_w(Rg)/mean_w(Rg)^2, scattering weighted';
Results.BestV = best;
Results.BestV_SE = bestSE;
Results.BestV_CI95 = best + 1.95996398454005*bestSE*[-1 1];
Results.strategy = c.strategy;
Results.requestedObservables = c.observables;
Results.usedObservables = names(used);
Results.observed = cell2struct(num2cell(observed),names,2);
Results.observableSE = cell2struct(num2cell(observableSE),names,2);
Results.V = cell2struct(num2cell(Vest),names,2);
Results.deltaV = cell2struct(num2cell(deltaV),names,2);
Results.dVdObservable = cell2struct(num2cell(1./slope),names,2);
Results.calibration = calibration;
Results.diagnostics.status = cell2struct(status,names,2);
Results.diagnostics.selectedMask = selected;
Results.diagnostics.usableMask = usable;
Results.diagnostics.fitR2 = get_value(res,{'R2w'},NaN);
Results.diagnostics.negativeBestVIsExtrapolation = isfinite(best) && best < 0;
end

function check_inputs(I,qx,qy,inst,sim,ens)
if ~isnumeric(I) || ~ismatrix(I) || isempty(I) || ~isreal(I)
    error('TENOR:Input','I_mat must be a nonempty real numeric 2-D array.');
end
if ~isnumeric(qx) || ~isnumeric(qy) || ~isreal(qx) || ~isreal(qy) || ...
        ~isequal(size(I),size(qx),size(qy))
    error('TENOR:Grid','I_mat, qx, and qy must be real arrays of equal size.');
end
if ~isstruct(inst) || ~isscalar(inst) || ...
        ~isstruct(sim) || ~isscalar(sim) || ...
        ~isstruct(ens) || ~isscalar(ens)
    error('TENOR:Struct','inst, sim, and ens must be scalar structures.');
end
finiteQ = isfinite(qx) & isfinite(qy);
valid = finiteQ & isfinite(I) & I > 0;
if nnz(valid) < 50
    error('TENOR:Samples', ...
        'At least 50 positive finite intensity pixels are required.');
end
if any(I(finiteQ) < 0)
    error('TENOR:Intensity', ...
        'Negative finite intensities are invalid for logarithmic fitting.');
end
if max(hypot(qx(valid),qy(valid))) <= 0
    error('TENOR:q','The q grid must contain nonzero radial values.');
end
end

function c = configuration(inst,sim,ens)
c.lambda = get_value(inst,{'lambda','WAVELENGTH'},0.1);
c.Pxn = get_value(sim,{'Pxn'},[87 85 125 123]);
c.signum = get_value(sim,{'signum'},4);
c.RG2 = get_value(sim,{'RG2'},[]);
c.use_r3 = logical(get_value(sim,{'use_r3','use_m3'},false));
c.use_g3 = logical(get_value(sim,{'use_g3'},false));
c.VRange = get_value(sim,{'VRange','V_range'},[-0.05 0.35]);
c.VGridN = get_value(sim,{'VGridN','nV'},4001);
c.observables = get_value(sim,{'observables','useObservables'}, ...
    {'Yg100','Yg210','Ym210','Jg10','Jg21','Jm'});
c.strategy = get_value(sim,{'strategy','choiceStrategy'}, ...
    'inverseVariance');
c.minSlope = get_value(sim,{'minSlope'},1e-8);
c.QRG_MAX = get_value(sim,{'QRG_MAX','qrgMax','qRgLimit'},0.79);
c.DEADPIX_FACTOR = get_value(sim,{'DEADPIX_FACTOR','deadpixFactor'},2);
c.nu = get_value(ens,{'nu','phi2','phi_double_prime'},NaN);

validateattributes(c.lambda,{'numeric'},{'scalar','finite','positive'});
validateattributes(c.Pxn,{'numeric'}, ...
    {'vector','numel',4,'integer','positive','finite'});
if any(mod(c.Pxn,2) ~= 1)
    error('TENOR:Pxn','All Pxn entries must be odd integers.');
end
validateattributes(c.signum,{'numeric'},{'scalar','finite','positive'});
validateattributes(c.QRG_MAX,{'numeric'},{'scalar','finite','positive'});
validateattributes(c.DEADPIX_FACTOR,{'numeric'},{'scalar','finite','nonnegative'});
if ~isempty(c.RG2)
    validateattributes(c.RG2,{'numeric'},{'scalar','finite','positive'});
end
if ~isscalar(c.use_r3) || ~isscalar(c.use_g3)
    error('TENOR:FitOrder', ...
        'sim.use_r3/use_m3 and sim.use_g3 must be scalar flags.');
end
validateattributes(c.VRange,{'numeric'},{'vector','numel',2,'finite'});
c.VRange = double(c.VRange(:).');
if c.VRange(1) >= c.VRange(2) || c.VRange(1) <= -1
    error('TENOR:VRange', ...
        'VRange must be increasing and must remain above the pole V=-1.');
end
validateattributes(c.VGridN,{'numeric'},{'scalar','integer','>=',101});
validateattributes(c.minSlope,{'numeric'},{'scalar','finite','positive'});
if ~isfinite(c.nu) || ~isscalar(c.nu)
    error('TENOR:nu', ...
        'ens.nu, the scalar monodisperse form-factor curvature, is required.');
end

if ischar(c.observables)
    c.observables = {c.observables};
elseif isstring(c.observables)
    c.observables = cellstr(c.observables);
end
if ~iscell(c.observables) || isempty(c.observables) || ...
        ~all(cellfun(@ischar,c.observables))
    error('TENOR:Observables', ...
        'sim.observables must be a nonempty cell array of names.');
end
allowed = {'yg100','yg210','ym210','jg10','jg21','jm'};
requested = cellfun(@lower,c.observables,'UniformOutput',false);
if any(~ismember(requested,allowed))
    error('TENOR:Observables','Unknown observable selection.');
end
if ~ischar(c.strategy) && ~(isstring(c.strategy) && isscalar(c.strategy))
    error('TENOR:Strategy','sim.strategy must be a character vector or scalar string.');
end
c.strategy = char(c.strategy);
validStrategies = {'inverseVariance','bestSingle','mean','median','robust'};
if ~any(strcmpi(c.strategy,validStrategies))
    error('TENOR:Strategy','Unknown combination strategy.');
end
end

function T = analytical_theory(V,nu)
% First-order analytical observable equations.
A = 1 + 18*nu + 10*V + 108*V*nu;
B = 1 + 9*nu + 6*V + 54*V*nu;
NG = 18*nu + 8*V + 450*V*nu;
NM = 18*nu + 8*V + 342*V*nu;

T.Yg100 = A./(4*(1+V).^2);
T.Yg210 = NG./(4*(1+V).*A);
T.Ym210 = NM./(4*(1+V).*B);
T.Jg10 = A./(-3*(1+V).^2);
T.Jg21 = NG./(-3*(1+V).*A);
T.Jm = NM./(-3*(1+V).*B);
end

function G = observable_gradients(p,R2,AG,use_g3)
% Gradients of the observed quantities with respect to fitted coefficients.
n = numel(p);
g0 = p(1);
g1 = p(2);
g2 = p(3);
mOffset = 3 + double(use_g3);
m1 = p(mOffset+1);
m2 = p(mOffset+2);
zero = @() zeros(1,n);

% Yg100 = AG*g1/g0^2.
d = zero();
d(1) = -2*AG*g1/g0^3;
d(2) = AG/g0^2;
G.Yg100 = d;

% Yg210 = AG*g2/(g1*g0).
d = zero();
d(1) = -AG*g2/(g1*g0^2);
d(2) = -AG*g2/(g0*g1^2);
d(3) = AG/(g0*g1);
G.Yg210 = d;

% Ym210 = AG*m2/(m1*g0).
d = zero();
d(1) = -AG*m2/(m1*g0^2);
d(mOffset+1) = -AG*m2/(g0*m1^2);
d(mOffset+2) = AG/(g0*m1);
G.Ym210 = d;

% Jg10 = (g1/g0)/R2.
d = zero();
d(1) = -g1/(g0^2*R2);
d(2) = 1/(g0*R2);
G.Jg10 = d;

% Jg21 = (g2/g1)/R2.
d = zero();
d(2) = -g2/(g1^2*R2);
d(3) = 1/(g1*R2);
G.Jg21 = d;

% Jm = (m2/m1)/R2.
d = zero();
d(mOffset+1) = -m2/(m1^2*R2);
d(mOffset+2) = 1/(m1*R2);
G.Jm = d;
end

function [V,slope,status] = invert_lookup(Vgrid,Ygrid,Yobserved,minSlope)
% Invert one continuous monotonic analytical branch without extrapolating
% beyond Vgrid. The extended default Vgrid itself supplies modest diagnostic
% extrapolation beyond the nominal physical interval.
V = NaN;
slope = NaN;
status = 'invalid observable';
valid = isfinite(Vgrid) & isfinite(Ygrid);
Vgrid = Vgrid(valid);
Ygrid = Ygrid(valid);
if ~isfinite(Yobserved) || numel(Vgrid) < 3
    return;
end

% Split at derivative sign changes and singular/non-monotonic points.
dY = gradient(Ygrid,Vgrid);
signSlope = sign(dY);
signSlope(abs(dY) < minSlope) = 0;
breaks = [1, find(signSlope(2:end).*signSlope(1:end-1) <= 0)+1, numel(Vgrid)+1];

candidates = [];
candidateSlopes = [];
for b = 1:numel(breaks)-1
    idx = breaks(b):(breaks(b+1)-1);
    if numel(idx) < 3
        continue;
    end
    Vbranch = Vgrid(idx);
    Ybranch = Ygrid(idx);
    [Ysorted,order] = sort(Ybranch);
    Vsorted = Vbranch(order);
    [Yunique,uniqueIndex] = unique(Ysorted,'stable');
    Vunique = Vsorted(uniqueIndex);
    if numel(Yunique) < 3 || Yobserved < Yunique(1) || Yobserved > Yunique(end)
        continue;
    end
    value = interp1(Yunique,Vunique,Yobserved,'pchip');
    localSlope = interp1(Vgrid,dY,value,'linear');
    if isfinite(value) && isfinite(localSlope)
        candidates(end+1) = value; %#ok<AGROW>
        candidateSlopes(end+1) = localSlope; %#ok<AGROW>
    end
end

if isempty(candidates)
    status = 'outside calibration range or no monotonic branch';
    return;
elseif numel(candidates) > 1
    % Multiple roots are genuinely ambiguous. Prefer no silent branch choice.
    status = 'multiple analytical roots';
    return;
end

V = candidates(1);
slope = candidateSlopes(1);
if abs(slope) < minSlope
    status = 'calibration locally flat';
else
    status = 'ok';
end
end

function [best,se,used] = combine_estimates(V,dV,usable,strategy)
used = false(size(usable));
best = NaN;
se = NaN;
indices = find(usable);
if isempty(indices)
    return;
end

switch lower(strategy)
    case 'bestsingle'
        [se,j] = min(dV(indices));
        used(indices(j)) = true;
        best = V(indices(j));

    case 'mean'
        used(indices) = true;
        best = mean(V(indices));
        se = sqrt(sum(dV(indices).^2))/numel(indices);

    case 'median'
        used(indices) = true;
        best = median(V(indices));
        se = 1.2533141373155*median(dV(indices))/sqrt(numel(indices));

    case 'robust'
        centre = median(V(indices));
        madValue = median(abs(V(indices)-centre));
        if madValue > 0
            keep = abs(V(indices)-centre) <= 3*1.4826022185056*madValue;
            indices = indices(keep);
        end
        if isempty(indices)
            return;
        end
        used(indices) = true;
        weights = 1./dV(indices).^2;
        best = sum(weights.*V(indices))/sum(weights);
        se = sqrt(1/sum(weights));

    otherwise % inverseVariance
        used(indices) = true;
        weights = 1./dV(indices).^2;
        best = sum(weights.*V(indices))/sum(weights);
        se = sqrt(1/sum(weights));
end
end

function labels = coefficient_layout(use_g3,use_r3)
labels = {'g0','g1','g2'};
if use_g3
    labels{end+1} = 'g3';
end
labels = [labels {'m1','m2'}];
if use_r3
    labels{end+1} = 'm3';
end
end

function require_nonzero(value,name)
threshold = 100*eps(max(1,abs(value)));
if ~isfinite(value) || abs(value) <= threshold
    error('TENOR:Singular','%s is zero or numerically singular.',name);
end
end

function value = get_value(s,names,defaultValue)
value = defaultValue;
for k = 1:numel(names)
    if isfield(s,names{k}) && ~isempty(s.(names{k}))
        value = s.(names{k});
        return;
    end
end
end
