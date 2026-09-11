function [Results_Table,cfg]=run_TENOR_benchmark(varargin)
% End-to-end benchmark. 
% Creates a noise free set of intensity maps. 
% Adds noise.
% Runs the TENOR-SAX protocol and outputs the Results_Table
% example:
% run the benchmark (only 3 repetitions for to make it quick (~1/2 minute)
% [Results_Table,cfg] = run_TENOR_benchmark('outputRoot','D:\SAXS_DATA\test','nReplicates',3);
%
% After you can visualize results with the violin plot
% plot_tenor_violin_1dGT(Results_Table, cfg.instrument)
%
%
cfg=TENOR_benchmark_setup(varargin{:});
if ~exist(cfg.outputRoot,'dir'),mkdir(cfg.outputRoot);end;
diary(cfg.logFile);
c=onCleanup(@()diary('off'));
TENOR_generate_clean_database(cfg);
Results_Table=TENOR_run_noise_benchmark(cfg);
end

function cfg=TENOR_benchmark_setup(varargin)
% Parameterized Figure-4/Appendix-C benchmark setup.
p=inputParser;
addParameter(p,'outputRoot',fullfile(pwd,'TENOR_benchmark'));
addParameter(p,'overwrite',false);
addParameter(p,'seed',314159);
addParameter(p,'nReplicates',30); %100
parse(p,varargin{:});
cfg.outputRoot=char(p.Results.outputRoot);
cfg.mapsFile=fullfile(cfg.outputRoot,'noise_free_maps.mat');
cfg.resultsFile=fullfile(cfg.outputRoot,'benchmark_results.mat');
cfg.logFile=fullfile(cfg.outputRoot,'benchmark_log.txt');
cfg.overwrite=p.Results.overwrite;
cfg.seed=p.Results.seed;
cfg.nReplicates=p.Results.nReplicates;
cfg.instrument=struct('SD_dist',360,'lambda',0.1,'det_side',3.5,'DETpix',500,'PSF0',bartlett_local(3,15));
% [uri2x-15]: 12-value V grid, linspace(.01,.55,12).^2 (was (0.01:.05:.55).^2, 11 values).
cfg.ensemble=struct('rg',3,'V',linspace(.01,.55,12).^2,'phi2',+1/18,'distribution','normal','dist_param',struct('N',25),'weightPower',0);
cfg.ensemble.targetOptions=struct('pMinimum',1e-5,'pMaximum',5,'varianceTolerance',2e-4,'maximumIterations',50,'generatorCoverage',.995,'applyWeightAgain',true,'expansionFactor',1.6);
% [uri2x-15]: seven half-decade photon Q-densities N_q2=1e6..1e9 photons/nm^-2
% (instrument-geometry-only -- see plot_tenor_violin_1dGT.m's dq formula),
% converted to peakPhotons = N_q2 * dq^2 at this benchmark's own geometry.
% (Was 6 levels via 10.^(2.5:.5:5)/1.65, i.e. N_q2 = 3.2e7..1.0e10.)
dq=4*pi/cfg.instrument.lambda*cfg.instrument.det_side/cfg.instrument.SD_dist/(2*round(cfg.instrument.DETpix/2)+1);
cfg.noise=struct('peakPhotons',10.^(6:0.5:9)*dq^2,'clipNegative',true);
% [uri2x-15]'s chosen quartet, [91 81 117 127] (was [85 75 111 125]); window
% now q*R0_tilde<1.0, margin 1x (was 0.79/2).
cfg.tenor=struct('Pxn',[91 81 117 127],'signum',4,'use_r3',false,'use_g3',false,'VRange',[-.05 .35],'VGridN',4001,'strategy','inverseVariance','QRG_MAX',1.0,'DEADPIX_FACTOR',1,'observables',{{'Yg100'}});
[V,P]=ndgrid(cfg.ensemble.V(:),cfg.ensemble.phi2(:));
cfg.caseTable=table((1:numel(V))',V(:),P(:),sqrt(V(:)),'VariableNames',{'CaseID','True_V','Phi2','True_p'});
end
function K=bartlett_local(n,m)
y=(0:n-1)';x=(0:m-1)';a=1-abs(y-(n-1)/2)/((n+1)/2);d=1-abs(x-(m-1)/2)/((m+1)/2);K=a*d';K=K/sum(K(:));
end

function manifest = TENOR_generate_clean_database(cfg)
%TENOR_GENERATE_CLEAN_DATABASE Generate or reuse noise-free scattering maps.
%
%   MANIFEST = TENOR_GENERATE_CLEAN_DATABASE(CFG) compares the clean
%   Scatter2D inputs in CFG with the inputs stored in CFG.mapsFile.
%
%   The existing clean database is reused only when all parameters that can
%   affect Scatter2D or the effective-distribution targeting are unchanged.
%   Noise-generation parameters and TENOR analysis parameters are ignored.
%
%   Parameters included in the comparison:
%       cfg.instrument.SD_dist
%       cfg.instrument.lambda
%       cfg.instrument.det_side
%       cfg.instrument.DETpix
%       cfg.instrument.PSF0
%       cfg.ensemble.rg
%       cfg.ensemble.distribution
%       cfg.ensemble.dist_param
%       cfg.ensemble.weightPower
%       cfg.ensemble.targetOptions
%       cfg.caseTable columns used for clean scattering
%
%   Parameters intentionally excluded:
%       cfg.noise
%       cfg.tenor
%       cfg.seed
%       cfg.nReplicates
%       output and logging options
%
%   If the stored parameters differ, or if the stored database is incomplete,
%   the old file is deleted and recreated. Cases for which the effective
%   distribution or Scatter2D fails are marked as "skipped".

validate_clean_configuration(cfg);

if ~exist(cfg.outputRoot, 'dir')
    mkdir(cfg.outputRoot);
end

cleanParameters = extract_clean_parameters(cfg);

[reuseExisting, reason] = can_reuse_database( ...
    cfg.mapsFile, cleanParameters, cfg.caseTable);

if reuseExisting
    stored = load(cfg.mapsFile, 'manifest');
    manifest = stored.manifest;

    fprintf('Reusing clean database: %s\n', cfg.mapsFile);
    return;
end

if exist(cfg.mapsFile, 'file')
    fprintf('Recreating clean database: %s\n', reason);
    delete(cfg.mapsFile);
else
    fprintf('Creating clean database: %s\n', reason);
end

caseTable = cfg.caseTable;
numberOfCases = height(caseTable);

manifest = caseTable;
manifest.Status = repmat("pending", numberOfCases, 1);
manifest.Message = repmat("", numberOfCases, 1);
manifest.InputVariance = nan(numberOfCases, 1);
manifest.RequestedRg = nan(numberOfCases, 1);
manifest.RealizedV = nan(numberOfCases, 1);
manifest.PredictedObservedRg = nan(numberOfCases, 1);
manifest.NumericalP = nan(numberOfCases, 1);
manifest.WeightedRMSFactor = nan(numberOfCases, 1);

mapFile = matfile(cfg.mapsFile, 'Writable', true);
gridWritten = false;

for caseIndex = 1:numberOfCases
    fprintf( ...
        'Clean map %d/%d: target V = %.6g, phi2 = %.6g\n', ...
        caseIndex, ...
        numberOfCases, ...
        caseTable.True_V(caseIndex), ...
        caseTable.Phi2(caseIndex));

    try
        target = target_effective_distribution( ...
            caseTable.True_V(caseIndex), ...
            cfg.ensemble.rg, ...
            cfg.ensemble.distribution, ...
            cfg.ensemble.dist_param.N, ...
            cfg.ensemble.weightPower, ...
            cfg.ensemble.targetOptions);

        manifest.InputVariance(caseIndex) = target.inputVariance;
        manifest.RequestedRg(caseIndex) = target.requestedRg;
        manifest.RealizedV(caseIndex) = target.realizedV;
        manifest.PredictedObservedRg(caseIndex) = ...
            target.predictedObservedRg;
        manifest.NumericalP(caseIndex) = target.pNumerical;
        manifest.WeightedRMSFactor(caseIndex) = ...
            target.weightedRMSFactor;

        [qx, qy, intensity] = Scatter2D( ...
            target.requestedRg, ...
            0, ...
            target.inputVariance, ...
            caseTable.Phi2(caseIndex), ...
            cfg.instrument.DETpix, ...
            cfg.instrument.SD_dist, ...
            cfg.instrument.lambda, ...
            cfg.instrument.det_side, ...
            cfg.instrument.PSF0, ...
            cfg.ensemble.distribution, ...
            cfg.ensemble.dist_param, ...
            cfg.ensemble.weightPower);

        if ~gridWritten
            mapFile.qx = single(qx);
            mapFile.qy = single(qy);
            mapFile.I_clean( ...
                size(intensity, 1), ...
                size(intensity, 2), ...
                numberOfCases) = single(0);

            gridWritten = true;
        else
            storedQxSize = size(mapFile, 'qx');
            storedQySize = size(mapFile, 'qy');

            if ~isequal(size(qx), storedQxSize) || ...
                    ~isequal(size(qy), storedQySize)
                error( ...
                    'TENOR:GridChanged', ...
                    'The q-grid size changed between clean cases.');
            end
        end

        mapFile.I_clean(:, :, caseIndex) = single(intensity);
        manifest.Status(caseIndex) = "ok";
    catch ME
        manifest.Status(caseIndex) = "skipped";
        manifest.Message(caseIndex) = string(ME.message);

        fprintf( ...
            2, ...
            'Skipping clean case %d: %s\n', ...
            caseIndex, ...
            ME.message);
    end
end

if ~gridWritten
    if exist(cfg.mapsFile, 'file')
        delete(cfg.mapsFile);
    end

    error( ...
        'TENOR:NoValidCases', ...
        'No case produced a valid distribution and clean intensity map.');
end

mapFile.cleanParameters = cleanParameters;
mapFile.manifest = manifest;
mapFile.cfg = cfg;
end

function cleanParameters = extract_clean_parameters(cfg)
%EXTRACT_CLEAN_PARAMETERS Return only inputs affecting clean map generation.

cleanParameters = struct();
cleanParameters.schemaVersion = 1;

cleanParameters.instrument = struct();
cleanParameters.instrument.SD_dist = cfg.instrument.SD_dist;
cleanParameters.instrument.lambda = cfg.instrument.lambda;
cleanParameters.instrument.det_side = cfg.instrument.det_side;
cleanParameters.instrument.DETpix = cfg.instrument.DETpix;
cleanParameters.instrument.PSF0 = cfg.instrument.PSF0;

cleanParameters.ensemble = struct();
cleanParameters.ensemble.rg = cfg.ensemble.rg;
cleanParameters.ensemble.distribution = cfg.ensemble.distribution;
cleanParameters.ensemble.dist_param = cfg.ensemble.dist_param;
cleanParameters.ensemble.weightPower = cfg.ensemble.weightPower;
cleanParameters.ensemble.targetOptions = cfg.ensemble.targetOptions;

% Store only columns actually passed to target_effective_distribution or
% Scatter2D. Extra plotting, labeling, noise, and analysis columns are ignored.
cleanParameters.cases = table();
cleanParameters.cases.True_V = cfg.caseTable.True_V;
cleanParameters.cases.Phi2 = cfg.caseTable.Phi2;
end

function [canReuse, reason] = can_reuse_database( ...
    mapsFile, requestedCleanParameters, requestedCaseTable)
%CAN_REUSE_DATABASE Check parameter equality and database completeness.

canReuse = false;
reason = "clean database does not exist";

if ~exist(mapsFile, 'file')
    return;
end

fileVariables = whos('-file', mapsFile);
variableNames = string({fileVariables.name});

requiredVariables = [ ...
    "cleanParameters", ...
    "manifest", ...
    "qx", ...
    "qy", ...
    "I_clean"];

if ~all(ismember(requiredVariables, variableNames))
    reason = "stored file is missing required clean-database variables";
    return;
end

stored = load(mapsFile, 'cleanParameters', 'manifest');

if ~isequaln(stored.cleanParameters, requestedCleanParameters)
    reason = "stored clean Scatter2D parameters differ from the requested parameters";
    return;
end

if height(stored.manifest) ~= height(requestedCaseTable)
    reason = "stored manifest has a different number of clean cases";
    return;
end

if ~all(ismember({'Status'}, stored.manifest.Properties.VariableNames))
    reason = "stored manifest has no Status column";
    return;
end

mapInfo = whos('-file', mapsFile, 'I_clean');

if isempty(mapInfo) || numel(mapInfo.size) < 3
    reason = "stored clean intensity array is invalid";
    return;
end

if mapInfo.size(3) ~= height(requestedCaseTable)
    reason = "stored clean intensity array has a different case count";
    return;
end

allowedStatus = stored.manifest.Status == "ok" | ...
    stored.manifest.Status == "skipped";

if ~all(allowedStatus)
    reason = "stored clean database contains unfinished cases";
    return;
end

canReuse = true;
reason = "stored clean database matches the requested clean parameters";
end

function validate_clean_configuration(cfg)
%VALIDATE_CLEAN_CONFIGURATION Validate fields required before comparison.

requiredTopLevelFields = { ...
    'outputRoot', ...
    'mapsFile', ...
    'instrument', ...
    'ensemble', ...
    'caseTable'};

for fieldIndex = 1:numel(requiredTopLevelFields)
    fieldName = requiredTopLevelFields{fieldIndex};

    if ~isfield(cfg, fieldName)
        error( ...
            'TENOR:Configuration', ...
            'Missing configuration field cfg.%s.', ...
            fieldName);
    end
end

requiredInstrumentFields = { ...
    'SD_dist', ...
    'lambda', ...
    'det_side', ...
    'DETpix', ...
    'PSF0'};

for fieldIndex = 1:numel(requiredInstrumentFields)
    fieldName = requiredInstrumentFields{fieldIndex};

    if ~isfield(cfg.instrument, fieldName)
        error( ...
            'TENOR:Configuration', ...
            'Missing configuration field cfg.instrument.%s.', ...
            fieldName);
    end
end

requiredEnsembleFields = { ...
    'rg', ...
    'distribution', ...
    'dist_param', ...
    'weightPower', ...
    'targetOptions'};

for fieldIndex = 1:numel(requiredEnsembleFields)
    fieldName = requiredEnsembleFields{fieldIndex};

    if ~isfield(cfg.ensemble, fieldName)
        error( ...
            'TENOR:Configuration', ...
            'Missing configuration field cfg.ensemble.%s.', ...
            fieldName);
    end
end

requiredCaseColumns = {'True_V', 'Phi2'};

if ~all(ismember(requiredCaseColumns, cfg.caseTable.Properties.VariableNames))
    error( ...
        'TENOR:Configuration', ...
        'cfg.caseTable must contain True_V and Phi2 columns.');
end

if exist('Scatter2D', 'file') ~= 2
    error( ...
        'TENOR:MissingFunction', ...
        'Scatter2D.m is not on the MATLAB path.');
end

if exist('target_effective_distribution', 'file') ~= 2
    error( ...
        'TENOR:MissingFunction', ...
        'target_effective_distribution.m is not on the MATLAB path.');
end

if exist('size_distribution_discrete_no_max', 'file') ~= 2
    error( ...
        'TENOR:MissingFunction', ...
        'size_distribution_discrete_no_max.m is not on the MATLAB path.');
end
end

function I=TENOR_add_noise(I0,level,varargin)
% Scatter2D-compatible noise on an existing normalized intensity map.
p=inputParser;addParameter(p,'seed',[]);addParameter(p,'clipNegative',true);parse(p,varargin{:});
if ~isempty(p.Results.seed),rng(double(p.Results.seed),'twister');end;
I=double(I0);
if level<0,
    I=I+sqrt(max(I,0)/abs(level)).*randn(size(I));
else,
    I=I+level*randn(size(I));
end;
if p.Results.clipNegative,I(isfinite(I)&I<0)=0;end;I=cast(I,'like',I0);
end
%%
function Results_Table=TENOR_run_noise_benchmark(cfg)
% Add fresh noise, run TENOR, and return violin-compatible rows.
if ~exist(cfg.mapsFile,'file'),TENOR_generate_clean_database(cfg);end;
D=matfile(cfg.mapsFile);
S=load(cfg.mapsFile,'manifest');
ok=find(S.manifest.Status=="ok");
C=cfg.caseTable(ok,:);n=height(C)*numel(cfg.noise.peakPhotons)*cfg.nReplicates;CaseID=zeros(n,1);Replicate=CaseID;Noise=CaseID;True_p=CaseID;True_V=CaseID;
Phi2=CaseID;True_Rg=CaseID;V_est=nan(n,1);p_est=V_est;Rg_in_est=V_est;Valid=false(n,1);
Seed=CaseID;Status=repmat("pending",n,1);Message=repmat("",n,1);Result=cell(n,1);z=0;
for c=1:height(C),I0=D.I_clean(:,:,ok(c));
    for j=1:numel(cfg.noise.peakPhotons),
        for r=1:cfg.nReplicates,
            z=z+1;ph=cfg.noise.peakPhotons(j);
            sd=mod(double(cfg.seed)+104729*ok(c)+1009*j+37*r,2^32-1);
            if sd==0,sd=1;end;
            CaseID(z)=C.CaseID(c);Replicate(z)=r;Noise(z)=-ph;True_p(z)=C.True_p(c);True_V(z)=C.True_V(c);
            Phi2(z)=C.Phi2(c);True_Rg(z)=cfg.ensemble.rg;Seed(z)=sd;
            try,
                I=TENOR_add_noise(I0,-ph,'seed',sd,'clipNegative',cfg.noise.clipNegative);
                ens=cfg.ensemble;
                ens.nu=C.Phi2(c);
                sim=cfg.tenor;
                sim.RG2=[];
                R=TENOR_protocol(I,D.qx,D.qy,cfg.instrument,sim,ens);
                R=rmfield(R,'calibration');
                R=rmfield(R,'VGrid');
                Result{z}=R;
                V_est(z)=R.BestV;
                p_est(z)=sqrt(max(R.BestV,0));
                Rg_in_est(z)=R.Rg;
                Valid(z)=isfinite(R.BestV);
                Status(z)="ok";
            catch ME,Status(z)="failed";
                Message(z)=string(ME.message);
            end
            fprintf('.');
        end,
        fprintf(':\n');
    end,
    fprintf(';');
end
fprintf('\n');
Results_Table=table(CaseID,Replicate,Noise,True_p,True_V,Phi2,True_Rg,p_est,V_est,Rg_in_est,Valid,Seed,Status,Message,Result);
save(cfg.resultsFile,'Results_Table','cfg','-v7.3');
end

%%

function [x,p] = size_distribution_discrete_no_max(N,Vrel,dist_type,threshold,xmin,weight_power)
%SIZE_DISTRIBUTION_DISCRETE_NO_MAX Discretize an unbounded size distribution.
% Uses weighted-CDF binning and an affine moment correction. This standalone
% file is required by target_effective_distribution and Scatter2D.
if nargin<3||isempty(dist_type),dist_type='normal';end
if nargin<4||isempty(threshold),threshold=0.9995;end
if nargin<5||isempty(xmin),xmin=0;end
if nargin<6||isempty(weight_power),weight_power=0;end
validateattributes(N,{'numeric'},{'scalar','integer','>=',3});
validateattributes(Vrel,{'numeric'},{'scalar','finite','positive'});
Mean=1;sigma=sqrt(Vrel);
if any(strcmpi(dist_type,{'lognormal','schulz','boltzmann'}))
 s0=sqrt(log(1+Vrel));m0=log(Mean)-0.5*s0^2+weight_power*s0^2;xmax=exp(m0+5*s0);
else
 xmax=Mean+20*sigma;
end
u=linspace(xmin,xmax,10000)';
switch lower(dist_type)
 case 'normal', f=exp(-(u-Mean).^2/(2*sigma^2));
 case 'lognormal'
  s=sqrt(log(1+Vrel));m=log(Mean)-0.5*s^2;f=(1./u).*exp(-(log(u)-m).^2/(2*s^2));
 case 'schulz'
  z=1/Vrel-1;lf=z*log(u)-(z+1)*u/Mean;lf(isnan(lf))=-inf;lf=lf-max(lf);f=exp(lf);
 case {'boltzmann','exponential'}, f=exp(-sqrt(2)*abs(u-Mean)/sigma);
 case 'triangular', L=sigma*sqrt(6);f=max(0,1-abs(u-Mean)/L);
 case 'uniform', L=sigma*sqrt(3);f=double(u>=Mean-L & u<=Mean+L);
 otherwise, error('TENOR:DistributionType','Unsupported distribution: %s.',dist_type);
end
f(~isfinite(f))=0;if sum(f)<=0,error('TENOR:Distribution','Continuous PDF is invalid.');end;f=f/sum(f);
fw=f.*u.^weight_power;if ~all(isfinite(fw))||sum(fw)<=0,error('TENOR:Weights','Weighted PDF is invalid.');end
Cw=cumsum(fw)/sum(fw);Cn=cumsum(f);Cw=Cw+linspace(0,1e-11,numel(Cw))';
[Cwu,ia]=unique(Cw,'stable');uu=u(ia);if numel(Cwu)<2,error('TENOR:Distribution','Weighted CDF is degenerate.');end
edges=interp1(Cwu,uu,linspace(0,threshold,N+1),'linear','extrap');
Cm=cumsum(f.*u);x=zeros(N,1);p=zeros(N,1);
for i=1:N
 p1=interp1(u,Cn,edges(i+1),'linear','extrap');p0=interp1(u,Cn,edges(i),'linear','extrap');p(i)=p1-p0;
 m1=interp1(u,Cm,edges(i+1),'linear','extrap');m0=interp1(u,Cm,edges(i),'linear','extrap');
 if ~isfinite(p(i))||p(i)<=0,error('TENOR:Distribution','Empty/nonfinite discrete bin %d.',i);end
 x(i)=(m1-m0)/p(i);
end
p=p/sum(p);mu=sum(p.*x);v=sum(p.*(x-mu).^2);if ~isfinite(v)||v<=0,error('TENOR:Distribution','Discrete variance is invalid.');end
x=sqrt(Vrel/v)*(x-mu)+Mean;x=max(x,xmin);
if any(~isfinite(x))||any(x<0),error('TENOR:Distribution','Corrected radii are invalid.');end
p=p/sum(p);
end

%%

function S=target_effective_distribution(targetV,targetObservedRg,distName,N,w,opt)
% Match effective scattering-weighted V and RMS Rg; never test p=0.
if nargin<6||isempty(opt),opt=struct;end
pmin=gv(opt,'pMinimum',1e-5); pmax=gv(opt,'pMaximum',5); tol=gv(opt,'varianceTolerance',2e-4); nit=gv(opt,'maximumIterations',50); cov=gv(opt,'generatorCoverage',.995); again=gv(opt,'applyWeightAgain',true); grow=gv(opt,'expansionFactor',1.6);
fun=@(x)trial(x,distName,N,w,cov,again); lo=pmin; L=fun(lo);
if targetV<=L.V, pn=lo; C=L; else
 hi=min(max(sqrt(targetV),2*pmin),pmax); H=safe(fun,hi);
 while (~H.valid||H.stats.V<targetV)&&hi<pmax
  old=hi;hi=min(pmax,max(grow*hi,hi+.02));H=safe(fun,hi);
  if ~H.valid,[hi,H]=lastvalid(fun,old,hi,nit);break,end
 end
 if ~H.valid||H.stats.V<targetV,error('TENOR:VarianceTargetUnreachable','Target effective V %.6g is unreachable for %s, N=%d.',targetV,distName,N);end
 for k=1:nit
  mid=(lo+hi)/2;M=safe(fun,mid);if ~M.valid,hi=mid;continue,end
  if abs(M.stats.V-targetV)/max(targetV,1e-8)<=tol,lo=mid;hi=mid;break;elseif M.stats.V<targetV,lo=mid;else,hi=mid;end
 end
 pn=(lo+hi)/2;C=fun(pn);
end
rr=targetObservedRg/C.rms;S=struct('targetV',targetV,'realizedV',C.V,'pNumerical',pn,'inputVariance',pn^2,'requestedRg',rr,'targetObservedRg',targetObservedRg,'predictedObservedRg',rr*C.rms,'weightedRMSFactor',C.rms,'r',C.r,'p',C.p);
end
function s=trial(pn,d,N,w,cov,again)
[r,p]=size_distribution_discrete_no_max(N,pn^2,d,cov,0,w);r=double(r(:));p=double(p(:));v=isfinite(r)&isfinite(p)&r>=0&p>=0;r=r(v);p=p(v);if numel(r)<3||sum(p)<=0,error('TENOR:Distribution','Invalid generated distribution.');end;p=p/sum(p);
if again,z=max(r);p=p.*(r/z).^w;if ~all(isfinite(p))||sum(p)<=0,error('TENOR:Weights','Invalid weighted distribution.');end;p=p/sum(p);end
mu=sum(p.*r);if ~isfinite(mu)||mu<=0,error('TENOR:Distribution','Invalid weighted mean.');end;s=struct('V',sum(p.*(r-mu).^2)/mu^2,'rms',sqrt(sum(p.*r.^2)),'r',r,'p',p);
end
function o=safe(f,p),try,o=struct('valid',true,'stats',f(p));catch,o=struct('valid',false,'stats',struct('V',NaN));end,end
function [p,b]=lastvalid(f,a,z,n),b=safe(f,a);if ~b.valid,error('TENOR:DistributionSearch','Lower point invalid.');end;for k=1:n,m=(a+z)/2;t=safe(f,m);if t.valid,a=m;b=t;else,z=m;end,end;p=a;end
function v=gv(s,n,d),if isfield(s,n)&&~isempty(s.(n)),v=s.(n);else,v=d;end,end

