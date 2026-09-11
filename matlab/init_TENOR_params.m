function [instrument,simulation,ensemble,dnames]=init_TENOR_params

dnames = {'normal','lognormal','schulz','exponential', ...
    'boltzmann','triangular','uniform'};

%% Setup Parameters
% 1. Instrument Parameters
ndiv = 2; % taking into account only a part of the detector for TENOR-SAXS
if 1
    % diamond
    instrument.SD_dist = 360;      % Sample-detector distance cm
    instrument.lambda = 0.1;       % wavelength nm
    instrument.det_side = 7/ndiv;  % detector side cm
    instrument.DETpix = round(1000/ndiv);
    instrument.PSF0 = bartlett2d(3,15);          % Calculated internally if empty
    % instrument.nphot=-1/1.6*10.^5;
else
    % lab saxs
    nstrument.SD_dist = 150;    %cm
    instrument.det_side = 7/ndiv; %cm
    instrument.DETpix = round(1000/ndiv);
    instrument.lambda = 0.15; %nm
    PSF0=bartlett2d(25,25);
end

% 2. Simulation Parameters
simulation.ndiv = ndiv;
% simulation.Pxn = [47 45 55 53];
simulation.Pxn = [87 85 125 123]; % TENOR PSF quarter. needs to be 4 different odd integers
simulation.signum = 4;  %number of sigmas to take in gaussian kernel 
simulation.use_r3 = 0;  %use 3rd order polynomial for m fit
simulation.use_g3 = 0;  %use 3rd order polynomial for g fit

% --- TENOR analysis window (see MG_extract.m) ---
% Round-4 update: the manuscript's final choice is q*R0_tilde < 1.0 and a
% one-kernel-width border margin; round 1-3 shipped 0.79/2.
simulation.QRG_MAX = 1.0;        % upper Guinier edge, q*Rg_apparent < QRG_MAX
simulation.DEADPIX_FACTOR = 1;   % border margin in units of max(Pxn) pixels

% 3. Ensemble Parameters
ensemble.rg = 5;
ensemble.p = 0.1;              %
ensemble.nu = -1/63;
ensemble.Scatter_R_g_weight = 6;
ensemble.d_nam = dnames{1};
ensemble.dist_param.N = 41;     % Example distribution width
end

function K = bartlett2d(n, m, mode)
%BARTLETT2D  Normalized 2D triangular (Bartlett-like) kernel with nonzero edges.
%   K = bartlett2d(n, m)          % raised-edge triangular (edges > 0)
%   K = bartlett2d(n, m, 'zero')  % classic Bartlett (edges = 0)
%
%   No toolboxes required.

    if nargin < 3, mode = 'raised'; end

    if nargin == 1, m=n; end

    wy = tri1d(n, mode);   % column
    wx = tri1d(m, mode).'; % row

    K = wy * wx;           % separable outer product
    s = sum(K(:));
    if s > 0, K = K / s; end
end

function w = tri1d(N, mode)
% 1D triangular window; 'raised' keeps nonzero endpoints.

    if N <= 0
        w = zeros(0,1); return
    elseif N == 1
        w = 1; return
    end

    idx = (0:N-1).';
    c   = (N-1)/2;                 % geometric center

    switch lower(mode)
        case 'zero'   % classic Bartlett (endpoints = 0 for N>=3)
            if N == 2
                % With only two taps, use flat weights (normalized later)
                w = [1; 1];
            else
                w = 1 - abs(idx - c) / c;
            end
        otherwise     % 'raised' (nonzero endpoints)
            d = (N+1)/2;           % larger denominator -> nonzero edges
            w = 1 - abs(idx - c) / d;
            w(w < 0) = 0;          % numerical safety for very small N
    end
end
