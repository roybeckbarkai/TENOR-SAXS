function run_reference_export(out_dir)
%RUN_REFERENCE_EXPORT Generate MATLAB/Octave-side reference data for the
% Python port's parity tests.
%
% Produces two .mat files in OUT_DIR (default: ../../tenor-saxs-v2-data/octave-reference
% relative to this file, i.e. the sibling data root, kept outside the git repo):
%
%   deterministic_grid.mat  - noise-free (Nois=0) cases over a grid of
%                              (V, phi'', distribution), fixed R0=5nm, Diamond-like
%                              instrument params. For each case: qx,qy,I, the
%                              MG_extract fit (p,covP,actualPSF,RG2) and the full
%                              TENOR_protocol Results struct (observables, V
%                              estimates per branch, BestV, diagnostics).
%   noisy_grid.mat           - one fixed ensemble (Gaussian chain, R0=3nm, V=0.1,
%                              normal distribution), swept over several peak-photon
%                              levels with several seeded replicates each, to
%                              support STATISTICAL (not bit-exact) comparison
%                              against the Python noise pipeline.
%
% Usage:
%   octave --no-gui --eval "run_reference_export"
%   octave --no-gui --eval "run_reference_export('/custom/out/dir')"

if nargin < 1 || isempty(out_dir)
    here = fileparts(mfilename('fullpath'));
    out_dir = fullfile(here, '..', '..', 'tenor-saxs-v2-data', 'octave-reference');
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
printf('Writing reference exports to: %s\n', out_dir);

[inst, sim, ens, dnames] = init_TENOR_params();

%% ------------------------------------------------------------------
%% 1. Deterministic (noise-free) grid
%% ------------------------------------------------------------------
V_grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
% phi'' values: canonical shell/sphere/rod/chain plus two intermediate values,
% matching the paper's Fig. 2 set (Table 1) plus its stated phi'' scan range.
phi2_grid = struct( ...
    'name', {'shell', 'sphere', 'intermediate', 'chain', 'lowcurv', 'highcurv'}, ...
    'value', {-1/45, -1/63, 0.01, 1/18, -0.02, 0.06});
dist_list = {'normal', 'lognormal', 'uniform'};
R0 = 5;

cases = struct('CaseID', {}, 'V', {}, 'Phi2', {}, 'Phi2Name', {}, 'Distribution', {}, ...
    'R0', {}, 'qx', {}, 'qy', {}, 'I', {}, 'RG2', {}, 'actualPSF', {}, ...
    'p', {}, 'covP', {}, 'Yg100', {}, 'Yg210', {}, 'Ym210', {}, ...
    'Jg10', {}, 'Jg21', {}, 'Jm', {}, 'BestV', {}, 'BestV_SE', {}, 'Rg', {}, ...
    'status', {}, 'error', {});

caseId = 0;
nTotal = numel(V_grid) * numel(phi2_grid) * numel(dist_list);
for iv = 1:numel(V_grid)
    for ip = 1:numel(phi2_grid)
        for id = 1:numel(dist_list)
            caseId = caseId + 1;
            V = V_grid(iv);
            phi2 = phi2_grid(ip).value;
            distName = dist_list{id};
            printf('[%d/%d] V=%.3f phi2=%s(%.5f) dist=%s\n', caseId, nTotal, V, ...
                phi2_grid(ip).name, phi2, distName);

            c = struct();
            c.CaseID = caseId; c.V = V; c.Phi2 = phi2; c.Phi2Name = phi2_grid(ip).name;
            c.Distribution = distName; c.R0 = R0;
            c.qx = []; c.qy = []; c.I = []; c.RG2 = NaN; c.actualPSF = [];
            c.p = []; c.covP = []; c.Yg100 = NaN; c.Yg210 = NaN; c.Ym210 = NaN;
            c.Jg10 = NaN; c.Jg21 = NaN; c.Jm = NaN; c.BestV = NaN; c.BestV_SE = NaN;
            c.Rg = NaN; c.status = ''; c.error = '';

            try
                distParam = ens.dist_param;
                [qx, qy, I, distrp] = Scatter2D(R0, 0, V, phi2, inst.DETpix, ...
                    inst.SD_dist, inst.lambda, inst.det_side, inst.PSF0, ...
                    distName, distParam, ens.Scatter_R_g_weight);
                c.qx = single(qx); c.qy = single(qy); c.I = single(I);

                [~, RG2, ~, res, actualPSF] = MG_extract(sim.Pxn, qx, qy, I, ...
                    sim.signum, [], sim.use_r3, sim.use_g3, inst.lambda, ...
                    sim.QRG_MAX, sim.DEADPIX_FACTOR);
                c.RG2 = RG2; c.actualPSF = actualPSF; c.p = res.p; c.covP = res.covP;

                simTP = sim; simTP.RG2 = [];
                ensTP = ens; ensTP.nu = phi2;
                Results = TENOR_protocol(I, qx, qy, inst, simTP, ensTP);
                c.Yg100 = Results.Yg100; c.Yg210 = Results.Yg210; c.Ym210 = Results.Ym210;
                c.Jg10 = Results.Jg10; c.Jg21 = Results.Jg21; c.Jm = Results.Jm;
                c.BestV = Results.BestV; c.BestV_SE = Results.BestV_SE; c.Rg = Results.Rg;
                c.status = 'ok';
            catch err
                c.status = 'failed';
                c.error = err.message;
                printf('  FAILED: %s\n', err.message);
            end
            cases(caseId) = c;
        end
    end
end

save('-v7', fullfile(out_dir, 'deterministic_grid.mat'), 'cases', 'V_grid', 'phi2_grid', 'dist_list', 'R0', 'inst', 'sim', 'ens');
printf('Wrote deterministic_grid.mat (%d cases)\n', numel(cases));

%% ------------------------------------------------------------------
%% 2. Noisy grid (statistical comparison only)
%% ------------------------------------------------------------------
noise_R0 = 3;
noise_V = 0.10;
noise_phi2 = 1/18;  % Gaussian chain, matches paper's Fig. 6 setup
noise_dist = 'normal';
peakPhotons = 10.^(2.5:0.5:5) / 1.65;
nReplicates = 20;
masterSeed = 314159;

[qx, qy, I0, ~] = Scatter2D(noise_R0, 0, noise_V, noise_phi2, inst.DETpix, ...
    inst.SD_dist, inst.lambda, inst.det_side, inst.PSF0, noise_dist, ens.dist_param, ...
    ens.Scatter_R_g_weight);

noisyResults = struct('NoiseLevelIndex', {}, 'PeakPhotons', {}, 'Replicate', {}, ...
    'Seed', {}, 'BestV', {}, 'BestV_SE', {}, 'Rg', {}, 'status', {}, 'error', {});
idx = 0;
for j = 1:numel(peakPhotons)
    ph = peakPhotons(j);
    for r = 1:nReplicates
        idx = idx + 1;
        sd = mod(masterSeed + 104729*1 + 1009*j + 37*r, 2^32-1);
        if sd == 0, sd = 1; end
        rng(sd, 'twister');
        I = I0 + sqrt(max(I0,0)/ph) .* randn(size(I0));
        I(isfinite(I) & I < 0) = 0;

        rr = struct();
        rr.NoiseLevelIndex = j; rr.PeakPhotons = ph; rr.Replicate = r; rr.Seed = sd;
        rr.BestV = NaN; rr.BestV_SE = NaN; rr.Rg = NaN; rr.status = ''; rr.error = '';
        try
            simTP = sim; simTP.RG2 = [];
            ensTP = ens; ensTP.nu = noise_phi2;
            Results = TENOR_protocol(I, qx, qy, inst, simTP, ensTP);
            rr.BestV = Results.BestV; rr.BestV_SE = Results.BestV_SE; rr.Rg = Results.Rg;
            rr.status = 'ok';
        catch err
            rr.status = 'failed';
            rr.error = err.message;
        end
        noisyResults(idx) = rr;
    end
    printf('Noise level %d/%d (peakPhotons=%.1f) done\n', j, numel(peakPhotons), ph);
end

save('-v7', fullfile(out_dir, 'noisy_grid.mat'), 'noisyResults', 'peakPhotons', ...
    'nReplicates', 'masterSeed', 'noise_R0', 'noise_V', 'noise_phi2', 'noise_dist', ...
    'qx', 'qy', 'I0');
printf('Wrote noisy_grid.mat (%d rows)\n', numel(noisyResults));

end
