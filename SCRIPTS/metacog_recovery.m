% Demonstration of hierarchical model fits
%
% SF 2014
% Adapted Marion Rouault November 2018
% Perform simulations to do model recovery and compare MLE vs. HMeta-d

clear all

Ntrials = 90; % Exp3 had 180 trials of no-feedback-conf, split into E and D
Nsub = 46; % Exp3 had 46 subjects

c = 0;
c1 = [-1.5 -1 -0.5];
c2 = [0.5 1 1.5];

% In Exp3, we have a mean across easy and diff conditions: d'=1.55
group_d = 1.55;

group_mratio = 0.8;
sigma = 0.5;

mcmc_params = fit_meta_d_params;
mcmc_params.estimate_dprime = 0;

Mratio_MLE = zeros(2,Nsub) ;

for i = 1:Nsub
    
    % Generate dprime
    d(i) = normrnd(group_d, sigma);
    metad = group_mratio.*d(i);
    
    Mratio_MLE(1,i) = metad ;
    
    % Generate data
    sim = metad_sim(d(i), metad, c, c1, c2, Ntrials);
    
    
    % MLE individual fit: deal with empty cells:
    adj_f = 1/length(sim.nR_S1);
    nR_S1_MLE = sim.nR_S1 + adj_f;
    nR_S2_MLE = sim.nR_S2 + adj_f;
    
    % single-subject fit meta-d' using Maniscalco and Lau code,
    % using maximum likelihood estimation:
    fitMLE = fit_meta_d_MLE(nR_S1_MLE, nR_S2_MLE) ;
    Mratio_MLE(2,i) = fitMLE.meta_da ;
    
    % Store data for hierarchical fit:
    nR_S1{i} = sim.nR_S1;
    nR_S2{i} = sim.nR_S2;
    
end

% Fit group data all at once
fit = fit_meta_d_mcmc_group(nR_S1, nR_S2, mcmc_params);

% Call plotSamples to plot posterior of group Mratio
plotSamples(exp(fit.mcmc.samples.mu_logMratio))
hdi = calc_HDI(exp(fit.mcmc.samples.mu_logMratio(:)));
fprintf(['\n HDI on meta-d/d: ', num2str(hdi) '\n\n'])


% Plot simulated meta-d against recovered meta-d values
figure;
subplot(1,2,1)
hold on;
for i=1:Nsub
   plot(Mratio_MLE(1,i),fit.meta_d(i),'bs','LineWidth',3) 
end
plot(.1:.1:2.4,.1:.1:2.4,'k','LineWidth',2)
xlabel('Simulated', 'fontsize', 25)
ylabel('Recovered', 'fontsize', 25)
title('Hierarchical', 'fontsize', 25)
set(gca, 'fontsize', 25)
axis([0 2.5 0 2.5])
hold off

subplot(1,2,2)
hold on;
for i=1:Nsub
   plot(Mratio_MLE(1,i),Mratio_MLE(2,i),'bs','LineWidth',3) 
end
plot(.1:.1:2.4,.1:.1:2.4,'k','LineWidth',2)
title('MLE', 'fontsize', 25)
xlabel('Simulated', 'fontsize', 25)
ylabel('Recovered', 'fontsize', 25)
set(gca, 'fontsize', 25)
axis([0 2.5 0 2.5])
hold off


