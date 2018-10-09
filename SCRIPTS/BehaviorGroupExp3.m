% Forming global estimates of self-performance from local confidence
% Rouault, Dayan and Fleming (2018). Experiment 3 (N=46).


% ---------------------------------------------------
%        Load group data to reproduce plots
% ---------------------------------------------------

load Exp3

T           = Exp3.T ;
T1chperser  = Exp3.T1chperser ;
T2chperser  = Exp3.T2chperser ;
T1chperserH = Exp3.T1chperserH ;
T2chperserH = Exp3.T2chperserH ;
T1chperserL = Exp3.T1chperserL ;
T2chperserL = Exp3.T2chperserL ;
acct1perser = Exp3.acct1perser ;
acct2perser = Exp3.acct2perser ;
RTt1perser  = Exp3.RTt1perser ;
RTt2perser  = Exp3.RTt2perser ;
L_H1_H2     = Exp3.L_H1_H2 ;
RT_H1_H2    = Exp3.RT_H1_H2 ;
task1val    = Exp3.task1val ;
task2val    = Exp3.task2val ;
P           = Exp3.P ;
evol_indiv_RT_LB        = Exp3.evol_indiv_RT_LB ;
task_ch_fbeasy          = Exp3.task_ch_fbeasy ;
task_ch_fbdif           = Exp3.task_ch_fbdif ;
task_ch_nofbeasy        = Exp3.task_ch_nofbeasy ;
task_ch_nofbdif         = Exp3.task_ch_nofbdif ;
confidence_level        = Exp3.confidence_level ;
confidence_level_pooled = Exp3.confidence_level_pooled ;
mratios                 = Exp3.mratios ; % Classic MLE fits
confCorrIncorr          = Exp3.confCorrIncorr ;
conf_perf_correlation   = Exp3.conf_perf_correlation ;
regLB_all               = Exp3.regLB_all ;
X_ser6_all              = Exp3.X_ser6_all ;
% Load hierarchical metacognitive efficiency fit (HMeta-d)
% Separately fitted for easy (E) and difficult (D) trials
fitHMD                  = Exp3.fitHMD ;
fitHME                  = Exp3.fitHME ;


% abscisses for graph:
x2  = 1:2 ;
x3  = 1:3;
x4  = 1:4 ;
x5  = 1:5 ;
x6  = 1:6 ;
x8  = 1:8 ;
x12 = 1:12 ;
x30 = 1:30 ;

color_fbeasy   = [120 147 60 ]/255;
color_fbdif    = [228 108 11 ]/255;
color_nofbeasy = [216 228 189]/255;
color_nofbdif  = [253 214 180]/255;

% how many participants: N=46
nS = size(T,1) ;




% ---------------------------------------------------
%        Make the statistical analyses
% ---------------------------------------------------
 
disp(['Across subj median correl betw perf and conf: ', ...
    num2str(median(conf_perf_correlation))])
disp(['Across subj mean correl betw perf and conf: ', ...
    num2str(mean(conf_perf_correlation))])

disp 'Mean confidence for easy vs. difficult trials: '
[~,p,~,stats] = ttest(confidence_level_pooled(:,1), ...
    confidence_level_pooled(:,2))

% factor 1 [FB,NFB]
% factor 2 [EASY,DIFF]

disp('ANOVA on task choice, factors Difficulty and Feedback: ')
% transformation arcsine classic for proportion data:
[~,F,~,p,~,~,~,~,~]=repanova(2*asin(sqrt(T)),[2 2]);	


% Meta-learning effects.
% Statistical tests for comparing first and second halves of experiment:

[~,p,~,stats] = ttest(L_H1_H2(:,1),L_H1_H2(:,2)) ;
[~,p,~,stats] = ttest(L_H1_H2(:,3),L_H1_H2(:,4)) ;
[~,p,~,stats] = ttest(L_H1_H2(:,5),L_H1_H2(:,6)) ;
[~,p,~,stats] = ttest(L_H1_H2(:,7),L_H1_H2(:,8)) ;

[~,p,~,stats] = ttest(RT_H1_H2(:,1),RT_H1_H2(:,2)) ;
[~,p,~,stats] = ttest(RT_H1_H2(:,3),RT_H1_H2(:,4)) ;
[~,p,~,stats] = ttest(RT_H1_H2(:,5),RT_H1_H2(:,6)) ;
[~,p,~,stats] = ttest(RT_H1_H2(:,7),RT_H1_H2(:,8)) ;

[~,p,~,stats] = ttest(evol_indiv_RT_LB(:,1),evol_indiv_RT_LB(:,2)) ;
[~,p,~,stats] = ttest(evol_indiv_RT_LB(:,3),evol_indiv_RT_LB(:,4)) ;
[~,p,~,stats] = ttest(evol_indiv_RT_LB(:,5),evol_indiv_RT_LB(:,6)) ;
[~,p,~,stats] = ttest(evol_indiv_RT_LB(:,7),evol_indiv_RT_LB(:,8)) ;




% Logistic regression examining influence of learning block duration on
% task choice, separately per pairing, in fixed effects:

pair1 = regLB_all(regLB_all(:,2)==1,1:2:3) ;
pair2 = regLB_all(regLB_all(:,2)==2,1:2:3) ;
pair3 = regLB_all(regLB_all(:,2)==3,1:2:3) ;
pair4 = regLB_all(regLB_all(:,2)==4,1:2:3) ;
pair5 = regLB_all(regLB_all(:,2)==5,1:2:3) ;
pair6 = regLB_all(regLB_all(:,2)==6,1:2:3) ;


% pool all regressors you want in the GLM function:
X = pair1(:,2)/2 ; % block duration

Y = boolean(pair1(:,1)-1); % convert vector in logical (islogical)

[B,DEV,STATS] = glmfit(X,Y,'binomial','link','logit');
% NB. All relevant regression coef (betas) are in STATS.





% Logistic regression examining if difference in confidence level between
% tasks has an influence on task choice over and above difference in
% accuracy and in RTs, in pairing 6 (all NO-FB) in FFX:

% pool all regressors you want in the GLM function:
Xacc  = X_ser6_all(:,4) ;
Xrt   = X_ser6_all(:,5) ;
Xconf = X_ser6_all(:,6) ;

% z-score regressors for meaningful comparison of beta coefficients:
Xacc_  = (Xacc-mean(Xacc))/std(Xacc) ;
Xrt_   = (Xrt-mean(Xrt))/std(Xrt) ;
Xconf_ = (Xconf-mean(Xconf))/std(Xconf) ;

% Gram-Schmidt orthogonalization (optional)
% [Q,R] = qr_gs([Xconf_ Xacc_ Xrt_]) ;
% Xconf_  = Q(:,1) ;
% Xrt_    = Q(:,3) ;
% Xacc_   = Q(:,2) ;

X = [Xacc_ Xrt_ Xconf_] ;

Y = boolean(X_ser6_all(:,7)-1); % convert vector in logical (islogical)
% NB. coded such as we predict task 2 choice.

[Bser6,DEVser6,STATSser6] = glmfit(X,Y,'binomial','link','logit');

DEVser6full = DEVser6 ;

% Plot Figure 4 from the paper:
figure;
hold on;
bar(Bser6,'FaceColor',[1 1 1],'EdgeColor',[0 153 204]/255,'LineWidth',4)
errorbar(1:4,Bser6,STATSser6.se,'k.','LineWidth',4)
title('Contribution to task choice','fontsize',30)
ylabel('regression coefficient (a.u.)','fontsize',30)
set(gca,'fontsize',30, 'XTickLabel',{'','baseline','accDiff','rtDiff','confDiff',''})
hold off



% Calculate LLH and BIC associated with this regression:

% calculate probability of choice of each action, given logit function
% over linear combination of all regressors:
p = zeros(size(X,1),1); % size(X,1) gives nb of data points
CL= zeros(size(X,1),1);

% p is the probability for choosing action 2 (Y = a_chosen)
for t=1:size(X,1)
    CL(t) = Bser6(1) + Bser6(2).*X(t,1) + Bser6(3).*X(t,2) + Bser6(4).*X(t,3);
    p(t) = 1 / (1+exp(-CL(t)));
end

% build r: proba associated with action chosen by the subject
r = zeros(size(X,1),1);
for t=1:size(X,1)
    if X_ser6_all(t,7)==1
        r(t) = 1-p(t); % proba for choosing task 2
    elseif X_ser6_all(t,7)==2
        r(t) = p(t);
    end
end

% provide log-likelihood as if this logistic fcn was a model:
LLHser6full = sum(log(max(r,1e-60))) ;

% provide BIC, nb of parameters is nb of regressors + 1 for constant term:
BICser6full = (-2)*LLHser6full + (3+1)*log(size(X,1)) ;

disp(['BIC full model: ',num2str(BICser6full)])



% Identical regression without confidence regressor:
X = [Xacc_ Xrt_] ;

Y = boolean(X_ser6_all(:,7)-1); % convert vector in logical (islogical)

[Bser6,DEVser6,STATSser6] = glmfit(X,Y,'binomial','link','logit');

DEVser6reduced = DEVser6 ;


% Calculate LLH and BIC associated with this regression:

% calculate probability of choice of each action, given logit function
% over linear combination of all regressors:
p = zeros(size(X,1),1); % size(X,1) gives nb of data points
CL= zeros(size(X,1),1);

% p is the probability for choosing action 2 (Y = a_chosen)
for t=1:size(X,1)
    CL(t) = Bser6(1) + Bser6(2).*X(t,1) + Bser6(3).*X(t,2) ;
    p(t) = 1 / (1+exp(-CL(t)));
end

% build r: proba associated with action chosen by the subject
r = zeros(size(X,1),1);
for t=1:size(X,1)
    if X_ser6_all(t,7)==1
        r(t) = 1-p(t); % proba for choosing task 2
    elseif X_ser6_all(t,7)==2
        r(t) = p(t);
    end
end

% provide log-likelihood as if this logistic fcn was a model:
LLHser6reduced = sum(log(max(r, 1e-60))) ;

% provide BIC, nb of parameters is nb of regressors + 1 for constant term:
BICser6reduced = (-2)*LLHser6reduced + (2+1)*log(size(X,1)) ;

disp(['BIC reduced model: ',num2str(BICser6reduced)])




% ---------------------------------------------------
%         Check H-Metad fits:
% ---------------------------------------------------


disp(['mean H-Mratio easy: ',num2str(exp(fitHME.mu_logMratio))])
disp(['mean H-Mratio diff: ',num2str(exp(fitHMD.mu_logMratio))])

plotSamples(exp(fitHME.mcmc.samples.mu_logMratio))
plotSamples(exp(fitHMD.mcmc.samples.mu_logMratio))

% print to check convergence:
fitHME.mcmc.Rhat
fitHMD.mcmc.Rhat




% ---------------------------------------------------
%        Make the figures
% ---------------------------------------------------


% Plot mean confidence for correct and incorrect trials.
% clean subjects with performance at ceiling in easy condition:
confCorrIncorrPlot = confCorrIncorr(all(~isnan(confCorrIncorr),2),:);

figure;
hold on;
errorbar(1:length(x2),[mean(confCorrIncorrPlot(:,3)) mean(confCorrIncorrPlot(:,1))], ...
    [std(confCorrIncorrPlot(:,3)) std(confCorrIncorrPlot(:,1))]/sqrt(nS),'Color',[0 .6 0],'LineWidth',3)
errorbar(1:length(x2),[mean(confCorrIncorrPlot(:,4)) mean(confCorrIncorrPlot(:,2))], ...
    [std(confCorrIncorrPlot(:,4)) std(confCorrIncorrPlot(:,2))]/sqrt(nS),'Color',[.6 0 0],'LineWidth',3)
ylabel('Confidence level','fontsize',28)
set(gca,'fontsize',28,'LineWidth',1.2,'XTickLabel',{'','Diff.','Easy',''})
axis([0 length(x2)+1 .6 .9])
hold off



% Global self-performance estimates across the 4 experimental conditions:
figure;
hold on;
errorbar(1:length(x5),mean(task_ch_fbeasy),std(task_ch_fbeasy)/sqrt(nS), ...
    'Color',color_fbeasy,'LineWidth',3)
errorbar(1:length(x5),mean(task_ch_fbdif),std(task_ch_fbdif)/sqrt(nS), ...
    'Color',color_fbdif,'LineWidth',3)
errorbar(1:length(x5),mean(task_ch_nofbeasy),std(task_ch_nofbeasy)/sqrt(nS), ...
    'Color',color_nofbeasy,'LineWidth',3)
errorbar(1:length(x5),mean(task_ch_nofbdif),std(task_ch_nofbdif)/sqrt(nS), ...
    'Color',color_nofbdif,'LineWidth',3)
ylabel('Task choice frequency','fontsize',28)
xlabel('Learning duration (trials)','fontsize',28)
set(gca,'fontsize',28,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 .5])
hold off



% Global self-performance estimates in each of the 6 task pairings:

T1valmean = mean(task1val,3) ;
T2valmean = mean(task2val,3) ;
T1valstd  = std(task1val,0,3)/sqrt(nS) ;
T2valstd  = std(task2val,0,3)/sqrt(nS) ;

figure;
subplot(2,3,1) % pairing 1: FB70   T1 - FB85   T2
hold on;
errorbar(1:length(x5),T1valmean(1,:),T1valstd(1,:),'Color',color_fbdif,'LineWidth',3)
errorbar(1:length(x5),T2valmean(1,:),T2valstd(1,:),'Color',color_fbeasy,'LineWidth',3)
ylabel('Task choice freq','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,2) % pairing 2: FB70   T1 - NOFB70 T2
hold on;
errorbar(1:length(x5),T1valmean(2,:),T1valstd(2,:),'Color',color_fbdif,'LineWidth',3)
errorbar(1:length(x5),T2valmean(2,:),T2valstd(2,:),'Color',color_nofbdif,'LineWidth',3)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,3) % pairing 3: FB70   T2 - NOFB85 T1
hold on;
errorbar(1:length(x5),T1valmean(3,:),T1valstd(3,:),'Color',color_nofbeasy,'LineWidth',3)
errorbar(1:length(x5),T2valmean(3,:),T2valstd(3,:),'Color',color_fbdif,'LineWidth',3)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,4) % pairing 4: FB85   T1 - NOFB70 T2
hold on;
errorbar(1:length(x5),T1valmean(4,:),T1valstd(4,:),'Color',color_fbeasy,'LineWidth',3)
errorbar(1:length(x5),T2valmean(4,:),T2valstd(4,:),'Color',color_nofbdif,'LineWidth',3)
ylabel('Task choice freq','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,5) % pairing 5: FB85   T2 - NOFB85 T1
hold on;
errorbar(1:length(x5),T1valmean(5,:),T1valstd(5,:),'Color',color_nofbeasy,'LineWidth',3)
errorbar(1:length(x5),T2valmean(5,:),T2valstd(5,:),'Color',color_fbeasy,'LineWidth',3)
xlabel('Learning duration (number of trials)','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,6) % pairing 6: NOFB70 T1 - NOFB85 T2
hold on;
errorbar(1:length(x5),T1valmean(6,:),T1valstd(6,:),'Color',color_nofbdif,'LineWidth',3)
errorbar(1:length(x5),T2valmean(6,:),T2valstd(6,:),'Color',color_nofbeasy,'LineWidth',3)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off




% Split task choice data as a function of fluctuations in performance:
figure;
subplot(2,3,1) % pairing 1: FB70   T1 - FB85   T2
hold on;
errorbar(x3,[mean(T1chperserL(:,1)) mean(T1chperser(:,1)) mean(T1chperserH(:,1))], ...
    [std(T1chperserL(:,1)) std(T1chperser(:,1)) std(T1chperserH(:,1))]/sqrt(nS),'Color',color_fbdif,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_fbdif,'MarkerFaceColor',[1 1 1],...
    'MarkerSize',11,'LineWidth',3)
errorbar(x3,[mean(T2chperserL(:,1)) mean(T2chperser(:,1)) mean(T2chperserH(:,1))], ...
    [std(T2chperserL(:,1)) std(T2chperser(:,1)) std(T2chperserH(:,1))]/sqrt(nS),'Color',color_fbeasy,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_fbeasy,'MarkerFaceColor',[1 1 1],...
    'MarkerSize',11,'LineWidth',3)
ylabel('Task choice frequency','fontsize',20)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x3)+1 0 1])
hold off

subplot(2,3,2) % pairing 2: FB70   T1 - NOFB70 T2
hold on;
errorbar(x3,[mean(T1chperserL(:,2)) mean(T1chperser(:,2)) mean(T1chperserH(:,2))], ...
    [std(T1chperserL(:,2)) std(T1chperser(:,2)) std(T1chperserH(:,2))]/sqrt(nS),'Color',color_fbdif,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_fbdif,'MarkerFaceColor',[1 1 1], ...
    'MarkerSize',11,'LineWidth',3)
errorbar(x3,[mean(T2chperserL(:,2)) mean(T2chperser(:,2)) mean(T2chperserH(:,2))], ...
    [std(T2chperserL(:,2)) std(T2chperser(:,2)) std(T2chperserH(:,2))]/sqrt(nS),'Color',color_nofbdif,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_nofbdif,'MarkerFaceColor',[1 1 1],...
    'MarkerSize',11,'LineWidth',3)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x3)+1 0 1])
hold off

subplot(2,3,3) % pairing 3: FB70   T2 - NOFB85 T1
hold on;
errorbar(x3,[mean(T1chperserL(:,3)) mean(T1chperser(:,3)) mean(T1chperserH(:,3))], ...
    [std(T1chperserL(:,3)) std(T1chperser(:,3)) std(T1chperserH(:,3))]/sqrt(nS),'Color',color_nofbeasy,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_nofbeasy,'MarkerFaceColor',[1 1 1], ...
    'MarkerSize',11,'LineWidth',3)
errorbar(x3,[mean(T2chperserL(:,3)) mean(T2chperser(:,3)) mean(T2chperserH(:,3))], ...
    [std(T2chperserL(:,3)) std(T2chperser(:,3)) std(T2chperserH(:,3))]/sqrt(nS),'Color',color_fbdif,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_fbdif,'MarkerFaceColor',[1 1 1],...
    'MarkerSize',11,'LineWidth',3)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x3)+1 0 1])
hold off

subplot(2,3,4) % pairing 4: FB85   T1 - NOFB70 T2
hold on;
errorbar(x3,[mean(T1chperserL(:,4)) mean(T1chperser(:,4)) mean(T1chperserH(:,4))], ...
    [std(T1chperserL(:,4)) std(T1chperser(:,4)) std(T1chperserH(:,4))]/sqrt(nS),'Color',color_fbeasy,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_fbeasy,'MarkerFaceColor',[1 1 1], ...
    'MarkerSize',11,'LineWidth',3)
errorbar(x3,[mean(T2chperserL(:,4)) mean(T2chperser(:,4)) mean(T2chperserH(:,4))], ...
    [std(T2chperserL(:,4)) std(T2chperser(:,4)) std(T2chperserH(:,4))]/sqrt(nS),'Color',color_nofbdif,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_nofbdif,'MarkerFaceColor',[1 1 1],...
    'MarkerSize',11,'LineWidth',3)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x3)+1 0 1])
hold off

subplot(2,3,5) % pairing 5: FB85   T2 - NOFB85 T1
hold on;
errorbar(x3,[mean(T1chperserL(:,5)) mean(T1chperser(:,5)) mean(T1chperserH(:,5))], ...
    [std(T1chperserL(:,5)) std(T1chperser(:,5)) std(T1chperserH(:,5))]/sqrt(nS),'Color',color_nofbeasy,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_nofbeasy,'MarkerFaceColor',[1 1 1], ...
    'MarkerSize',11,'LineWidth',3)
errorbar(x3,[mean(T2chperserL(:,5)) mean(T2chperser(:,5)) mean(T2chperserH(:,5))], ...
    [std(T2chperserL(:,5)) std(T2chperser(:,5)) std(T2chperserH(:,5))]/sqrt(nS),'Color',color_fbeasy,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_fbeasy,'MarkerFaceColor',[1 1 1],...
    'MarkerSize',11,'LineWidth',3)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x3)+1 0 1])
hold off

subplot(2,3,6) % pairing 6: NOFB70 T1 - NOFB85 T2
hold on;
errorbar(x3,[mean(T1chperserL(:,6)) mean(T1chperser(:,6)) mean(T1chperserH(:,6))], ...
    [std(T1chperserL(:,6)) std(T1chperser(:,6)) std(T1chperserH(:,6))]/sqrt(nS),'Color',color_nofbdif,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_nofbdif,'MarkerFaceColor',[1 1 1], ...
    'MarkerSize',11,'LineWidth',3)
errorbar(x3,[mean(T2chperserL(:,6)) mean(T2chperser(:,6)) mean(T2chperserH(:,6))], ...
    [std(T2chperserL(:,6)) std(T2chperser(:,6)) std(T2chperserH(:,6))]/sqrt(nS),'Color',color_nofbeasy,...
    'LineStyle','-','Marker','o','MarkerEdgeColor',color_nofbeasy,'MarkerFaceColor',[1 1 1],...
    'MarkerSize',11,'LineWidth',3)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x3)+1 0 1])
hold off




% Objective performance against subjective self-performance estimates:
figure;

subplot(1,2,1)
hold on;
bar(x4,[mean(P(:,1)) 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 mean(P(:,2)) 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x4,[0 0 mean(P(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 0 0 mean(P(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x4),mean(P),std(P)/sqrt(nS),'k.','LineWidth',2)
ylabel('Mean performance in learning blocks','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x4)+1 .5 1])
hold off

subplot(1,2,2)
hold on;
bar(x4,[mean(T(:,1)) 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 mean(T(:,2)) 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x4,[0 0 mean(T(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 0 0 mean(T(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x4),mean(T),std(T)/sqrt(nS),'k.','LineWidth',2)
ylabel('Task choice frequency','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
legend('FB easy','FB diff','CONF easy','CONF diff')
axis([0 length(x4)+1 0 .5])
hold off





% Focus on trials without feedback, in blocks with only confidence ratings
% 4 panels: Performance, Global self-performance estimates i.e. Task
% choice, Mean local confidence, Metacognitive efficiency.

figure;
subplot(2,2,1)
hold on;
bar(x2,[mean(acct1perser(:,6)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,6))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,6)) 0],[std(acct1perser(:,6))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,6))],[0 std(acct2perser(:,6))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
ylabel('Objective performance','fontsize',20)
axis([0 length(x2)+1 0 1])
hold off

subplot(2,2,2)
hold on;
bar(x2,[mean(T1chperser(:,6)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(T2chperser(:,6))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(T1chperser(:,6)) 0],[std(T1chperser(:,6))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(T2chperser(:,6))],[0 std(T2chperser(:,6))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
ylabel('Task choice','fontsize',20)
axis([0 length(x2)+1 0 1])
hold off

subplot(2,2,3)
hold on;
bar(x2,[0 mean(confidence_level_pooled(:,1))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[mean(confidence_level_pooled(:,2)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[0 mean(confidence_level_pooled(:,1))],[0 std(confidence_level_pooled(:,1))/sqrt(nS)],'k.','LineWidth',2)
errorbar(1:length(x2),[mean(confidence_level_pooled(:,2)) 0],[std(confidence_level_pooled(:,2))/sqrt(nS) 0],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
ylabel('Confidence level','fontsize',20)
axis([0 length(x2)+1 0 1])
hold off

subplot(2,2,4)
hold on;
bar(x2,[0 mean(mratios(:,1))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[mean(mratios(:,2)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[0 mean(mratios(:,1))],[0 std(mratios(:,1))/sqrt(nS)],'k.','LineWidth',2)
errorbar(1:length(x2),[mean(mratios(:,2)) 0],[std(mratios(:,2))/sqrt(nS) 0],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
ylabel('M-ratio','fontsize',20)
axis([0 length(x2)+1 0 1.5])
hold off




% Mean local confidence in easy vs difficult conditions:
figure;
hold on;
bar(x2,[0 mean(confidence_level_pooled(:,1))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[mean(confidence_level_pooled(:,2)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[0 mean(confidence_level_pooled(:,1))], ...
    [0 std(confidence_level_pooled(:,1))/sqrt(nS)],'k.','LineWidth',2)
errorbar(1:length(x2),[mean(confidence_level_pooled(:,2)) 0], ...
    [std(confidence_level_pooled(:,2))/sqrt(nS) 0],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
ylabel('Confidence level','fontsize',30)
axis([0 length(x2)+1 .5 1])
hold off




% Accuracy in each of the six task pairings:
figure;

subplot(2,3,1) % pairing 1: FB70   T1 - FB85   T2
hold on;
bar(x2,[mean(acct1perser(:,1)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,1))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,1)) 0],[std(acct1perser(:,1))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,1))],[0 std(acct2perser(:,1))/sqrt(nS)],'k.','LineWidth',2)
ylabel('Accuracy per pairing','fontsize',15)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,2) % pairing 2: FB70   T1 - NOFB70 T2
hold on;
bar(x2,[mean(acct1perser(:,2)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,2))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,2)) 0],[std(acct1perser(:,2))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,2))],[0 std(acct2perser(:,2))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,3) % pairing 3: FB70   T2 - NOFB85 T1
hold on;
bar(x2,[mean(acct1perser(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,3))],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,3)) 0],[std(acct1perser(:,3))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,3))],[0 std(acct2perser(:,3))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,4) % pairing 4: FB85   T1 - NOFB70 T2
hold on;
bar(x2,[mean(acct1perser(:,4)) 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,4)) 0],[std(acct1perser(:,4))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,4))],[0 std(acct2perser(:,4))/sqrt(nS)],'k.','LineWidth',2)
ylabel('Accuracy per pairing','fontsize',15)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,5) % pairing 5: FB85   T2 - NOFB85 T1
hold on;
bar(x2,[mean(acct1perser(:,5)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,5))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,5)) 0],[std(acct1perser(:,5))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,5))],[0 std(acct2perser(:,5))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,6) % pairing 6: NOFB70 T1 - NOFB85 T2
hold on;
bar(x2,[mean(acct1perser(:,6)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,6))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,6)) 0],[std(acct1perser(:,6))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,6))],[0 std(acct2perser(:,6))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off




% RTs in each of the six task pairings:
figure;

subplot(2,3,1) % pairing 1: FB70   T1 - FB85   T2
hold on;
bar(x2,[mean(RTt1perser(:,1)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,1))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,1)) 0],[std(RTt1perser(:,1))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,1))],[0 std(RTt2perser(:,1))/sqrt(nS)],'k.','LineWidth',2)
ylabel('RTs per pairing','fontsize',15)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 900])
hold off

subplot(2,3,2) % pairing 2: FB70   T1 - NOFB70 T2
hold on;
bar(x2,[mean(RTt1perser(:,2)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,2))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,2)) 0],[std(RTt1perser(:,2))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,2))],[0 std(RTt2perser(:,2))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 900])
hold off

subplot(2,3,3) % pairing 3: FB70   T2 - NOFB85 T1
hold on;
bar(x2,[mean(RTt1perser(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,3))],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,3)) 0],[std(RTt1perser(:,3))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,3))],[0 std(RTt2perser(:,3))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 900])
hold off

subplot(2,3,4) % pairing 4: FB85   T1 - NOFB70 T2
hold on;
bar(x2,[mean(RTt1perser(:,4)) 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,4)) 0],[std(RTt1perser(:,4))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,4))],[0 std(RTt2perser(:,4))/sqrt(nS)],'k.','LineWidth',2)
ylabel('RTs per pairing','fontsize',15)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 900])
hold off

subplot(2,3,5) % pairing 5: FB85   T2 - NOFB85 T1
hold on;
bar(x2,[mean(RTt1perser(:,5)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,5))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,5)) 0],[std(RTt1perser(:,5))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,5))],[0 std(RTt2perser(:,5))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 900])
hold off

subplot(2,3,6) % pairing 6: NOFB70 T1 - NOFB85 T2
hold on;
bar(x2,[mean(RTt1perser(:,6)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,6))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,6)) 0],[std(RTt1perser(:,6))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,6))],[0 std(RTt2perser(:,6))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',10,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 900])
hold off




% Meta-learning effects over the course of the experiment:
% Comparison of performance and RTs in first vs. second halves of
% experiment shows no differential evolution in each of the 4 conditions:
figure;

subplot(3,1,1)
hold on;
bar(x8,[mean(L_H1_H2(:,1))  0 0 0 0 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x8,[0 mean(L_H1_H2(:,2))  0 0 0 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x8,[0 0 mean(L_H1_H2(:,3))  0 0 0 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 mean(L_H1_H2(:,4))  0 0 0 0],'FaceColor',color_fbdif ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 mean(L_H1_H2(:,5))  0 0 0],'FaceColor',color_nofbeasy ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 mean(L_H1_H2(:,6))  0 0],'FaceColor',color_nofbeasy ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 0 mean(L_H1_H2(:,7))  0],'FaceColor',color_nofbdif ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 0 0 mean(L_H1_H2(:,8)) ],'FaceColor',color_nofbdif ,'EdgeColor',[1 1 1])
errorbar(x8,mean(L_H1_H2),std(L_H1_H2)/sqrt(nS),'k.','LineWidth',2)
ylabel('Perf','fontsize',25)
    set(gca,'FontSize',15,'LineWidth',1.2,'XTickLabel',{'' 'fbE_H1'...
        'fbE_H2' 'fbD_H1' 'fbD_H2' 'nofbE_H1' 'nofbE_H2' 'nofbD_H1' 'nofbD_H2' ''})
axis([0 length(x8)+1 0 1])
hold off

subplot(3,1,2)
hold on;
bar(x8,[mean(RT_H1_H2(:,1))  0 0 0 0 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x8,[0 mean(RT_H1_H2(:,2))  0 0 0 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x8,[0 0 mean(RT_H1_H2(:,3))  0 0 0 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 mean(RT_H1_H2(:,4))  0 0 0 0],'FaceColor',color_fbdif ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 mean(RT_H1_H2(:,5))  0 0 0],'FaceColor',color_nofbeasy ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 mean(RT_H1_H2(:,6))  0 0],'FaceColor',color_nofbeasy ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 0 mean(RT_H1_H2(:,7))  0],'FaceColor',color_nofbdif ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 0 0 mean(RT_H1_H2(:,8)) ],'FaceColor',color_nofbdif ,'EdgeColor',[1 1 1])
errorbar(x8,mean(RT_H1_H2),std(RT_H1_H2)/sqrt(nS),'k.','LineWidth',2)
ylabel('RTs (raw)','fontsize',25)
    set(gca,'FontSize',15,'LineWidth',1.2,'XTickLabel',{'' 'fbE_H1'...
        'fbE_H2' 'fbD_H1' 'fbD_H2'...
        'nofbE_H1' 'nofbE_H2' 'nofbD_H1' 'nofbD_H2' ''})
axis([0 length(x8)+1 0 1000])
hold off

subplot(3,1,3)
hold on;
bar(x8,[nanmean(evol_indiv_RT_LB(:,1))  0 0 0 0 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x8,[0 nanmean(evol_indiv_RT_LB(:,2))  0 0 0 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x8,[0 0 nanmean(evol_indiv_RT_LB(:,3))  0 0 0 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 nanmean(evol_indiv_RT_LB(:,4))  0 0 0 0],'FaceColor',color_fbdif ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 nanmean(evol_indiv_RT_LB(:,5))  0 0 0],'FaceColor',color_nofbeasy ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 nanmean(evol_indiv_RT_LB(:,6))  0 0],'FaceColor',color_nofbeasy ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 0 nanmean(evol_indiv_RT_LB(:,7))  0],'FaceColor',color_nofbdif ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 0 0 nanmean(evol_indiv_RT_LB(:,8)) ],'FaceColor',color_nofbdif ,'EdgeColor',[1 1 1])
errorbar(x8,nanmean(evol_indiv_RT_LB),nanstd(evol_indiv_RT_LB)/sqrt(nS),'k.','LineWidth',2)
ylabel('RTs (log)','fontsize',25)
    set(gca,'FontSize',15,'LineWidth',1.2,'XTickLabel',{'' 'fbE_H1'...
        'fbE_H2' 'fbD_H1' 'fbD_H2'...
        'nofbE_H1' 'nofbE_H2' 'nofbD_H1' 'nofbD_H2' ''})
axis([0 length(x8)+1 -.7 .7])
hold off




% Individual shifts in evolution betw first and second halves for each
% participant separately:
figure;
subplot(2,2,1)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,1),L_H1_H2(i,2)],'Color',color_fbeasy,'LineWidth',3)
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
ylabel('Performance','fontsize',30)
axis([0 3 .4 1])
hold off

subplot(2,2,2)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,3),L_H1_H2(i,4)],'Color',color_fbdif,'LineWidth',3)
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
axis([0 3 .4 1])
hold off

subplot(2,2,3)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,5),L_H1_H2(i,6)],'Color',color_nofbeasy,'LineWidth',3)
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
ylabel('Performance','fontsize',30)
axis([0 3 .4 1])
hold off

subplot(2,2,4)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,7),L_H1_H2(i,8)],'Color',color_nofbdif,'LineWidth',3)
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
axis([0 3 .4 1])
hold off




% Relationship between metacognitive efficiency and global self-performance
% estimates: between-subjects scatterplot

% Difference in self-performance estimates between NFB_Eeasy - NFB_Diff.:
delta_task_ch_pairing6 = T2chperser(:,6)-T1chperser(:,6) ;

% Average metacognitive efficiency between fit in easy and diff. trials:
MratioMoy = (mratios(:,1)+mratios(:,2))/2 ; % classic fit

% Additionally removing one subject with negative Metacog efficiency:
[rho,pval] = corrcoef(delta_task_ch_pairing6(MratioMoy>0), ...
    (MratioMoy(MratioMoy>0)));

disp(['Correlation between metacognitive efficiency MLE and global self-performance estimates: ', ...
    num2str(rho(2)),', significance: ',num2str(pval(2))])

figure;
plot(delta_task_ch_pairing6,MratioMoy,'Color',[.8 .8 .2],...
    'LineStyle','none','Marker','o','MarkerEdgeColor',[.8 .8 .2],...
    'MarkerFaceColor',[.8 .8 .2],'Markersize',15);lsline;
set(gca,'fontsize',25,'LineWidth',2)
ylabel('Metacognitive efficiency','fontsize',25)
xlabel('Task choice Easy vs. Difficult','fontsize',25)



% Average metacognitive efficiency between fit in easy and diff. trials:
MratioMoyHM = (fitHME.Mratio+fitHMD.Mratio)/2; % hierarchical fit

[rho,pval] = corrcoef(delta_task_ch_pairing6(MratioMoyHM>0), ...
    (MratioMoyHM(MratioMoyHM>0)));

disp(['Correlation between metacognitive efficiency HMeta-d and global self-performance estimates: ', ...
    num2str(rho(2)),', significance: ',num2str(pval(2))])



% NB. helper function regression_line_ci.m works only under Mac, but
% the regression line without c.i. can still be plotted:

STATSreg = regstats(MratioMoyHM,delta_task_ch_pairing6,'linear','beta');

figure;
ylabel('Metacognitive efficiency','fontsize',30)
xlabel('Task choice Easy minus Difficult','fontsize',30)
regression_line_ci(.05,STATSreg.beta,delta_task_ch_pairing6,MratioMoyHM)
set(gca,'fontsize',30,'LineWidth',2)







