% Forming global estimates of self-performance from local confidence
% Rouault, Dayan and Fleming (2018). Experiment 2 (N=29).

% FACTORS OF INTEREST:
% - task choice proportion in each of the 4 conditions
% - analysis of learning blocks: percent correct as a function of difficulty
% - analysis of learning blocks: percent correct in each of the 4 conditions



% ---------------------------------------------------
%        Load group data to reproduce plots
% ---------------------------------------------------

load Exp2

T           = Exp2.T ;
T1chperser  = Exp2.T1chperser ;
T2chperser  = Exp2.T2chperser ;
T1chperserH = Exp2.T1chperserH ;
T2chperserH = Exp2.T2chperserH ;
T1chperserL = Exp2.T1chperserL ;
T2chperserL = Exp2.T2chperserL ;
acct1perser = Exp2.acct1perser ;
acct2perser = Exp2.acct2perser ;
task_ch_fbeasy   = Exp2.task_ch_fbeasy ;
task_ch_fbdif    = Exp2.task_ch_fbdif ;
task_ch_nofbeasy = Exp2.task_ch_nofbeasy ;
task_ch_nofbdif  = Exp2.task_ch_nofbdif ;
task1val    = Exp2.task1val ;
task2val    = Exp2.task2val ;
P           = Exp2.P ;
perf_rt_LB  = Exp2.perf_rt_LB ;
perf_rt_LB_rel = Exp2.perf_rt_LB_rel ;
L_H1_H2     = Exp2.L_H1_H2 ;
RT_H1_H2    = Exp2.RT_H1_H2 ;
evol_indiv_RT_LB = Exp2.evol_indiv_RT_LB ;
MetaL       = Exp2.MetaL ;
regLB_all   = Exp2.regLB_all ;
RTt1perser  = Exp2.RTt1perser ;
RTt2perser  = Exp2.RTt2perser ;


% number of subjects
nS = size(T,1);

% abscissa for graph:
x2  = 1:2 ;
x3  = 1:3 ;
x4  = 1:4 ;
x5  = 1:5 ;
x6  = 1:6 ;
x8  = 1:8 ;
x30 = 1:30 ;

% colors for graph:
color_fbeasy   = [120 147 60 ]/255;
color_fbdif    = [228 108 11 ]/255;
color_nofbeasy = [216 228 189]/255;
color_nofbdif  = [253 214 180]/255;



% ---------------------------------------------------
%        Make the statistical analyses
% ---------------------------------------------------

% ANOVA on task choice frequencies according to 2 by 2 design:
% factor A [FB,NFB]
% factor B [EASY,DIFF]

disp('ANOVA on task choice, factors Feedback and Difficulty: ')
% classic arcsine transformation for proportion data:
[~,F,~,p,~,~,~,~,~] = repanova(2*asin(sqrt(T)),[2 2]);



% Regression examining influence of learning block duration on
% task choice, separately per pairing, in FFX due to limited number of
% task choice (30 data points) per participant:

pair1 = regLB_all(regLB_all(:,2)==1,1:2:3) ;
pair2 = regLB_all(regLB_all(:,2)==2,1:2:3) ;
pair3 = regLB_all(regLB_all(:,2)==3,1:2:3) ;
pair4 = regLB_all(regLB_all(:,2)==4,1:2:3) ;
pair5 = regLB_all(regLB_all(:,2)==5,1:2:3) ;
pair6 = regLB_all(regLB_all(:,2)==6,1:2:3) ;


% Pool all regressors you want in the GLM function:
X = pair6(:,2)/2 ; % Block duration

Y = boolean(pair6(:,1)-1) ; % convert vector in logical

[B, DEV, STATS] = glmfit(X,Y,'binomial','link','logit') ;




% Statistical tests regarding meta-learning effects, testing performance
% and RTs between first and second halves of experiment:

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



% ---------------------------------------------------
%        Make the figures
% ---------------------------------------------------


% Self-performance estimates (Task choice) as a function of block duration
figure(1)

hold on;
errorbar(1:length(x5),mean(task_ch_fbeasy),std(task_ch_fbeasy)/sqrt(nS),'Color',color_fbeasy,'LineWidth',3)
errorbar(1:length(x5),mean(task_ch_fbdif),std(task_ch_fbdif)/sqrt(nS),'Color',color_fbdif,'LineWidth',3)
errorbar(1:length(x5),mean(task_ch_nofbeasy),std(task_ch_nofbeasy)/sqrt(nS),'Color',color_nofbeasy,'LineWidth',3)
errorbar(1:length(x5),mean(task_ch_nofbdif),std(task_ch_nofbdif)/sqrt(nS),'Color',color_nofbdif,'LineWidth',3)
ylabel('Task choice frequency','fontsize',28)
xlabel('Learning duration (trials)','fontsize',28)
set(gca,'fontsize',28,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 .5])
hold off



% Self-performance estimates (Task choice) as a function of block duration
% Plotted separately in the six pairings
% Marginalise for plotting:
T1valmean = mean(task1val,3) ;
T2valmean = mean(task2val,3) ;
T1valstd  = std(task1val,0,3)/sqrt(nS) ;
T2valstd  = std(task2val,0,3)/sqrt(nS) ;

figure(2)

subplot(2,3,1) % pairing 1: FB70   T1 - FB85   T2
hold on;
errorbar(1:length(x5),T1valmean(1,:),T1valstd(1,:),'Color',color_fbdif,'LineWidth',4)
errorbar(1:length(x5),T2valmean(1,:),T2valstd(1,:),'Color',color_fbeasy,'LineWidth',4)
ylabel('Task choice frequency','fontsize',27)
set(gca,'fontsize',27,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,2) % pairing 2: FB70   T1 - NOFB70 T2
hold on;
errorbar(1:length(x5),T1valmean(2,:),T1valstd(2,:),'Color',color_fbdif,'LineWidth',4)
errorbar(1:length(x5),T2valmean(2,:),T2valstd(2,:),'Color',color_nofbdif,'LineWidth',4)
set(gca,'fontsize',27,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,3) % pairing 3: FB70   T2 - NOFB85 T1
hold on;
errorbar(1:length(x5),T1valmean(3,:),T1valstd(3,:),'Color',color_nofbeasy,'LineWidth',4)
errorbar(1:length(x5),T2valmean(3,:),T2valstd(3,:),'Color',color_fbdif,'LineWidth',4)
set(gca,'fontsize',27,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,4) % pairing 4: FB85   T1 - NOFB70 T2
hold on;
errorbar(1:length(x5),T1valmean(4,:),T1valstd(4,:),'Color',color_fbeasy,'LineWidth',4)
errorbar(1:length(x5),T2valmean(4,:),T2valstd(4,:),'Color',color_nofbdif,'LineWidth',4)
set(gca,'fontsize',27,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,5) % pairing 5: FB85   T2 - NOFB85 T1
hold on;
errorbar(1:length(x5),T1valmean(5,:),T1valstd(5,:),'Color',color_nofbeasy,'LineWidth',4)
errorbar(1:length(x5),T2valmean(5,:),T2valstd(5,:),'Color',color_fbeasy,'LineWidth',4)
xlabel('Learning duration (trials per task)','fontsize',25)
set(gca,'fontsize',27,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off

subplot(2,3,6) % pairing 6: NOFB70 T1 - NOFB85 T2
hold on;
errorbar(1:length(x5),T1valmean(6,:),T1valstd(6,:),'Color',color_nofbdif,'LineWidth',4)
errorbar(1:length(x5),T2valmean(6,:),T2valstd(6,:),'Color',color_nofbeasy,'LineWidth',4)
set(gca,'fontsize',27,'LineWidth',1.2,'XTickLabel',{'','4','8','12','16','20',''})
axis([0 length(x5)+1 0 1])
hold off



% Objective performance and subjective self-performance estimates:
figure(3)

subplot(1,2,1)
hold on;
bar(x4,[mean(P(:,1)) 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 mean(P(:,2)) 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x4,[0 0 mean(P(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 0 0 mean(P(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x4),mean(P),std(P)/sqrt(nS),'k.','LineWidth',2)
ylabel('Mean performance','fontsize',30)
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
legend('FB easy','FB diff','NO FB easy','NO FB diff')
axis([0 length(x4)+1 0 .5])
hold off



% Task choice plotted in each of the six possible task pairings:
figure(4)

subplot(2,3,1) % pairing 1: FB70   T1 - FB85   T2
hold on;
bar(x2,[mean(T1chperser(:,1)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(T2chperser(:,1))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(T1chperser(:,1)) 0],[std(T1chperser(:,1))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(T2chperser(:,1))],[0 std(T2chperser(:,1))/sqrt(nS)],'k.','LineWidth',2)
ylabel('Task choice frequency','fontsize',15)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,2) % pairing 2: FB70   T1 - NOFB70 T2
hold on;
bar(x2,[mean(T1chperser(:,2)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(T2chperser(:,2))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(T1chperser(:,2)) 0],[std(T1chperser(:,2))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(T2chperser(:,2))],[0 std(T2chperser(:,2))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,3) % pairing 3: FB70   T2 - NOFB85 T1
hold on;
bar(x2,[mean(T1chperser(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(T2chperser(:,3))],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(T1chperser(:,3)) 0],[std(T1chperser(:,3))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(T2chperser(:,3))],[0 std(T2chperser(:,3))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,4) % pairing 4: FB85   T1 - NOFB70 T2
hold on;
bar(x2,[mean(T1chperser(:,4)) 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(T2chperser(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(T1chperser(:,4)) 0],[std(T1chperser(:,4))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(T2chperser(:,4))],[0 std(T2chperser(:,4))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,5) % pairing 5: FB85   T2 - NOFB85 T1
hold on;
bar(x2,[mean(T1chperser(:,5)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(T2chperser(:,5))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(T1chperser(:,5)) 0],[std(T1chperser(:,5))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(T2chperser(:,5))],[0 std(T2chperser(:,5))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,6) % pairing 6: NOFB70 T1 - NOFB85 T2
hold on;
bar(x2,[mean(T1chperser(:,6)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(T2chperser(:,6))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(T1chperser(:,6)) 0],[std(T1chperser(:,6))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(T2chperser(:,6))],[0 std(T2chperser(:,6))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off



% Task choice plotted in each of the six pairings and
% additionally split according to fluctuations in objective performance:
figure(5)

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
set(gca,'fontsize',30,'LineWidth',1.5,'XTickLabel',{'','','',''})
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
set(gca,'fontsize',30,'LineWidth',1.5,'XTickLabel',{'','','',''})
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
set(gca,'fontsize',30,'LineWidth',1.5,'XTickLabel',{'','','',''})
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
set(gca,'fontsize',30,'LineWidth',1.5,'XTickLabel',{'','','',''})
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
set(gca,'fontsize',30,'LineWidth',1.5,'XTickLabel',{'','','',''})
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
set(gca,'fontsize',30,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x3)+1 0 1])
hold off





% Objective mean accuracy in each of the six possible task pairings:
figure(6)

subplot(2,3,1) % pairing 1: FB70   T1 - FB85   T2
hold on;
bar(x2,[mean(acct1perser(:,1)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,1))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,1)) 0],[std(acct1perser(:,1))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,1))],[0 std(acct2perser(:,1))/sqrt(nS)],'k.','LineWidth',2)
ylabel('Mean accuracy','fontsize',15)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,2) % pairing 2: FB70   T1 - NOFB70 T2
hold on;
bar(x2,[mean(acct1perser(:,2)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,2))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,2)) 0],[std(acct1perser(:,2))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,2))],[0 std(acct2perser(:,2))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,3) % pairing 3: FB70   T2 - NOFB85 T1
hold on;
bar(x2,[mean(acct1perser(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,3))],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,3)) 0],[std(acct1perser(:,3))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,3))],[0 std(acct2perser(:,3))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,4) % pairing 4: FB85   T1 - NOFB70 T2
hold on;
bar(x2,[mean(acct1perser(:,4)) 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,4)) 0],[std(acct1perser(:,4))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,4))],[0 std(acct2perser(:,4))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,5) % pairing 5: FB85   T2 - NOFB85 T1
hold on;
bar(x2,[mean(acct1perser(:,5)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,5))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,5)) 0],[std(acct1perser(:,5))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,5))],[0 std(acct2perser(:,5))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(2,3,6) % pairing 6: NOFB70 T1 - NOFB85 T2
hold on;
bar(x2,[mean(acct1perser(:,6)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(acct2perser(:,6))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(acct1perser(:,6)) 0],[std(acct1perser(:,6))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(acct2perser(:,6))],[0 std(acct2perser(:,6))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off



% Reaction times plotted in each of the six pairings
figure(7)

subplot(2,3,1) % pairing 1: FB70   T1 - FB85   T2
hold on;
bar(x2,[mean(RTt1perser(:,1)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,1))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,1)) 0],[std(RTt1perser(:,1))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,1))],[0 std(RTt2perser(:,1))/sqrt(nS)],'k.','LineWidth',2)
ylabel('RTs per pairing','fontsize',15)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 800])
hold off

subplot(2,3,2) % pairing 2: FB70   T1 - NOFB70 T2
hold on;
bar(x2,[mean(RTt1perser(:,2)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,2))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,2)) 0],[std(RTt1perser(:,2))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,2))],[0 std(RTt2perser(:,2))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 800])
hold off

subplot(2,3,3) % pairing 3: FB70   T2 - NOFB85 T1
hold on;
bar(x2,[mean(RTt1perser(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,3))],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,3)) 0],[std(RTt1perser(:,3))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,3))],[0 std(RTt2perser(:,3))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 800])
hold off

subplot(2,3,4) % pairing 4: FB85   T1 - NOFB70 T2
hold on;
bar(x2,[mean(RTt1perser(:,4)) 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,4)) 0],[std(RTt1perser(:,4))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,4))],[0 std(RTt2perser(:,4))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 800])
hold off

subplot(2,3,5) % pairing 5: FB85   T2 - NOFB85 T1
hold on;
bar(x2,[mean(RTt1perser(:,5)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,5))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,5)) 0],[std(RTt1perser(:,5))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,5))],[0 std(RTt2perser(:,5))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 800])
hold off

subplot(2,3,6) % pairing 6: NOFB70 T1 - NOFB85 T2
hold on;
bar(x2,[mean(RTt1perser(:,6)) 0],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(RTt2perser(:,6))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(RTt1perser(:,6)) 0],[std(RTt1perser(:,6))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(RTt2perser(:,6))],[0 std(RTt2perser(:,6))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',20,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 800])
hold off




% Main effect of Difficulty manipulation on accuracy and RTs,
% independently from Feedback manipulation:
figure(8)

subplot(2,2,1)
hold on;
bar(1:2,[mean(perf_rt_LB(:,1)) 0],'FaceColor',color_fbdif ,'EdgeColor',[1 1 1])
bar(1:2,[0 mean(perf_rt_LB(:,2))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:2,mean(perf_rt_LB(:,1:2)),std(perf_rt_LB(:,1:2))/sqrt(nS),'k.','LineWidth',2)
ylabel('Performance','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','Dif','Easy',''})
axis([0 3 0 1])
hold off

subplot(2,2,2)
hold on;
bar(1:2,[mean(perf_rt_LB(:,3)) 0],'FaceColor',color_fbdif ,'EdgeColor',[1 1 1])
bar(1:2,[0 mean(perf_rt_LB(:,4))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:2,mean(perf_rt_LB(:,3:4)),std(perf_rt_LB(:,3:4))/sqrt(nS),'k.','LineWidth',2)
ylabel('RT','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','Dif','Easy',''})
axis([0 3 1 1300])
hold off

subplot(2,2,3)
hold on;
bar(1,[mean(perf_rt_LB_rel(:,1))],'FaceColor',[0 0 .6] ,'EdgeColor',[1 1 1])
errorbar(1,mean(perf_rt_LB_rel(:,1)),std(perf_rt_LB_rel(:,1))/sqrt(nS),'k.','LineWidth',2)
ylabel('Performance','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','Easy-Dif',''})
axis([0 2 0 .2])
hold off

subplot(2,2,4)
hold on;
bar(1,[mean(perf_rt_LB_rel(:,2))],'FaceColor',[0 0 .6] ,'EdgeColor',[1 1 1])
errorbar(1,mean(perf_rt_LB_rel(:,2)),std(perf_rt_LB_rel(:,2))/sqrt(nS),'k.','LineWidth',2)
ylabel('RT','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','Easy-Dif',''})
axis([0 2 -280 0])
hold off





% Meta-learning effects over the course of the experiment:
figure(9)

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




% Individual differences in behavior on first vs. second
% halves of the experiment:
figure(10)

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



% Overall learning effect regardless of condition: no consistent drift.
figure(11)
hold on;
errorbar(1:length(x30),mean(MetaL),std(MetaL)/sqrt(nS),'k','LineWidth',2)
ylabel('Meta-learning regardless of condition','fontsize',30)
xlabel('Time (block number)','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2)
axis([0 length(x30)+1 .5 1])
hold off

