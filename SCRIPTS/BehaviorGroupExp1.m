% Forming global estimates of self-performance from local confidence
% Rouault, Dayan and Fleming (2018). Experiment 1 (N=29).

% FACTORS OF INTEREST: Global self-performance estimates:
% 1) Task choice proportion in each of the 4 conditions
% 2) Task ratings according to the 4 conditions




% ---------------------------------------------------
%        Load group data to reproduce plots
% ---------------------------------------------------

load Exp1

T              = Exp1.T ;
T1chperser     = Exp1.T1chperser ;
T2chperser     = Exp1.T2chperser ;
Rnorm          = Exp1.Rnorm ;
P              = Exp1.P ;
PP             = Exp1.PP ;
perf_rt_LB     = Exp1.perf_rt_LB ;
perf_rt_LB_rel = Exp1.perf_rt_LB_rel ;
PerfTB         = Exp1.PerfTB ;
L_H1_H2        = Exp1.L_H1_H2 ;
RT_H1_H2       = Exp1.RT_H1_H2 ;
evol_indiv_RT_LB = Exp1.evol_indiv_RT_LB ;
MetaL          = Exp1.MetaL ;
SPEs_correl    = Exp1.SPEs_correl ;
allSPEs        = Exp1.allSPEs ;
percent_rt_outlier = Exp1.percent_rt_outlier ;

% number of subjects
nS = size(T,1);

% abscissa for graph:
x2 = 1:2 ;
x4 = 1:4 ;
x6 = 1:6 ;
x8 = 1:8 ;
x12 = 1:12 ;

% colors for graph:
color_fbeasy   = [120 147 60 ]/255;
color_fbdif    = [228 108 11 ]/255;
color_nofbeasy = [216 228 189]/255;
color_nofbdif  = [253 214 180]/255;




% ---------------------------------------------------
%        Make the statistical analyses
% ---------------------------------------------------


% print raw averages
disp(['Perf. and RTs in diff. and easy and conditions: ', ...
    num2str(mean(perf_rt_LB))]) ;


% perf easy against diff
disp('Easy vs. Diff. performance, paired t-test: ')
[~,p,~,stats] = ttest(perf_rt_LB(:,1),perf_rt_LB(:,2))

% RTs easy against diff
disp('Easy vs. Diff. RTs, paired t-test: ')
[~,p,~,stats] = ttest(perf_rt_LB(:,3),perf_rt_LB(:,4))


% perf fb vs no fb inside each diff level
disp('Feedback vs. No-Feedback performance, paired t-tests: ')
[~,p,~,stats] = ttest(P(:,1),P(:,3))
[~,p,~,stats] = ttest(P(:,2),P(:,4))

% reaction times fb vs no fb inside each diff level
disp('Feedback vs. No-Feedback RTs, paired t-tests: ')
[~,p,~,stats] = ttest(PP(:,1),PP(:,3))
[~,p,~,stats] = ttest(PP(:,2),PP(:,4))


disp('Correlation between task choice and task ratings: ')
disp(['Median: ',num2str(median(SPEs_correl))])


% Pearson changed to Point-Biserial correlation as task ch. is binary data:
[rho_SPE,~,pval_SPE] = pointbiserial(allSPEs(:,1),allSPEs(:,2)) ;



% ANOVA according to 2 by 2 design:
% factor A [FB,NFB]
% factor B [EASY,DIFF]	

disp('ANOVA on task choice, factors Difficulty and Feedback: ')
% transformation arcsine for proportion data:
[~,F,~,p,~,~,~,~,~]=repanova(2*asin(sqrt(T)),[2 2]);

disp('ANOVA on task ratings, factors Difficulty and Feedback:')
% regular ANOVA for ratings:
[~,F,~,p,~,~,~,~,~]=repanova(Rnorm,[2 2]);



disp(['Mean perf in test blocks for the group: ',num2str(mean(PerfTB))]) ;
disp(['On average, RT outlier trials removed represent: ', ...
    num2str(mean(percent_rt_outlier)),' %']) ;


% Statistical tests comparing performance in first vs 2nd halves:
[~,p,~,stats] = ttest(L_H1_H2(:,1),L_H1_H2(:,2)) ;
[~,p,~,stats] = ttest(L_H1_H2(:,3),L_H1_H2(:,4)) ;
[~,p,~,stats] = ttest(L_H1_H2(:,5),L_H1_H2(:,6)) ;
[~,p,~,stats] = ttest(L_H1_H2(:,7),L_H1_H2(:,8)) ;

% Statistical tests comparing RTs in first vs 2nd halves:
[~,p,~,stats] = ttest(RT_H1_H2(:,1),RT_H1_H2(:,2)) ;
[~,p,~,stats] = ttest(RT_H1_H2(:,3),RT_H1_H2(:,4)) ;
[~,p,~,stats] = ttest(RT_H1_H2(:,5),RT_H1_H2(:,6)) ;
[~,p,~,stats] = ttest(RT_H1_H2(:,7),RT_H1_H2(:,8)) ;




% ---------------------------------------------------
%        Make the figures
% ---------------------------------------------------


% Performance, Task choice and Task ratings over the 4 experimental cond:
figure(1)

subplot(1,3,1)
hold on;
bar(x4,[mean(P(:,1)) 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 mean(P(:,2)) 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x4,[0 0 mean(P(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 0 0 mean(P(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x4),mean(P),std(P)/sqrt(nS),'k.','LineWidth',2)
ylabel('Mean performance in learning blocks','fontsize',25)
set(gca,'fontsize',25,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x4)+1 .5 1])
hold off

subplot(1,3,2)
hold on;
bar(x4,[mean(T(:,1)) 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 mean(T(:,2)) 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x4,[0 0 mean(T(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 0 0 mean(T(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x4),mean(T),std(T)/sqrt(nS),'k.','LineWidth',2)
ylabel('Task choice frequency','fontsize',25)
set(gca,'fontsize',25,'LineWidth',1.5,'XTickLabel',{'','','',''})
% legend('FB easy','FB diff','NO FB easy','NO FB diff')
axis([0 length(x4)+1 0 .5])
hold off

subplot(1,3,3)
hold on;
bar(x4,[mean(Rnorm(:,1)) 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 mean(Rnorm(:,2)) 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x4,[0 0 mean(Rnorm(:,3)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x4,[0 0 0 mean(Rnorm(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x4),mean(Rnorm),std(Rnorm)/sqrt(nS),'k.','LineWidth',2)
ylabel('Task ability ratings','fontsize',25)
set(gca,'fontsize',25,'LineWidth',1.5,'XTickLabel',{'','','',''})
legend('FB easy','FB diff','NO FB easy','NO FB diff')
axis([0 length(x4)+1 0 1])
hold off




% Task choice plotted per pairing
figure(2)

subplot(3,6,1) % pairing 1: FB70   T1 - FB85   T2
hold on;
bar(x2,[0 mean(T1chperser(:,1))],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[mean(T2chperser(:,1)) 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[0 mean(T1chperser(:,1))],[0 std(T1chperser(:,1))/sqrt(nS)],'k.','LineWidth',2)
errorbar(1:length(x2),[mean(T2chperser(:,1)) 0],[std(T2chperser(:,1))/sqrt(nS) 0],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(3,6,2) % pairing 2: FB70   T1 - NOFB70 T2
hold on;
bar(x2,[mean(T1chperser(:,2)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x2,[0 mean(T2chperser(:,2))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(T1chperser(:,2)) 0],[std(T1chperser(:,2))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(T2chperser(:,2))],[0 std(T2chperser(:,2))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(3,6,3) % pairing 3: FB70   T2 - NOFB85 T1
hold on;
bar(x2,[0 mean(T1chperser(:,3))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[mean(T2chperser(:,3)) 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[0 mean(T1chperser(:,3))],[0 std(T1chperser(:,3))/sqrt(nS)],'k.','LineWidth',2)
errorbar(1:length(x2),[mean(T2chperser(:,3)) 0],[std(T2chperser(:,3))/sqrt(nS) 0],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(3,6,4) % pairing 4: FB85   T1 - NOFB70 T2
hold on;
bar(x2,[mean(T1chperser(:,4)) 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x2,[0 mean(T2chperser(:,4))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[mean(T1chperser(:,4)) 0],[std(T1chperser(:,4))/sqrt(nS) 0],'k.','LineWidth',2)
errorbar(1:length(x2),[0 mean(T2chperser(:,4))],[0 std(T2chperser(:,4))/sqrt(nS)],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(3,6,5) % pairing 5: FB85   T2 - NOFB85 T1
hold on;
bar(x2,[0 mean(T1chperser(:,5))],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
bar(x2,[mean(T2chperser(:,5)) 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[0 mean(T1chperser(:,5))],[0 std(T1chperser(:,5))/sqrt(nS)],'k.','LineWidth',2)
errorbar(1:length(x2),[mean(T2chperser(:,5)) 0],[std(T2chperser(:,5))/sqrt(nS) 0],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(3,6,6) % pairing 6: NOFB70 T1 - NOFB85 T2
hold on;
bar(x2,[0 mean(T1chperser(:,6))],'FaceColor',color_nofbdif,'EdgeColor',[1 1 1])
bar(x2,[mean(T2chperser(:,6)) 0],'FaceColor',color_nofbeasy,'EdgeColor',[1 1 1])
errorbar(1:length(x2),[0 mean(T1chperser(:,6))],[0 std(T1chperser(:,6))/sqrt(nS)],'k.','LineWidth',2)
errorbar(1:length(x2),[mean(T2chperser(:,6)) 0],[std(T2chperser(:,6))/sqrt(nS) 0],'k.','LineWidth',2)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off



% Accuracy and RT according to Difficulty manipulation:
figure(3)

subplot(2,2,1)
hold on;
bar(1:2,[mean(perf_rt_LB(:,1)) 0],'FaceColor',color_fbdif ,'EdgeColor',[1 1 1])
bar(1:2,[0 mean(perf_rt_LB(:,2))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:2,mean(perf_rt_LB(:,1:2)),std(perf_rt_LB(:,1:2))/sqrt(nS),'k.','LineWidth',2)
ylabel('Accuracy, learning blocks','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','Dif','Easy',''})
axis([0 3 0 1])
hold off

subplot(2,2,2)
hold on;
bar(1:2,[mean(perf_rt_LB(:,3)) 0],'FaceColor',color_fbdif ,'EdgeColor',[1 1 1])
bar(1:2,[0 mean(perf_rt_LB(:,4))],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
errorbar(1:2,mean(perf_rt_LB(:,3:4)),std(perf_rt_LB(:,3:4))/sqrt(nS),'k.','LineWidth',2)
ylabel('Reaction times, learning blocks','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','Dif','Easy',''})
axis([0 3 1 1000])
hold off

subplot(2,2,3)
hold on;
bar(1,mean(perf_rt_LB_rel(:,1)),'FaceColor',[0 0 .6] ,'EdgeColor',[1 1 1])
errorbar(1,mean(perf_rt_LB_rel(:,1)),std(perf_rt_LB_rel(:,1))/sqrt(nS),'k.','LineWidth',2)
ylabel('Accuracy, learning blocks','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','Easy-Dif',''})
axis([0 2 0 .3])
hold off

subplot(2,2,4)
hold on;
bar(1,mean(perf_rt_LB_rel(:,2)),'FaceColor',[0 0 .6] ,'EdgeColor',[1 1 1])
errorbar(1,mean(perf_rt_LB_rel(:,2)),std(perf_rt_LB_rel(:,2))/sqrt(nS),'k.','LineWidth',2)
ylabel('Reaction times, learning blocks','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','Easy-Dif',''})
axis([0 2 -70 0])
hold off



% Meta-learning effects
figure(4)

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
        'fbE_H2' 'fbD_H1' 'fbD_H2'...
        'nofbE_H1' 'nofbE_H2' 'nofbD_H1' 'nofbD_H2' ''})
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
axis([0 length(x8)+1 0 700])
hold off

subplot(3,1,3)
hold on;
bar(x8,[mean(evol_indiv_RT_LB(:,1))  0 0 0 0 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x8,[0 mean(evol_indiv_RT_LB(:,2))  0 0 0 0 0 0],'FaceColor',color_fbeasy,'EdgeColor',[1 1 1])
bar(x8,[0 0 mean(evol_indiv_RT_LB(:,3))  0 0 0 0 0],'FaceColor',color_fbdif,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 mean(evol_indiv_RT_LB(:,4))  0 0 0 0],'FaceColor',color_fbdif ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 mean(evol_indiv_RT_LB(:,5))  0 0 0],'FaceColor',color_nofbeasy ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 mean(evol_indiv_RT_LB(:,6))  0 0],'FaceColor',color_nofbeasy ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 0 mean(evol_indiv_RT_LB(:,7))  0],'FaceColor',color_nofbdif ,'EdgeColor',[1 1 1])
bar(x8,[0 0 0 0 0 0 0 mean(evol_indiv_RT_LB(:,8)) ],'FaceColor',color_nofbdif ,'EdgeColor',[1 1 1])
errorbar(x8,mean(evol_indiv_RT_LB),std(evol_indiv_RT_LB)/sqrt(nS),'k.','LineWidth',2)
ylabel('RTs (log)','fontsize',25)
    set(gca,'FontSize',15,'LineWidth',1.2,'XTickLabel',{'' 'fbE_H1'...
        'fbE_H2' 'fbD_H1' 'fbD_H2'...
        'nofbE_H1' 'nofbE_H2' 'nofbD_H1' 'nofbD_H2' ''})
axis([0 length(x8)+1 -.7 .7])
hold off




% Individual differences in speed-up between first and 2nd halves of expe.:
figure(5)
subplot(2,2,1)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,1),L_H1_H2(i,2)],'Color',color_fbeasy,'LineWidth',3)
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
ylabel('Performance','fontsize',30)
axis([0 3 .5 1])
hold off

subplot(2,2,2)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,3),L_H1_H2(i,4)],'Color',color_fbdif,'LineWidth',3)
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
axis([0 3 .5 1])
hold off

subplot(2,2,3)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,5),L_H1_H2(i,6)],'Color',color_nofbeasy,'LineWidth',3)
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
ylabel('Performance','fontsize',30)
axis([0 3 .5 1])
hold off

subplot(2,2,4)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,7),L_H1_H2(i,8)],'Color',color_nofbdif,'LineWidth',3)
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
axis([0 3 .5 1])
hold off



figure(6)
hold on;
errorbar(1:length(x12),mean(MetaL),std(MetaL)/sqrt(nS),'k','LineWidth',2)
ylabel('Meta-learning regardless of condition','fontsize',30)
xlabel('Block number)','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2)
axis([0 length(x12)+1 .5 1])
hold off


