% Forming global estimates of self-performance from local confidence
% Rouault M., Dayan P. and Fleming S. M. Nature Communications (2019).  
% Experiment 1 (N=29).

% FACTORS OF INTEREST: Global self-performance estimates:
% 1) Task choice proportion in each of the 4 conditions
% 2) Task ratings according to the 4 conditions




% ---------------------------------------------------
%        Load group data to reproduce plots
% ---------------------------------------------------

load Exp1

T                  = Exp1.T ;
T1chperser         = Exp1.T1chperser ;
T2chperser         = Exp1.T2chperser ;
T1rateperser       = Exp1.T1rateperser;
T2rateperser       = Exp1.T2rateperser;
Rnorm              = Exp1.Rnorm ;
P                  = Exp1.P ;
PP                 = Exp1.PP ;
perf_rt_LB         = Exp1.perf_rt_LB ;
perf_rt_LB_rel     = Exp1.perf_rt_LB_rel ;
PerfTB             = Exp1.PerfTB ;
L_H1_H2            = Exp1.L_H1_H2 ;
RT_H1_H2           = Exp1.RT_H1_H2 ;
evol_indiv_RT_LB   = Exp1.evol_indiv_RT_LB ;
MetaL              = Exp1.MetaL ;
SPEs_correl        = Exp1.SPEs_correl ;
allSPEs            = Exp1.allSPEs ;
percent_rt_outlier = Exp1.percent_rt_outlier ;
rating_ch_unch     = Exp1.rating_ch_unch ;
rating_ch_unch_ffx = Exp1.rating_ch_unch_ffx ;
Ytaskch            = Exp1.Ytaskch ;
Xdiff_in_rating    = Exp1.Xdiff_in_rating ;
FIRST4             = Exp1.FIRST4 ;
SECOND4            = Exp1.SECOND4 ;
THIRD4             = Exp1.THIRD4 ;
FOURTH4            = Exp1.FOURTH4 ;
Yorder             = Exp1.Yorder ;


% number of subjects
nS = size(T,1);

% abscissa for graph:
x2 = 1:2 ;
x4 = 1:4 ;
x6 = 1:6 ;
x8 = 1:8 ;
x12 = 1:12 ;

sss = 0.27 ;
ttt = 0.17 ;

% colors for graphs:
colorE = [0 153 51]/255 ;
colorD = [255 153 21]/255 ;
color_grey = [.2 .2 .2] ;



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


% Are task ability ratings and task choices consistent SPEs measures?
DiffRatings = rating_ch_unch_ffx(:,1)-rating_ch_unch_ffx(:,2) ;
howOftenRatingHigher4Chosen = length(find(DiffRatings>0))/length(DiffRatings)*100 ;

disp(['How often the chosen task is rated higher than the unchosen task:', ...
    num2str(howOftenRatingHigher4Chosen),'% over all blocks over all subjects'])


% Ask if difference in ratings between tasks predicts task choice:
Ytaskch = boolean(Ytaskch) ; % convert vector in logical

[B, DEV, STATS] = glmfit(Xdiff_in_rating,Ytaskch,'binomial','link','logit') ;

yhat = glmval(B,Xdiff_in_rating,'logit');

% percentage of variance explained
[r2,r2adj] = rsquared(Ytaskch,yhat); 



% ---------------------------------------------------
%        Make the figures
% ---------------------------------------------------



% Performance, Task choice and Task ratings over the 4 experimental 
% conditions. Main Figure 2.

figure(2)

subplot(1,3,1)
hold on;
bar(x4,[mean(P(:,1)) 0 0 0],'FaceColor',colorE,'EdgeColor',colorE,'LineWidth',5)
bar(x4,[0 mean(P(:,2)) 0 0],'FaceColor',colorD,'EdgeColor',colorD,'LineWidth',5)
bar(x4,[0 0 mean(P(:,3)) 0],'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',5,'LineStyle','--')
bar(x4,[0 0 0 mean(P(:,4))],'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',5,'LineStyle','--')
for o = 1:nS
    hold on;
    plot([1+sss,1+ttt],[P(o,1),P(o,1)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[P(o,2),P(o,2)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([3+sss,3+ttt],[P(o,3),P(o,3)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([4+sss,4+ttt],[P(o,4),P(o,4)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(1:length(x4),mean(P),std(P)/sqrt(nS),'k.','LineWidth',5)
ylabel('Mean performance','fontsize',25)
set(gca,'fontsize',25,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x4)+1 .5 1])
hold off

subplot(1,3,2)
bar(1, mean(T(:,1)),'FaceColor',colorE,'EdgeColor',colorE,'LineWidth',5)
hold on
bar(2, mean(T(:,2)),'FaceColor',colorD,'EdgeColor',colorD,'LineWidth',5)
hold on
bar(3, mean(T(:,3)), 'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',5,'LineStyle','--')
hold on
bar(4, mean(T(:,4)), 'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',5,'LineStyle','--')
for o = 1:nS
    hold on;
    plot([1+sss,1+ttt],[T(o,1),T(o,1)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[T(o,2),T(o,2)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([3+sss,3+ttt],[T(o,3),T(o,3)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([4+sss,4+ttt],[T(o,4),T(o,4)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(1:length(x4),mean(T),std(T)/sqrt(nS),'k.','LineWidth',5)
ylabel('Task choice frequency','fontsize',25)
set(gca,'fontsize',25,'LineWidth',1.5,'XTickLabel',{'','','',''})
legend('FB easy','FB diff','NO FB easy','NO FB diff')
axis([0 length(x4)+1 0 .5])
hold off

subplot(1,3,3)
bar(1, mean(Rnorm(:,1)),'FaceColor',colorE,'EdgeColor',colorE,'LineWidth',5)
hold on
bar(2, mean(Rnorm(:,2)),'FaceColor',colorD,'EdgeColor',colorD,'LineWidth',5)
hold on
bar(3, mean(Rnorm(:,3)), 'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',5,'LineStyle','--')
hold on
bar(4, mean(Rnorm(:,4)), 'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',5,'LineStyle','--')
for o = 1:nS
    hold on;
    plot([1+sss,1+ttt],[Rnorm(o,1),Rnorm(o,1)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[Rnorm(o,2),Rnorm(o,2)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([3+sss,3+ttt],[Rnorm(o,3),Rnorm(o,3)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([4+sss,4+ttt],[Rnorm(o,4),Rnorm(o,4)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(1:length(x4),mean(Rnorm),std(Rnorm)/sqrt(nS),'k.','LineWidth',5)
ylabel('Task ability ratings','fontsize',25)
set(gca,'fontsize',25,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x4)+1 0 1])
hold off




% Task choice and task ability ratings plotted per pairing (Main Figure 3)
figure(3)

subplot(4,7,1) % series type 1: FB70   T1 - FB85   T2
bar(1, mean(T2chperser(:,1)),'FaceColor',colorE,'EdgeColor',colorE,'LineWidth',5)
hold on
bar(2, mean(T1chperser(:,1)),'FaceColor',colorD,'EdgeColor',colorD,'LineWidth',5)
errorbar(x2,[mean(T2chperser(:,1)) mean(T1chperser(:,1))], ...
    [std(T2chperser(:,1)) std(T1chperser(:,1))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,2) % series type 2: FB70   T1 - NOFB70 T2
bar(1, mean(T1chperser(:,2)),'FaceColor',colorD,'EdgeColor',colorD,'LineWidth',5)
hold on
bar(2, mean(T2chperser(:,2)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',5,'LineStyle','--')
errorbar(x2,[mean(T1chperser(:,2)) mean(T2chperser(:,2))], ...
    [std(T1chperser(:,2)) std(T2chperser(:,2))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,3) % series type 3: FB70   T2 - NOFB85 T1
bar(1, mean(T2chperser(:,3)),'FaceColor',colorD,'EdgeColor',colorD,'LineWidth',5)
hold on
bar(2, mean(T1chperser(:,3)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',5,'LineStyle','--')
errorbar(x2,[mean(T2chperser(:,3)) mean(T1chperser(:,3))], ...
    [std(T2chperser(:,3)) std(T1chperser(:,3))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,4) % series type 4: FB85   T1 - NOFB70 T2
bar(1, mean(T1chperser(:,4)),'FaceColor',colorE,'EdgeColor',colorE,'LineWidth',5)
hold on
bar(2, mean(T2chperser(:,4)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',5,'LineStyle','--')
errorbar(x2,[mean(T1chperser(:,4)) mean(T2chperser(:,4))], ...
    [std(T1chperser(:,4)) std(T2chperser(:,4))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,5) % series type 5: FB85   T2 - NOFB85 T1
bar(1, mean(T2chperser(:,5)),'FaceColor',colorE,'EdgeColor',colorE,'LineWidth',5)
hold on
bar(2, mean(T1chperser(:,5)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',5,'LineStyle','--')
errorbar(x2,[mean(T2chperser(:,5)) mean(T1chperser(:,5))], ...
    [std(T2chperser(:,5)) std(T1chperser(:,5))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,6) % series type 6: NOFB70 T1 - NOFB85 T2
bar(1, mean(T2chperser(:,6)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',5,'LineStyle','--')
hold on
bar(2, mean(T1chperser(:,6)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',5,'LineStyle','--')
errorbar(x2,[mean(T2chperser(:,6)) mean(T1chperser(:,6))], ...
    [std(T2chperser(:,6)) std(T1chperser(:,6))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,8) % series type 1: FB70   T1 - FB85   T2
bar(1, mean(T2rateperser(:,1)),'FaceColor',colorE,'EdgeColor',colorE,'LineWidth',5)
hold on
bar(2, mean(T1rateperser(:,1)),'FaceColor',colorD,'EdgeColor',colorD,'LineWidth',5)
for o = 1:nS
    hold on;
    plot([1+sss,1+ttt],[T2rateperser(o,1),T2rateperser(o,1)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[T1rateperser(o,1),T1rateperser(o,1)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(x2,[mean(T2rateperser(:,1)) mean(T1rateperser(:,1))], ...
    [std(T2rateperser(:,1)) std(T1rateperser(:,1))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,9) % series type 2: FB70   T1 - NOFB70 T2
bar(1, mean(T1rateperser(:,2)),'FaceColor',colorD,'EdgeColor',colorD,'LineWidth',5)
hold on
bar(2, mean(T2rateperser(:,2)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',5,'LineStyle','--')
for o = 1:nS
    hold on;
    plot([1+sss,1+ttt],[T1rateperser(o,2),T1rateperser(o,2)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[T2rateperser(o,2),T2rateperser(o,2)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(x2,[mean(T1rateperser(:,2)) mean(T2rateperser(:,2))], ...
    [std(T1rateperser(:,2)) std(T2rateperser(:,2))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,10) % series type 3: FB70   T2 - NOFB85 T1
bar(1, mean(T2rateperser(:,3)),'FaceColor',colorD,'EdgeColor',colorD,'LineWidth',5)
hold on
bar(2, mean(T1rateperser(:,3)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',5,'LineStyle','--')
for o = 1:nS
    hold on;
    plot([1+sss,1+ttt],[T2rateperser(o,3),T2rateperser(o,3)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[T1rateperser(o,3),T1rateperser(o,3)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(x2,[mean(T2rateperser(:,3)) mean(T1rateperser(:,3))], ...
    [std(T2rateperser(:,3)) std(T1rateperser(:,3))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,11) % series type 4: FB85   T1 - NOFB70 T2
bar(1, mean(T1rateperser(:,4)),'FaceColor',colorE,'EdgeColor',colorE,'LineWidth',5)
hold on
bar(2, mean(T2rateperser(:,4)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',5,'LineStyle','--')
for o = 1:nS
    hold on;
    plot([1+sss,1+ttt],[T1rateperser(o,4),T1rateperser(o,4)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[T2rateperser(o,4),T2rateperser(o,4)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(x2,[mean(T1rateperser(:,4)) mean(T2rateperser(:,4))],[std(T1rateperser(:,4)) std(T2rateperser(:,4))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,12) % series type 5: FB85   T2 - NOFB85 T1
bar(1, mean(T2rateperser(:,5)),'FaceColor',colorE,'EdgeColor',colorE,'LineWidth',5)
hold on
bar(2, mean(T1rateperser(:,5)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',5,'LineStyle','--')
for o = 1:nS
    hold on;
    plot([1+sss,1+ttt],[T2rateperser(o,5),T2rateperser(o,5)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[T1rateperser(o,5),T1rateperser(o,5)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(x2,[mean(T2rateperser(:,5)) mean(T1rateperser(:,5))],[std(T2rateperser(:,5)) std(T1rateperser(:,5))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off

subplot(4,7,13) % series type 6: NOFB70 T1 - NOFB85 T2
bar(1, mean(T2rateperser(:,6)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',5,'LineStyle','--')
hold on
bar(2, mean(T1rateperser(:,6)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',5,'LineStyle','--')
for o = 1:nS
    hold on;
    plot([1+sss,1+ttt],[T2rateperser(o,6),T2rateperser(o,6)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[T1rateperser(o,6),T1rateperser(o,6)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(x2,[mean(T2rateperser(:,6)) mean(T1rateperser(:,6))], ...
    [std(T2rateperser(:,6)) std(T1rateperser(:,6))]/sqrt(nS),'k.','LineWidth',5)
set(gca,'fontsize',22,'LineWidth',1.5,'XTickLabel',{'','','',''})
axis([0 length(x2)+1 0 1])
hold off


% Task ability ratings for chosen and unchosen tasks:
subplot(4,7,[7 ; 14])
hold on;
for o = 1:nS
    plot([1+sss,1+ttt],[rating_ch_unch(o,1),rating_ch_unch(o,1)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
    plot([2+sss,2+ttt],[rating_ch_unch(o,2),rating_ch_unch(o,2)],'Color',color_grey,'LineWidth',2,'LineStyle','-') ;
end
errorbar(x2,mean(rating_ch_unch),std(rating_ch_unch)/sqrt(nS),'b','LineWidth',5)
set(gca,'fontsize',22, 'LineWidth', 1.5, 'XTickLabel',{'','Ch.','Unch.',''})
xlabel('Task choice','fontsize', 22)
axis([0 3 0 1])
hold off


% Pie charts indicating individual data points for task choice in each of
% the six series types: [pie corresponds to SECOND bar of each plot]
% Relate to main Figure 3.

figure(31)
h=pie([length(find(T1chperser(:,1)==0))/length(T1chperser(:,1)) ...
    length(find(T1chperser(:,1)==0.5))/length(T1chperser(:,1)) ...
    length(find(T1chperser(:,1)==1))/length(T1chperser(:,1))],{'0','0.5','1'});
set(findobj(h,'type','text'),'fontsize',80);
colormap([0 0 0;.5 .5 .5;1 1 1]);

figure(32)
h=pie([length(find(T2chperser(:,2)==0))/length(T2chperser(:,2)) ...
    length(find(T2chperser(:,2)==0.5))/length(T2chperser(:,2))],{'0','0.5'});
set(findobj(h,'type','text'),'fontsize',80);
colormap([0 0 0;.5 .5 .5]);

figure(33)
h=pie([length(find(T1chperser(:,3)==0))/length(T1chperser(:,3)) ...
    length(find(T1chperser(:,3)==0.5))/length(T1chperser(:,3)) ...
    length(find(T1chperser(:,3)==1))/length(T1chperser(:,3))],{'0','0.5','1'});
set(findobj(h,'type','text'),'fontsize',80);
colormap([0 0 0;.5 .5 .5;1 1 1]);

figure(34)
h=pie([length(find(T2chperser(:,4)==0))/length(T2chperser(:,4)) ...
    length(find(T2chperser(:,4)==0.5))/length(T2chperser(:,4))],{'0','0.5'});
set(findobj(h,'type','text'),'fontsize',80);
colormap([0 0 0;.5 .5 .5]);

figure(35)
h=pie([length(find(T1chperser(:,5)==0))/length(T1chperser(:,5)) ...
    length(find(T1chperser(:,5)==0.5))/length(T1chperser(:,5)) ...
    length(find(T1chperser(:,5)==1))/length(T1chperser(:,5))],{'0','0.5','1'});
set(findobj(h,'type','text'),'fontsize',80);
colormap([0 0 0;.5 .5 .5;1 1 1]);

figure(36)
h=pie([length(find(T1chperser(:,6)==0))/length(T1chperser(:,6)) ...
    length(find(T1chperser(:,6)==0.5))/length(T1chperser(:,6)) ...
    length(find(T1chperser(:,6)==1))/length(T1chperser(:,6))],{'0','0.5','1'});
set(findobj(h,'type','text'),'fontsize',80);
colormap([0 0 0;.5 .5 .5;1 1 1]);




% Accuracy and RT according to Difficulty manipulation:
figure(4)

subplot(2,2,1)
hold on;
bar(1:2,[mean(perf_rt_LB(:,1)) 0],'FaceColor',colorD ,'EdgeColor',[1 1 1])
bar(1:2,[0 mean(perf_rt_LB(:,2))],'FaceColor',colorE,'EdgeColor',[1 1 1])
errorbar(1:2,mean(perf_rt_LB(:,1:2)),std(perf_rt_LB(:,1:2))/sqrt(nS),'k.','LineWidth',2)
ylabel('Accuracy, learning blocks','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2,'XTickLabel',{'','Dif','Easy',''})
axis([0 3 0 1])
hold off

subplot(2,2,2)
hold on;
bar(1:2,[mean(perf_rt_LB(:,3)) 0],'FaceColor',colorD ,'EdgeColor',[1 1 1])
bar(1:2,[0 mean(perf_rt_LB(:,4))],'FaceColor',colorE,'EdgeColor',[1 1 1])
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
figure(5)

subplot(3,1,1)
hold on;
bar(1, mean(L_H1_H2(:,1)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','-')
hold on
bar(2, mean(L_H1_H2(:,2)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','-')
hold on
bar(3, mean(L_H1_H2(:,3)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','-')
hold on
bar(4, mean(L_H1_H2(:,4)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','-')
hold on
bar(5, mean(L_H1_H2(:,5)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','--')
hold on
bar(6, mean(L_H1_H2(:,6)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','--')
hold on
bar(7, mean(L_H1_H2(:,7)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','--')
hold on
bar(8, mean(L_H1_H2(:,8)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','--')
errorbar(x8,mean(L_H1_H2),std(L_H1_H2)/sqrt(nS),'k.','LineWidth',2)
ylabel('Perf','fontsize',25)
    set(gca,'FontSize',15,'LineWidth',1.2,'XTickLabel',{'' 'fbE_H1'...
        'fbE_H2' 'fbD_H1' 'fbD_H2'...
        'nofbE_H1' 'nofbE_H2' 'nofbD_H1' 'nofbD_H2' ''})
axis([0 length(x8)+1 0 1])
hold off

subplot(3,1,2)

hold on;
bar(1, mean(RT_H1_H2(:,1)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','-')
hold on
bar(2, mean(RT_H1_H2(:,2)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','-')
hold on
bar(3, mean(RT_H1_H2(:,3)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','-')
hold on
bar(4, mean(RT_H1_H2(:,4)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','-')
hold on
bar(5, mean(RT_H1_H2(:,5)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','--')
hold on
bar(6, mean(RT_H1_H2(:,6)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','--')
hold on
bar(7, mean(RT_H1_H2(:,7)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','--')
hold on
bar(8, mean(RT_H1_H2(:,8)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','--')
errorbar(x8,mean(RT_H1_H2),std(RT_H1_H2)/sqrt(nS),'k.','LineWidth',2)
ylabel('RTs (raw)','fontsize',25)
    set(gca,'FontSize',15,'LineWidth',1.2,'XTickLabel',{'' 'fbE_H1'...
        'fbE_H2' 'fbD_H1' 'fbD_H2'...
        'nofbE_H1' 'nofbE_H2' 'nofbD_H1' 'nofbD_H2' ''})
axis([0 length(x8)+1 0 1000])
hold off

subplot(3,1,3)

hold on;
bar(1, mean(evol_indiv_RT_LB(:,1)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','-')
hold on
bar(2, mean(evol_indiv_RT_LB(:,2)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','-')
hold on
bar(3, mean(evol_indiv_RT_LB(:,3)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','-')
hold on
bar(4, mean(evol_indiv_RT_LB(:,4)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','-')
hold on
bar(5, mean(evol_indiv_RT_LB(:,5)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','--')
hold on
bar(6, mean(evol_indiv_RT_LB(:,6)),'FaceColor',[1 1 1],'EdgeColor',colorE,'LineWidth',3,'LineStyle','--')
hold on
bar(7, mean(evol_indiv_RT_LB(:,7)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','--')
hold on
bar(8, mean(evol_indiv_RT_LB(:,8)),'FaceColor',[1 1 1],'EdgeColor',colorD,'LineWidth',3,'LineStyle','--')
errorbar(x8,mean(evol_indiv_RT_LB),std(evol_indiv_RT_LB)/sqrt(nS),'k.','LineWidth',2)
ylabel('RTs (log)','fontsize',25)
    set(gca,'FontSize',15,'LineWidth',1.2,'XTickLabel',{'' 'fbE_H1'...
        'fbE_H2' 'fbD_H1' 'fbD_H2'...
        'nofbE_H1' 'nofbE_H2' 'nofbD_H1' 'nofbD_H2' ''})
axis([0 length(x8)+1 -.7 .7])
hold off




% Individual differences in speed-up between first and 2nd halves of expe.:
figure(6)
subplot(2,2,1)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,1),L_H1_H2(i,2)],'Color',colorE,'LineWidth',3,'LineStyle','-')
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
ylabel('Performance','fontsize',30)
axis([0 3 .5 1])
hold off

subplot(2,2,2)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,3),L_H1_H2(i,4)],'Color',colorD,'LineWidth',3,'LineStyle','-')
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
axis([0 3 .5 1])
hold off

subplot(2,2,3)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,5),L_H1_H2(i,6)],'Color',colorE,'LineWidth',3,'LineStyle','--')
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
ylabel('Performance','fontsize',30)
axis([0 3 .5 1])
hold off

subplot(2,2,4)
for i=1:nS
    hold on;
    plot(1:2,[L_H1_H2(i,7),L_H1_H2(i,8)],'Color',colorD,'LineWidth',3,'LineStyle','--')
end
set(gca,'FontSize',30,'LineWidth',1.2,'XTickLabel',{'' '1st_half' '2nd_half' ''})
axis([0 3 .5 1])
hold off


% Overall meta-learning
figure(7)
hold on;
errorbar(1:length(x12),mean(MetaL),std(MetaL)/sqrt(nS),'k','LineWidth',2)
ylabel('Meta-learning regardless of condition','fontsize',30)
xlabel('Block number)','fontsize',30)
set(gca,'fontsize',30,'LineWidth',1.2)
axis([0 length(x12)+1 .5 1])
hold off



% Order effects
% Supplementary Figure 1

% z-score all regressors for commensurability of regression coefficients:
FIRST4  = (FIRST4-mean(FIRST4))/std(FIRST4);
SECOND4 = (SECOND4-mean(SECOND4))/std(SECOND4);
THIRD4  = (THIRD4-mean(THIRD4))/std(THIRD4);
FOURTH4 = (FOURTH4-mean(FOURTH4))/std(FOURTH4);

X = [FIRST4 SECOND4 THIRD4 FOURTH4];

Yorder = boolean(Yorder); % convert vector in logical (islogical)

[Border,~,STATSorder] = glmfit(X,Yorder,'binomial','link','logit');


figure(8)
hold on;
bar(Border)
errorbar(1:5,Border,STATSorder.se,'bo','LineWidth',3)
plot([0 0 0 0 0],'k--')
set(gca,'fontsize',25, 'LineWidth', 1.5,'XTick',1:5,'XTickLabel',{'intercept','1st','2nd','3rd','last'})
ylabel('Contribution to task choice (a.u.)','fontsize',25)
xlabel('Position inside learning block','fontsize',25)
title('Accuracy','fontsize',25)
hold off

