% Forming global estimates of self-performance from local confidence
% Rouault M., Dayan P. and Fleming S. M. Nature Communications (2019)
% Experiment 2 (N=29).
% Hierarchical learning model simulations


% We have 28 participants so 28 design matrices, repeated 7 times to obtain
% about 200 simulations:
sim = [1:28 1:28 1:28 1:28 1:28 1:28 1:28];
nS = length(sim);

% Allocate for 6 pairings and 5 learning durations per instance
task1val = zeros(6,5,nS);
task2val = zeros(6,5,nS);

% Instances of design on which model will be run:
load DESIGN


% Loop over simulations
for si = 1:nS
    
    [task1val(:,:,si), task2val(:,:,si)] = PerSubj_simulations(sim(si),DESIGN) ;
    
end


% for graphs:
x5  = 1:5 ;

colorE = [0 153 51]/255 ;
colorD = [255 153 21]/255 ;

couleur = [.8 0 .6] ;


% average over number of simulations
T1valmean = mean(task1val,3) ;
T2valmean = mean(task2val,3) ;
T1valstd  = std(task1val,0,3)/sqrt(nS) ;
T2valstd  = std(task2val,0,3)/sqrt(nS) ;


figure;

subplot(2,3,1)% series type 1: FB70   T1 - FB85   T2
hold on;
errorbar(x5,T1valmean(1,:),T1valstd(1,:),'Color',colorD,'LineWidth',3.5,'LineStyle','-','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
errorbar(x5,T2valmean(1,:),T2valstd(1,:),'Color',colorE,'LineWidth',3.5,'LineStyle','-','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
ylabel('Task choice freq.','fontsize',25)
axis([0 length(x5)+1 0 1])
set(gca,'fontsize',25,'LineWidth',1.5,'YTick',0:.5:1,'XTick',4:4:20,'XTickLabel',{'','4','8','12','16','20',''})
hold off

subplot(2,3,2)% series type 2: FB70   T1 - NOFB70 T2
hold on;
errorbar(x5,T1valmean(2,:),T1valstd(2,:),'Color',colorD,'LineWidth',3.5,'LineStyle','--','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
errorbar(x5,T2valmean(2,:),T2valstd(2,:),'Color',colorD,'LineWidth',3.5,'LineStyle','-','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
axis([0 length(x5)+1 0 1])
set(gca,'fontsize',25,'LineWidth',1.5,'YTick',0:.5:1,'XTick',4:4:20,'XTickLabel',{'','4','8','12','16','20',''})
hold off

subplot(2,3,3)% series type 3: FB70   T2 - NOFB85 T1
hold on;
errorbar(x5,T1valmean(3,:),T1valstd(3,:),'Color',colorE,'LineWidth',3.5,'LineStyle','--','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
errorbar(x5,T2valmean(3,:),T2valstd(3,:),'Color',colorD,'LineWidth',3.5,'LineStyle','-','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
axis([0 length(x5)+1 0 1])
set(gca,'fontsize',25,'LineWidth',1.5,'YTick',0:.5:1,'XTick',4:4:20,'XTickLabel',{'','4','8','12','16','20',''})
hold off

subplot(2,3,4)% series type 4: FB85   T1 - NOFB70 T2
hold on;
errorbar(x5,T1valmean(4,:),T1valstd(4,:),'Color',colorE,'LineWidth',3.5,'LineStyle','-','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
errorbar(x5,T2valmean(4,:),T2valstd(4,:),'Color',colorD,'LineWidth',3.5,'LineStyle','--','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
axis([0 length(x5)+1 0 1])
set(gca,'fontsize',25,'LineWidth',1.5,'YTick',0:.5:1,'XTick',4:4:20,'XTickLabel',{'','4','8','12','16','20',''})
hold off

subplot(2,3,5)% series type 5: FB85   T2 - NOFB85 T1
hold on;
errorbar(x5,T1valmean(5,:),T1valstd(5,:),'Color',colorE,'LineWidth',3.5,'LineStyle','--','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
errorbar(x5,T2valmean(5,:),T2valstd(5,:),'Color',colorE,'LineWidth',3.5,'LineStyle','-','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
xlabel('Learning duration (trials per task)','fontsize',25)
axis([0 length(x5)+1 0 1])
set(gca,'fontsize',25,'LineWidth',1.5,'YTick',0:.5:1,'XTiCk',4:4:20)
set(gca,'XTickLabel',{'','4','8','12','16','20',''})
hold off

subplot(2,3,6)% series type 6: NOFB70 T1 - NOFB85 T2
hold on;
errorbar(x5,T1valmean(6,:),T1valstd(6,:),'Color',colorD,'LineWidth',3.5,'LineStyle','--','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
errorbar(x5,T2valmean(6,:),T2valstd(6,:),'Color',colorE,'LineWidth',3.5,'LineStyle','--','Marker','o','MarkerFaceColor',couleur,'MarkerSize',10)
axis([0 length(x5)+1 0 1])
set(gca,'fontsize',25,'LineWidth',1.5,'YTick',0:.5:1,'XTick',4:4:20,'XTickLabel',{'','4','8','12','16','20',''})
hold off


