% Forming global estimates of self-performance from local confidence
% Rouault M., Dayan P. and Fleming S. M. Nature Communications (2019)
% Experiment 2 (N=29).
% Hierarchical learning model simulations



function [task1val, task2val] = PerSubj_simulations(si,DESIGN)


% experimental design on which the model is run, as if it was a subject:
acc_level = DESIGN{si}.acc_level;
fb_type = DESIGN{si}.fb_type;
current_task = DESIGN{si}.current_task;
target_left = DESIGN{si}.target_left;
ser_type = DESIGN{si}.ser_type;
LBduration = DESIGN{si}.LBduration;
k_from_d_prime = DESIGN{si}.k_from_d_prime;
k = DESIGN{si}.k;
Position_whichtask = DESIGN{si}.Position_whichtask ;
seriestype = DESIGN{si}.seriestype ;


% calls the model
[chosen_task] = BAY_learning_model(acc_level,fb_type,current_task,...
    target_left,ser_type,LBduration,k_from_d_prime,k) ;


% collect sequence of task choices
chosen_task = chosen_task ;


% allocate for 6 pairings and 5 block durations
task1val = zeros(6,5) ;
task2val = zeros(6,5) ;

% re-map so LBduration is an index:
LBduree = LBduration ;
LBduree(LBduration == 4)  = 1 ;
LBduree(LBduration == 8)  = 2 ;
LBduree(LBduration == 12) = 3 ;
LBduree(LBduration == 16) = 4 ;
LBduree(LBduration == 20) = 5 ;


i = 0 ;

for pos = Position_whichtask % loop through the 30 blocks
    i = i+1 ;
    
    % series type 1: FB70   T1 - FB85   T2
    if     seriestype(pos) == 1 && chosen_task(i) == 1
        task1val(seriestype(pos),LBduree(i)) = task1val(seriestype(pos),LBduree(i)) + 1 ;
    elseif seriestype(pos) == 1 && chosen_task(i) == 2
        task2val(seriestype(pos),LBduree(i)) = task2val(seriestype(pos),LBduree(i)) + 1 ;
        
        % series type 2: FB70   T1 - NOFB70 T2
    elseif seriestype(pos) == 2 && chosen_task(i) == 1
        task1val(seriestype(pos),LBduree(i)) = task1val(seriestype(pos),LBduree(i)) + 1 ;
    elseif seriestype(pos) == 2 && chosen_task(i) == 2
        task2val(seriestype(pos),LBduree(i)) = task2val(seriestype(pos),LBduree(i)) + 1 ;
        
        % series type 3: FB70   T2 - NOFB85 T1
    elseif seriestype(pos) == 3 && chosen_task(i) == 1
        task1val(seriestype(pos),LBduree(i)) = task1val(seriestype(pos),LBduree(i)) + 1 ;
    elseif seriestype(pos) == 3 && chosen_task(i) == 2
        task2val(seriestype(pos),LBduree(i)) = task2val(seriestype(pos),LBduree(i)) + 1 ;
        
        % series type 4: FB85   T1 - NOFB70 T2
    elseif seriestype(pos) == 4 && chosen_task(i) == 1
        
        task1val(seriestype(pos),LBduree(i)) = task1val(seriestype(pos),LBduree(i)) + 1 ;
    elseif seriestype(pos) == 4 && chosen_task(i) == 2
        task2val(seriestype(pos),LBduree(i)) = task2val(seriestype(pos),LBduree(i)) + 1 ;
        
        % series type 5: FB85   T2 - NOFB85 T1
    elseif seriestype(pos) == 5 && chosen_task(i) == 1
        task1val(seriestype(pos),LBduree(i)) = task1val(seriestype(pos),LBduree(i)) + 1 ;
    elseif seriestype(pos) == 5 && chosen_task(i) == 2
        task2val(seriestype(pos),LBduree(i)) = task2val(seriestype(pos),LBduree(i)) + 1 ;
        
        % series type 6: NOFB70 T1 - NOFB85 T2
    elseif seriestype(pos) == 6 && chosen_task(i) == 1
        task1val(seriestype(pos),LBduree(i)) = task1val(seriestype(pos),LBduree(i)) + 1 ;
    elseif seriestype(pos) == 6 && chosen_task(i) == 2
        task2val(seriestype(pos),LBduree(i)) = task2val(seriestype(pos),LBduree(i)) + 1 ;
    end
    
end



end
