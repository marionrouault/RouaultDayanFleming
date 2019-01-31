% Forming global estimates of self-performance from local confidence
% Rouault M., Dayan P. and Fleming S. M. Nature Communications (2019)
% Experiment 2 (N=29).
% Hierarchical learning model simulations

% Normative model determines the trial-by-trial choice and confidence
% (p_success) that are embedded into a learning model updating task
% ability beliefs modelled as beta distributions.

% Parameters:
% kch: individual perception of dots difference between easy and
% difficult conditions (not fitted)
% kconf: same for reading out confidence (fitted)



function [chosen_task] = BAY_learning_model(accuracylevel,fbtype, ...
    currenttask,target_left,seriestype,LBduration,kch,kconf)


% dot difference for the two difficulty levels
easy = 60 ;
hard = 24 ;

% internal state by design
d = target_left ;
d(target_left == 1) = -1 ; % left
d(target_left == 0) = 1  ; % right

% nb of blocks - task choices
N_series = length(seriestype) ;

% over how many block trajectories do we average
nb_iter = 1000 ;

% allocate perceptual action, perceptual correctness and chosen task
a           = zeros(sum(LBduration),1);
correct     = zeros(sum(LBduration),1);
chosen_task = zeros(N_series,1) ;
pWh         = zeros(N_series,2) ; % column1 task1, column2 task2

% self-beliefs for both tasks in a block
esperance_T1 = zeros(nb_iter,sum(LBduration));
esperance_T2 = zeros(nb_iter,sum(LBduration));

LBdur = [0, LBduration] ; % just for sampling indexes easier


% initialisation beta distribution parameters for self-beliefs:
alpha0  = 6 ;
beta0   = 3 ;
alphaT1 = alpha0 ;
betaT1  = beta0 ;
alphaT2 = alpha0 ;
betaT2  = beta0 ;



for ser = 1:N_series % loop over blocks in chronological order as participants did
    
    
    Q_iterred = zeros(nb_iter,4) ;
    
    for iter = 1:nb_iter % inner loop marginalizing over instances (trajectories) inside the block
        
        
        % initialised at each iter, not saved:
        X         = zeros(LBduration(ser),1);
        p_success = zeros(LBduration(ser),1);
        
        
        for t = 1:LBduration(ser)
            
            
            % ///////// perceptual module providing choice and p_success (conf) at each trial /////////
            
            
            if     accuracylevel(sum(LBdur(1:ser))+t)==70
                dhat = [-1*kconf*hard 1*kconf*hard];
            elseif accuracylevel(sum(LBdur(1:ser))+t)==85
                dhat = [-1*kconf*easy 1*kconf*easy];
            end
            
            
            % the model does perceptual choice
            
            % For X, sigma = 1 because no variance in dot difference
            if     accuracylevel(sum(LBdur(1:ser))+t)==70
                X(t) = normrnd(d(sum(LBdur(1:ser))+t)*kch*hard, 1) ;
            elseif accuracylevel(sum(LBdur(1:ser))+t)==85
                X(t) = normrnd(d(sum(LBdur(1:ser))+t)*kch*easy, 1) ;
            end
            
            % choose action according to X:
            if     X(t) < 0
                a(sum(LBdur(1:ser))+t) = 1 ; % choose left
            elseif X(t) > 0
                a(sum(LBdur(1:ser))+t) = 0 ; % choose right
            end
            
            % "correct" gives the feedback the model obtains at each trial:
            if     a(sum(LBdur(1:ser))+t) ~= target_left(sum(LBdur(1:ser))+t)
                correct(sum(LBdur(1:ser))+t) = 0 ; % incorrect
            elseif a(sum(LBdur(1:ser))+t) == target_left(sum(LBdur(1:ser))+t)
                correct(sum(LBdur(1:ser))+t) = 1 ; % correct
            end
            
            
            
            % Compute confidence at each trial from sample X
            
            if sign(X(t)) == 1 % chose a = 1 right
                p_success(t) = normpdf(X(t),dhat(2),1) / (normpdf(X(t),dhat(1),1)+normpdf(X(t),dhat(2),1)) ;
            else               % chose a = -1 left
                p_success(t) = 1 - (normpdf(X(t),dhat(2),1) / (normpdf(X(t),dhat(1),1)+normpdf(X(t),dhat(2),1))) ;
            end
            
            % ///////// end of perceptual module /////////
            
            
            
            
            
            
            % ///////// learning module about the 2 tasks /////////
            
            % beta distrib params for cued task are updated:
            if t==1
                alphaT1 = alpha0 ;
                betaT1  = beta0 ;
                alphaT2 = alpha0 ;
                betaT2  = beta0 ;
            else
                if fbtype(sum(LBdur(1:ser))+t)==1
                    if     currenttask(sum(LBdur(1:ser))+t) == 1
                        if correct(sum(LBdur(1:ser))+t)==1
                            alphaT1 = alphaT1 + 1 ;
                        elseif correct(sum(LBdur(1:ser))+t)==0
                            betaT1  = betaT1 + 1 ;
                        end
                    elseif currenttask(sum(LBdur(1:ser))+t) == 2
                        if correct(sum(LBdur(1:ser))+t)==1
                            alphaT2 = alphaT2 + 1 ;
                        elseif correct(sum(LBdur(1:ser))+t)==0
                            betaT2  = betaT2 + 1 ;
                        end
                    end
                    
                elseif fbtype(sum(LBdur(1:ser))+t)==0
                    if currenttask(sum(LBdur(1:ser))+t) == 1
                        alphaT1 = alphaT1 + p_success(t) ;
                        betaT1  = betaT1 + (1-p_success(t)) ;
                    elseif currenttask(sum(LBdur(1:ser))+t) == 2
                        alphaT2 = alphaT2 + p_success(t) ;
                        betaT2  = betaT2 + (1-p_success(t)) ;
                    end
                    
                end
            end
            % ///////// end of learning module about the 2 tasks /////////
            
            esperance_T1(iter,sum(LBdur(1:ser))+t) = alphaT1/(alphaT1+betaT1) ;
            esperance_T2(iter,sum(LBdur(1:ser))+t) = alphaT2/(alphaT2+betaT2) ;
            
        end
        
        
        % store last belief Q of the block for each iteration
        Q_iterred(iter,:) = [alphaT1 betaT1 alphaT2 betaT2] ;
        
    end
    
    
    
    
    % ///////// Task choice at the end of the block /////////
    
    mean_Q_iterred = mean(Q_iterred) ;
    
    % average over iterations over blocks
    alphaT1 = mean_Q_iterred(1) ;
    betaT1  = mean_Q_iterred(2) ;
    alphaT2 = mean_Q_iterred(3) ;
    betaT2  = mean_Q_iterred(4) ;
    
    
    % compare posterior beliefs of the two tasks
    BeliefT1 = betarnd(alphaT1,betaT1,100000,1);
    BeliefT2 = betarnd(alphaT2,betaT2,100000,1);
    
    deltaBelief = BeliefT1-BeliefT2 ;
    proba_T1 = sum(deltaBelief > 0)/length(deltaBelief) ;
    proba_T2 = sum(deltaBelief < 0)/length(deltaBelief) ;
    
    
    pWh(ser,:) = [proba_T1 proba_T2] ;
    
    pWh(ser,:) = pWh(ser,:)/sum(pWh(ser,:)) ;
    
    
    % choose probabilistically according to pWh(ser,:)
    if rand(1,1) < pWh(ser,1)
        chosen_task(ser) = 1 ;
    else
        chosen_task(ser) = 2 ;
    end
    
    
end



end