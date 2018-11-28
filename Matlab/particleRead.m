clear all;
close all;

% meas = fileread('PARTICLEnormal/log_meas_2018-11-27 00_00_11_535777.csv');
% meas = clean(meas);
% meas = reshape(meas,3,length(meas)/3);
% 
% prob = fileread('PARTICLEnormal/log_p_2018-11-27 00_00_11_535777.csv');
% prob = clean(prob);
% prob = reshape(prob,3,length(prob)/3);
% 
% s0 = fileread('PARTICLEnormal/log_means0_2018-11-27 00_00_11_535777.csv');
% s0 = clean(s0);
% s0 = reshape(s0,2,length(s0)/2);
% 
% s1 = fileread('PARTICLEnormal/log_means1_2018-11-27 00_00_11_535777.csv');
% s1 = clean(s1);
% s1 = reshape(s1,4,length(s1)/4);
% 
% s2 = fileread('PARTICLEnormal/log_means2_2018-11-27 00_00_11_535777.csv');
% s2 = clean(s2);
% s2 = reshape(s2,5,length(s2)/5);

% meas = fileread('PARTICLEhighn/log_meas_2018-11-27 18_34_43_188070.csv');
% meas = clean(meas);
% meas = reshape(meas,3,length(meas)/3);
% 
% prob = fileread('PARTICLEhighn/log_p_2018-11-27 18_34_43_188070.csv');
% prob = clean(prob);
% prob = reshape(prob,3,length(prob)/3);
% 
% s0 = fileread('PARTICLEhighn/log_means0_2018-11-27 18_34_43_188070.csv');
% s0 = clean(s0);
% s0 = reshape(s0,2,length(s0)/2);
% 
% s1 = fileread('PARTICLEhighn/log_means1_2018-11-27 18_34_43_188070.csv');
% s1 = clean(s1);
% s1 = reshape(s1,4,length(s1)/4);
% 
% s2 = fileread('PARTICLEhighn/log_means2_2018-11-27 18_34_43_188070.csv');
% s2 = clean(s2);
% s2 = reshape(s2,5,length(s2)/5);

% meas = fileread('PARTICLEhighqhighn/log_meas_2018-11-27 18_39_49_785217.csv');
% meas = clean(meas);
% meas = reshape(meas,3,length(meas)/3);
% 
% prob = fileread('PARTICLEhighqhighn/log_p_2018-11-27 18_39_49_785217.csv');
% prob = clean(prob);
% prob = reshape(prob,3,length(prob)/3);
% 
% s0 = fileread('PARTICLEhighqhighn/log_means0_2018-11-27 18_39_49_785217.csv');
% s0 = clean(s0);
% s0 = reshape(s0,2,length(s0)/2);
% 
% s1 = fileread('PARTICLEhighqhighn/log_means1_2018-11-27 18_39_49_785217.csv');
% s1 = clean(s1);
% s1 = reshape(s1,4,length(s1)/4);
% 
% s2 = fileread('PARTICLEhighqhighn/log_means2_2018-11-27 18_39_49_785217.csv');
% s2 = clean(s2);
% s2 = reshape(s2,5,length(s2)/5);

% meas = fileread('PARTICLE3600/log_meas_2018-11-27 18_42_45_682633.csv');
% meas = clean(meas);
% meas = reshape(meas,3,length(meas)/3);
% 
% prob = fileread('PARTICLE3600/log_p_2018-11-27 18_42_45_682633.csv');
% prob = clean(prob);
% prob = reshape(prob,3,length(prob)/3);
% 
% s0 = fileread('PARTICLE3600/log_means0_2018-11-27 18_42_45_682633.csv');
% s0 = clean(s0);
% s0 = reshape(s0,2,length(s0)/2);
% 
% s1 = fileread('PARTICLE3600/log_means1_2018-11-27 18_42_45_682633.csv');
% s1 = clean(s1);
% s1 = reshape(s1,4,length(s1)/4);
% 
% s2 = fileread('PARTICLE3600/log_means2_2018-11-27 18_42_45_682633.csv');
% s2 = clean(s2);
% s2 = reshape(s2,5,length(s2)/5);

s = {s0, s1, s2};
sz = [2 4 5];


%%%%%%%%%%%%%%%
t0 = 10;
tf=160;

h = plot(prob(:,t0:tf)');
set(h, {'color'}, {[1 0 0]; [0 1 0]; [0 0 1]});
title("Model probabilities")
xlabel("Samples")
ylabel("Probabilities")
legend;

[pmax pmax_i] = max(prob, [], 1);

figure;
plot(meas(1,t0:tf), meas(2,t0:tf), 's')
hold on;

mstates = [];
mcolors = [];

for i=t0:tf
    n = pmax_i(i);
   mstates = [mstates, s{n}(1:2,i)];
   mcolors = [mcolors; n];
end

plot(mstates(1,:), mstates(2,:), 'color', 'black')
gscatter(mstates(1,:), mstates(2,:), mcolors)
axis equal
title("Position measurements and filter output")
xlabel("x (px)")
ylabel("y (px)")
legend("Measurements", "", "Model 1", "Model 2", "Model 3");

