clear all;
close all;

meas = fileread('PARTICLEnormal/log_meas_2018-11-27 00_00_11_535777.csv');
meas = clean(meas);
meas = reshape(meas,3,length(meas)/3);

prob = fileread('PARTICLEnormal/log_p_2018-11-27 00_00_11_535777.csv');
prob = clean(prob);
prob = reshape(prob,3,length(prob)/3);

s0 = fileread('PARTICLEnormal/log_means0_2018-11-27 00_00_11_535777.csv');
s0 = clean(s0);
s0 = reshape(s0,2,length(s0)/2);

s1 = fileread('PARTICLEnormal/log_means1_2018-11-27 00_00_11_535777.csv');
s1 = clean(s1);
s1 = reshape(s1,4,length(s1)/4);

s2 = fileread('PARTICLEnormal/log_means2_2018-11-27 00_00_11_535777.csv');
s2 = clean(s2);
s2 = reshape(s2,5,length(s2)/5);

s = {s0, s1, s2};
sz = [2 4 5];


%%%%%%%%%%%%%%%
t0 = 60;
tf = 180;

h = plot(prob(:,t0:tf)');
set(h, {'color'}, {[1 0 0]; [0 1 0]; [0 0 1]});

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

