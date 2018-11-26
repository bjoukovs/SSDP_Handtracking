clear all;
close all;

meas = fileread('KALMANnormal/log_meas_2018-11-26 22_19_01_655293.csv');
meas = clean(meas);
meas = reshape(meas,3,length(meas)/3);

prob = fileread('KALMANnormal/log_p_2018-11-26 22_19_01_655293.csv');
prob = clean(prob);
prob = reshape(prob,3,length(prob)/3);

s0 = fileread('KALMANnormal/log_states0_2018-11-26 22_19_01_655293.csv');
s0 = clean(s0);
s0 = reshape(s0,2,length(s0)/2);

s1 = fileread('KALMANnormal/log_states1_2018-11-26 22_19_01_655293.csv');
s1 = clean(s1);
s1 = reshape(s1,4,length(s1)/4);

s2 = fileread('KALMANnormal/log_states2_2018-11-26 22_19_01_655293.csv');
s2 = clean(s2);
s2 = reshape(s2,5,length(s2)/5);

Q0 = fileread('KALMANnormal/log_covar0_2018-11-26 22_19_01_655293.csv');
Q0 = clean(Q0);

Q1 = fileread('KALMANnormal/log_covar1_2018-11-26 22_19_01_655293.csv');
Q1 = clean(Q1);

Q2 = fileread('KALMANnormal/log_covar2_2018-11-26 22_19_01_655293.csv');
Q2 = clean(Q2);

s = {s0, s1, s2};
Q = {Q0 Q1 Q2};
sz = [2 4 5];


%%%%%%%%%%%%%%%
t0 = 20;
tf = 183;

h = plot(prob(:,t0:tf)');
set(h, {'color'}, {[1 0 0]; [0 1 0]; [0 0 1]});

[pmax pmax_i] = max(prob, [], 1);

figure;
plot(meas(1,t0:tf), meas(2,t0:tf), 's')
hold on;

mstates = [];
mcolors = [];
mcov = [];
for i=t0:tf
    n = pmax_i(i);
   mstates = [mstates, s{n}(1:2,i)];
   mcov = [mcov; Q{n}(i*sz(n):i*sz(n)+1, 1:2)];
   mcolors = [mcolors; n];
end

plot(mstates(1,:), mstates(2,:), 'color', 'black')
gscatter(mstates(1,:), mstates(2,:), mcolors)

figure;
idx = 1:2:(tf-t0)*2;
plot(mcov(idx,1));
hold on
plot(mcov(idx+1,2));

