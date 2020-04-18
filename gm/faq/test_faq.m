addpath('./../../../ElasticGraphs/src')

n = 5;prob = 0.5;
A1 = createBinomialGraph(n,prob);
p = randperm(n);
A2 = A1(p,p);

pp = sfw(-A2,A1,30);

fprintf('ground truth: ')
disp(p)
fprintf('faq: ')
disp(pp)