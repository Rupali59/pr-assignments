A = load('probability_class_1.txt');
Test = load('testData.txt');
mapping = zeros(200,3);
X = zeros(200,1)
for i = 1:200
    if(A(i, 1) > 0.5)
        mapping(i, :) = [0 1 0];
    else
        mapping(i, :) = [1 0 0];
    end
    X(i) = i
end
mapping;
scatter(X, Test, 15, mapping);
