function [HammingLoss,RankingLoss,OneError,Coverage,AveragePrecision] = test_PML( test_data,test_target,model)

[num_test,~]=size(test_target);
[~,num_class]=size(test_target);


W = model.W;

fea_matrix = test_data';
Outputs = W'*fea_matrix;


Pre_Labels=zeros(num_test,num_class);
outputValue = Outputs';
[outputValue,~] = mapminmax(outputValue,0,1);

for i=1:num_test
    for k=1:num_class
        if(outputValue(i,k)>=0.7)
            Pre_Labels(i,k)=1;
        else
            Pre_Labels(i,k)=0;
        end
    end
end

HammingLoss=Hamming_loss(Pre_Labels',test_target');
RankingLoss=Ranking_loss(outputValue',test_target');
OneError=One_error(outputValue',test_target');
Coverage=coverage(outputValue',test_target');
AveragePrecision=Average_precision(outputValue',test_target');

end

