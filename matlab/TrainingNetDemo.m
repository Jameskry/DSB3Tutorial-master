rootFolder = fullfile('/VISL2_net/talandhaim/File_Sorting', 'images');
categories = {'dicomcancer', 'dicomnocancer'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames','FileExtensions','.dcm' );
tbl = countEachLabel(imds);
%minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
minSetCount = 500;


% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds)
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
layers = [imageInputLayer([32 32]);
          convolution2dLayer(5,20);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(2);
          softmaxLayer();
          classificationLayer()];
options = trainingOptions('sgdm','MaxEpochs',25,'MiniBatchSize',300,...
	'InitialLearnRate',0.0005,'LearnRateSchedule','piecewise');
%options = trainingOptions('sgdm','MaxEpochs',25,'MiniBatchSize',300,...
%	'InitialLearnRate',0.0005,'LearnRateSchedule','piecewise','OutputFcn',@plotTrainingAccuracy);
convnet = trainNetwork(trainingSet,layers,options);
[YTest, err] = classify(convnet,testSet);
TTest = testSet.Labels;
%accuracy = LogLoss
%% Turning the TTest (Test Labels) to an array of double (probality) struct data


TTest_ = grp2idx(TTest);   %this case - 'cancer' is '1' and 'non cancer' is '2'
TTest_prob = [];
for i=1:(length(TTest_))  
    if TTest_(i)==1
        TTest_prob(i) = 1;
    else
        TTest_prob(i) = 0;
    end 
end

TTest_prob=TTest_prob';
        
%%

res=[];

err=err(:,1);
res=TTest_prob.*log10(err(:))+(1-TTest_prob).*log10(1-err(:));
LogLoss = -(1/length(res)).*sum(res)



        
        
    

