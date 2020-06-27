%%Introduction
% ECGs record the electrical activity of a person's heart over a period of time. Physicians use ECGs to detect visually if a patient's heartbeat is normal or irregular.
% Atrial fibrillation (AFib) is a type of irregular heartbeat that occurs when the heart's upper chambers, the atria, beat out of coordination with the lower chambers, the ventricles.
% This example uses ECG data from the PhysioNet 2017 Challenge [1], [2], [3], which is available at https://physionet.org/challenge/2017/. The data consists of a set of ECG signals sampled at 300 Hz and divided by a group of experts into four different classes: Normal (N), AFib (A), Other Rhythm (O), and Noisy Recording (~). This example shows how to automate the classification process using deep learning. The procedure explores a binary classifier that can differentiate Normal ECG signals from signals showing signs of AFib. 
% This example uses long short-term memory (LSTM) networks, a type of recurrent neural network (RNN) well-suited to study sequence and time-series data. An LSTM network can learn long-term dependencies between time steps of a sequence. The LSTM layer (lstmLayer) can look at the time sequence in the forward direction, while the bidirectional LSTM layer (bilstmLayer) can look at the time sequence in both forward and backward directions. This example uses a bidirectional LSTM layer.
% To accelerate the training process, run this example on a machine with a GPU. If your machine has a GPU and Parallel Computing Toolbox™, then MATLAB® automatically uses the GPU for training; otherwise, it uses the CPU. 
% 
 %% 
% Load and Examine the Data
% Run the ReadPhysionetData script to download the data and generate a MAT-file (PhysionetData.mat) that contains the ECG signals in the appropriate format. Downloading the data might take a few minutes.
ReadPhysionetData
load PhysionetData
load PhysionetTest
%% 
% 
%The loading operation adds two variables to the workspace: Signals and Labels. Signals is a cell array that holds the ECG signals. Labels is a categorical array that holds the corresponding ground-truth labels of the signals.
disp('5 first Signals')
Signals(1:5)
disp('5 first Test Signals')
Signals_test(1:5)
disp('5 first Labels')
Labels(1:5)
disp('5 first Test Labels')
Labels_test(1:5)
disp('Summary Labels')
summary(Labels)
disp('Summary Test Labels')
summary(Labels_test)
%% 
% 
%Generate a histogram of signal lengths. Notice that most of the signals are 9000 samples long.
figure
L = cellfun(@length,Signals);
h = histogram(L);
xticks(0:3000:18000);
xticklabels(0:3000:18000);
title('Signal Lengths')
xlabel('Length')
ylabel('Count')
%% 
% 
%Visualize a segment of one signal from N class and one from A class. 
%AFib heartbeats are spaced out at irregular intervals while Normal heartbeats occur regularly. 
%AFib heartbeat signals also often lack a P wave, which pulses before the QRS complex in a Normal heartbeat signal. 
%The plot of the Normal signal shows a P wave and a QRS complex.

normal = Signals{1};
aFib = Signals{4};
figure
subplot(2,1,1)
plot(normal)
title('Normal Rhythm')
xlim([4000,5200])
ylabel('Amplitude (mV)')
text(4330,150,'P','HorizontalAlignment','center')
text(4370,850,'QRS','HorizontalAlignment','center')

subplot(2,1,2)
plot(aFib)
title('Atrial Fibrillation')
xlim([4000,5200])
xlabel('Samples')
ylabel('Amplitude (mV)')
%% 
%Prepare the Data for Training
% During training, the trainNetwork function splits the data into mini-batches. The function then pads or truncates signals in the same mini-batch so they all have the same length. 
% To avoid excessive padding or truncating, apply the segmentSignals function to the ECG signals so they are all 9000 samples long. The function ignores signals with fewer than 9000 samples. If a signal has more than 9000 samples, segmentSignals breaks it into as many 9000-sample segments as possible and ignores the remaining samples. For example, a signal with 18500 samples becomes two 9000-sample signals, and the remaining 500 samples are ignored
[Signals,Labels] = segmentSignals(Signals,Labels);
[Signals_test,Labels_test] = segmentSignals(Signals_test,Labels_test);
%% 
% 
% View the first five elements of the Signals array to verify that each entry is now 9000 samples long.
disp('5 first signals- same length')
Signals(1:5)
disp('5 first test signals- same length')
Signals_test(1:5)
%% 
%Train the Classifier Using Raw Signal Data
%To design the classifier, use the raw signals generated in the previous section. 
%Split the signals into a training set to train the classifier and a validation set to validate the accuracy of the classifier on new data. 
%Use the summary function to show that now there are 4976 Normal signals and: 732 AFib signals with a ratio of aprox. 1:7; 2668 Other signals  with a ratio of aprox. 1:2 and 33 Noise signals with a ratio of aprox. 1:151.  
disp('Summary of signals')
summary(Labels)
disp('Summary of test signals')
summary(Labels_test)
%% 
% 
% Because 59.2% of the signals are Normal, the classifier would learn that it can achieve a good accuracy simply by classifying most signals as Normal. 
%To avoid this bias, augment the AFib data by duplicating AFib signals in the dataset so that there is the same number of Normal, AFib, Other and Noise signals. 
%This duplication, commonly called oversampling, is one form of data augmentation used in deep learning. 
%Split the signals according to their class.
% 

afibX = Signals(Labels=='A');
afibY = Labels(Labels=='A');

normalX = Signals(Labels=='N');
normalY = Labels(Labels=='N');

otherX = Signals(Labels=='O');
otherY = Labels(Labels=='O');

noiseX = Signals(Labels=='~');
noiseY = Labels(Labels=='~');

afibX_test = Signals_test(Labels_test=='A');
afibY_test = Labels_test(Labels_test=='A');

normalX_test = Signals_test(Labels_test=='N');
normalY_test = Labels_test(Labels_test=='N');

otherX_test = Signals_test(Labels_test=='O');
otherY_test = Labels_test(Labels_test=='O');

noiseX_test = Signals_test(Labels_test=='~');
noiseY_test = Labels_test(Labels_test=='~');
%% 
% 
% Next, use dividerand to divide targets from each class randomly into training and validation sets, 90% for training and 10% for validation.
[trainIndA,validIndA,~] = dividerand(732,0.9,0.1,0.0);
[trainIndN,validIndN,~] = dividerand(4976,0.9,0.1,0.0);
[trainIndO,validIndO,~] = dividerand(2668,0.9,0.1,0.0);
[trainIndNO,validIndNO,~] = dividerand(33,0.9,0.1,0.0);

XTrainA = afibX(trainIndA);
YTrainA = afibY(trainIndA);

XTrainN = normalX(trainIndN);
YTrainN = normalY(trainIndN);

XTrainO = otherX(trainIndO);
YTrainO = otherY(trainIndO);

XTrainNO = noiseX(trainIndNO);
YTrainNO = noiseY(trainIndNO);

XValidA = afibX(validIndA);
YValidA = afibY(validIndA);

XValidN = normalX(validIndN);
YValidN = normalY(validIndN);

XValidO = otherX(validIndO);
YValidO = otherY(validIndO);

XValidNO = noiseX(validIndNO);
YValidNO = noiseY(validIndNO);

XTestA = afibX_test;
YTestA = afibY_test;

XTestN = normalX_test;
YTestN = normalY_test;

XTestO = otherX_test;
YTestO = otherY_test;

XTestNO = noiseX_test;
YTestNO = noiseY_test;

%% 
% 
%Now there are 659 AFib signals, 2401 Other signals, 30 Noise signals and 4478 Normal signals for training. 
%To achieve the same number of signals in each class, use the first 4473 Normal signals, and then use repmat to repeat the first 639 AFib signals seven times, the first 2235 Other signals two times and the first 30 Noise signals 149 times. 
%For validation, there are 73 AFib signals, 267 Other signals, 3 Noise signals and 498 Normal signals. 
%Use the first 498 Normal signals, and then use repmat to repeat the first 71 AFib signals seven times, the first 249 Other signals two times and the first 3 Noise signals 166 times. 
%By default, the neural network randomly shuffles the data before training, ensuring that contiguous signals do not all have the same label.

%XTrain = [repmat(XTrainA(1:639),7,1); XTrainN(1:4473); repmat(XTrainO(1:2235),2,1); repmat(XTrainNO(1:30),149,1)];
%YTrain = [repmat(YTrainA(1:639),7,1); YTrainN(1:4473); repmat(YTrainO(1:2235),2,1); repmat(YTrainNO(1:30),149,1)];
XTrain = [repmat(XTrainA(1:639),4,1); XTrainN(1:2236); XTrainO(1:2235); repmat(XTrainNO(1:30),75,1)];
YTrain = [repmat(YTrainA(1:639),4,1); YTrainN(1:2236); YTrainO(1:2235); repmat(YTrainNO(1:30),75,1)];

XValid = [repmat(XValidA(1:71),7,1); XValidN(1:498); repmat(XValidO(1:249),2,1); repmat(XValidNO(1:3),166,1)];
YValid = [repmat(YValidA(1:71),7,1); YValidN(1:498); repmat(YValidO(1:249),2,1); repmat(YValidNO(1:3),166,1)];

XTest = [XTestA; XTestN; XTestO; XTestNO ];
YTest = [YTestA; YTestN; YTestO; YTestNO ];
%% 
% 
%The distribution between Normal and AFib signals is now evenly balanced in both the training set and the validation set. 
disp('Balanced Train')
summary(YTrain)
disp('Balanced Valid')
summary(YValid)
disp('Balanced Test')
summary(YTest)
%% 
% 
%Define the LSTM Network Architecture
%LSTM networks can learn long-term dependencies between time steps of sequence data. 
%This example uses the bidirectional LSTM layer bilstmLayer, as it looks at the sequence in both forward and backward directions.
%Because the input signals have one dimension each, specify the input size to be sequences of size 1. 
%Specify a bidirectional LSTM layer with an output size of 100 and output the last element of the sequence. 
%This command instructs the bidirectional LSTM layer to map the input time series into 100 features and then prepares the output for the fully connected layer. 
%Finally, specify two classes by including a fully connected layer of size 4, followed by a softmax layer and a classification layer. 

% layers = [ ...
%     sequenceInputLayer(1)
%     bilstmLayer(100,'OutputMode','last')
%     fullyConnectedLayer(4)
%     softmaxLayer
%     classificationLayer
%     ]
%% 
% 
%Next specify the training options for the classifier. 
%Set the 'MaxEpochs' to 10 to allow the network to make 10 passes through the training data. 
%A 'MiniBatchSize' of 150 directs the network to look at 150 training signals at a time. 
%An 'InitialLearnRate' of 0.01 helps speed up the training process. 
%Specify a 'SequenceLength' of 1000 to break the signal into smaller pieces so that the machine does not run out of memory by looking at too much data at one time. 
%Set 'GradientThreshold' to 1 to stabilize the training process by preventing gradients from getting too large. 
%Specify 'Plots' as 'training-progress' to generate plots that show a graphic of the training progress as the number of iterations increases. 
%Set 'Verbose' to false to suppress the table output that corresponds to the data shown in the plot. 
%If you want to see this table, set 'Verbose' to true.
%This example uses the adaptive moment estimation (ADAM) solver. 
%ADAM performs better with recurrent neural networks (RNNs) like LSTMs than the default stochastic gradient descent with momentum (SGDM) solver.


%options = trainingOptions('adam', ...
%     'MaxEpochs',10, ...
%     'MiniBatchSize', 150, ...
%     'InitialLearnRate', 0.01, ...
%     'SequenceLength', 1000, ...
%     'GradientThreshold', 1, ...
%     'ExecutionEnvironment',"auto", ...
%     'plots','training-progress', ...
%     'Verbose',false, ...
%     'ValidationData',{XValid,YValid});
%% 
%
%Train the LSTM Network
%
%Train the LSTM network with the specified training options and layer architecture by using trainNetwork. 
%Because the training set is large, the training process can take several minutes.

% net = trainNetwork(XTrain,YTrain,layers,options);
%% 
% 
%% 
%
% Visualize the Training and Testing Accuracy
% Calculate the training accuracy, which represents the accuracy of the classifier on the signals on which it was trained. 
% First, classify the training data. 

% trainPred = classify(net,XTrain,'SequenceLength',1000);
%% 
% 

% LSTMAccuracy = sum(trainPred == YTrain)/numel(YTrain)*100
% 
% figure
% ccLSTM = confusionchart(YTrain,trainPred);
% ccLSTM.Title = 'Confusion Chart for LSTM';
% ccLSTM.ColumnSummary = 'column-normalized';
% ccLSTM.RowSummary = 'row-normalized';
%% 
% 
%% 
% 
%Now classify the validation data with the same network.

% validPred = classify(net,XValid,'SequenceLength',1000);
%% 
% 
%Calculate the testing accuracy and visualize the classification performance as a confusion matrix.

% LSTMAccuracy = sum(validPred == YValid)/numel(YValid)*100
% 
% figure
% ccLSTM = confusionchart(YValid,validPred);
% ccLSTM.Title = 'Confusion Chart for LSTM';
% ccLSTM.ColumnSummary = 'column-normalized';
% ccLSTM.RowSummary = 'row-normalized';
%% 
% 
%% 
%
%Improve the Performance with Feature Extraction
%Feature extraction from the data can help improve the training and testing accuracies of the classifier. 
%To decide which features to extract, this example follows an approach that computes time-frequency images, such as spectrograms, and uses them to train convolutional neural networks (CNNs). 

%Visualize the spectrogram of each type of signal.

fs = 300;

figure
subplot(2,1,1);
pspectrum(normal,fs,'spectrogram','TimeResolution',0.5)
title('Normal Signal')

subplot(2,1,2);
pspectrum(aFib,fs,'spectrogram','TimeResolution',0.5)
title('AFib Signal')
%% 
% 
%Since this example uses an LSTM instead of a CNN, it is important to translate the approach so it applies to one-dimensional signals. 
%Time-frequency (TF) moments extract information from the spectrograms. 
%Each moment can be used as a one-dimensional feature to input to the LSTM. 
%Explore two TF moments in the time domain:
%  Instantaneous frequency (instfreq)
%  Spectral entropy (pentropy)

%% 
% 
%The instfreq function estimates the time-dependent frequency of a signal as the first moment of the power spectrogram. 
%The function computes a spectrogram using short-time Fourier transforms over time windows. 
%In this example, the function uses 255 time windows. 
%The time outputs of the function correspond to the centers of the time windows.
%Visualize the instantaneous frequency for each type of signal.

[instFreqA,tA] = instfreq(aFib,fs);
[instFreqN,tN] = instfreq(normal,fs);

figure
subplot(2,1,1);
plot(tN,instFreqN)
title('Normal Signal')
xlabel('Time (s)')
ylabel('Instantaneous Frequency')

subplot(2,1,2);
plot(tA,instFreqA)
title('AFib Signal')
xlabel('Time (s)')
ylabel('Instantaneous Frequency')
%% 
% 
%Use cellfun to apply the instfreq function to every cell in the training and testing sets. 

instfreqTrain = cellfun(@(x)instfreq(x,fs)',XTrain,'UniformOutput',false);
instfreqValid = cellfun(@(x)instfreq(x,fs)',XValid,'UniformOutput',false);
instfreqTest = cellfun(@(x)instfreq(x,fs)',XTest,'UniformOutput',false);
%% 
% 
%The spectral entropy measures how spiky flat the spectrum of a signal is. 
%A signal with a spiky spectrum, like a sum of sinusoids, has low spectral entropy. 
%A signal with a flat spectrum, like white noise, has high spectral entropy. 
%The pentropy function estimates the spectral entropy based on a power spectrogram. 
%As with the instantaneous frequency estimation case, pentropy uses 255 time windows to compute the spectrogram. 
%The time outputs of the function correspond to the center of the time windows.
%Visualize the spectral entropy for each type of signal.

[pentropyA,tA2] = pentropy(aFib,fs);
[pentropyN,tN2] = pentropy(normal,fs);

figure

subplot(2,1,1)
plot(tN2,pentropyN)
title('Normal Signal')
ylabel('Spectral Entropy')

subplot(2,1,2)
plot(tA2,pentropyA)
title('AFib Signal')
xlabel('Time (s)')
ylabel('Spectral Entropy')
%% 
% 
%Use cellfun to apply the pentropy function to every cell in the training and testing sets. 

pentropyTrain = cellfun(@(x)pentropy(x,fs)',XTrain,'UniformOutput',false);
pentropyValid = cellfun(@(x)pentropy(x,fs)',XValid,'UniformOutput',false);
pentropyTest = cellfun(@(x)pentropy(x,fs)',XTest,'UniformOutput',false);
%% 
% 
%Concatenate the features such that each cell in the new training and testing sets has two dimensions, or two features. 

XTrain2 = cellfun(@(x,y)[x;y],instfreqTrain,pentropyTrain,'UniformOutput',false);
XValid2 = cellfun(@(x,y)[x;y],instfreqValid,pentropyValid,'UniformOutput',false);
XTest2 = cellfun(@(x,y)[x;y],instfreqTest,pentropyTest,'UniformOutput',false);
%% 
% 
%Visualize the format of the new inputs. 
%Each cell no longer contains one 9000-sample-long signal; 
%now it contains two 255-sample-long features.
disp('Train new size')
XTrain2(1:5)
disp('Test new size')
XTest2(1:5)
%% 
%Standardize the Data
%The instantaneous frequency and the spectral entropy have means that differ by almost one order or magnitude. 
%Furthermore, the instantaneous frequency mean might be too high for the LSTM to learn effectively. 
%When a network is fit on data with a large mean and a large range of values, large inputs could slow down the learning and convergence of the network.

mean(instFreqN);
mean(pentropyN);
%% 
% 
%Use the training set mean and standard deviation to standardize the training and testing sets. 
%Standardization, or z-scoring, is a popular way to improve network performance during training.

XV = [XTrain2{:}];
mu = mean(XV,2);
sg = std(XV,[],2);

XTrainSD = XTrain2;
XTrainSD = cellfun(@(x)(x-mu)./sg,XTrainSD,'UniformOutput',false);

XValidSD = XValid2;
XValidSD = cellfun(@(x)(x-mu)./sg,XValidSD,'UniformOutput',false);

XTestSD = XTest2;
XTestSD = cellfun(@(x)(x-mu)./sg,XTestSD,'UniformOutput',false);
%% 
% 
% Show the means of the standardized instantaneous frequency and spectral entropy.
instFreqNSD = XTrainSD{1}(1,:);
pentropyNSD = XTrainSD{1}(2,:);

mean(instFreqNSD);
mean(pentropyNSD);
%% 
%Modify the LSTM Network Architecture
%Now that the signals each have two dimensions, it is necessary to modify the network architecture by specifying the input sequence size as 2. 
%Specify a bidirectional LSTM layer with an output size of 100, and output the last element of the sequence. 
%Specify two classes by including a fully connected layer of size 4, followed by a softmax layer and a classification layer.

layers = [ ...
    sequenceInputLayer(2)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer
    ]
%% 
% 
%Specify the training options. 
%Set the maximum number of epochs to 30 to allow the network to make 30 passes through the training data.

options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false, ...
     'ValidationData',{XValidSD,YValid});
%% 
%Train the LSTM Network with Time-Frequency Features
%Train the LSTM network with the specified training options and layer architecture by using trainNetwork.

net2 = trainNetwork(XTrainSD,YTrain,layers,options);

%% 
%Visualize the Training and Testing Accuracy
%Classify the training data using the updated LSTM network. 
%Visualize the classification performance as a confusion matrix.

trainPred2 = classify(net2,XTrainSD);
disp('Train Accuracy')
LSTMAccuracy = sum(trainPred2 == YTrain)/numel(YTrain)*100

figure
ccLSTM = confusionchart(YTrain,trainPred2);
ccLSTM.Title = ' Train Confusion Chart for LSTM';
ccLSTM.ColumnSummary = 'column-normalized';
ccLSTM.RowSummary = 'row-normalized';
%% 
% 

validPred2 = classify(net2,XValidSD);
disp('Validation Accuracy')
LSTMAccuracy = sum(validPred2 == YValid)/numel(YValid)*100

figure
ccLSTM = confusionchart(YValid,validPred2);
ccLSTM.Title = 'Validation Confusion Chart for LSTM';
ccLSTM.ColumnSummary = 'column-normalized';
ccLSTM.RowSummary = 'row-normalized';

%%
%
testPred2 = classify(net2,XTestSD);
disp('Test Accuracy')
LSTMAccuracy = sum(testPred2 == YTest)/numel(YTest)*100

figure
ccLSTM = confusionchart(YTest,testPred2);
ccLSTM.Title = 'Test Confusion Chart for LSTM';
ccLSTM.ColumnSummary = 'column-normalized';
ccLSTM.RowSummary = 'row-normalized';
%% 
%% 
%