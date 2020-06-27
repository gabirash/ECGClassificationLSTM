% This script parses the data from the PhysioNet 2017 Challenge and saves
% the data into PhysionetData.mat for quick and easy future use.

% Copyright 2017 The MathWorks, Inc.

% Download and unzip the data, training2017.zip, from the PhysioNet website
% https://physionet.org/challenge/2017/
% unzip('https://archive.physionet.org/challenge/2017/training2017.zip')

% Navigate to the directory
cd Data

% File with filenames and labels
ref = 'REFERENCE.csv';

% Create a table that contains the filenames and corresponding label data
tbl = readtable(ref,'ReadVariableNames',false);
tbl.Properties.VariableNames = {'Filename','Label'};

% Delete 'Other Rhythm' and 'Noisy Recording' signals
%toDelete = strcmp(tbl.Label,'O') | strcmp(tbl.Label,'~');
%tbl(toDelete,:) = [];

% Load each file in the table and store the corresponding signal data
H = height(tbl);
for ii = 1:H
    fileData = load([tbl.Filename{ii},'.mat']);
    tbl.Signal{ii} = fileData.val;
end

% Leave the training2017 directory
cd ..

% Format the data properly for LSTM training
% Signals: Cell array of predictors
% Labels: Categorical array of responses
Signals = tbl.Signal;
Labels = categorical(tbl.Label);

% Navigate to the directory
cd Test

% File with filenames and labels
ref_test = 'REFERENCE_TEST.csv';

% Create a table that contains the filenames and corresponding label data
tbl_test = readtable(ref_test,'ReadVariableNames',false);
tbl_test.Properties.VariableNames = {'Filename','Label'};

% Delete 'Other Rhythm' and 'Noisy Recording' signals
%toDelete = strcmp(tbl.Label,'O') | strcmp(tbl.Label,'~');
%tbl(toDelete,:) = [];

% Load each file in the table and store the corresponding signal data
H_test = height(tbl_test);
for ii = 1:H_test
    fileData = load([tbl_test.Filename{ii},'.mat']);
    tbl_test.Signal{ii} = fileData.val;
end

% Leave the training2017 directory
cd ..

% Format the data properly for LSTM training
% Signals: Cell array of predictors
% Labels: Categorical array of responses
Signals_test = tbl_test.Signal;
Labels_test = categorical(tbl_test.Label);

% Save the variables to a MAT-file
save PhysionetData.mat Signals Labels
save PhysionetTest.mat Signals_test Labels_test