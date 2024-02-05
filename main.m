% Load the data from the MAT file
load('ELEC3810_Final_project.mat');

% Remove NaN values from the training state data
validIndices = ~isnan(trainState);
validTrainSpike = trainSpike(:, validIndices);
validTrainState = trainState(validIndices);

% Set the sizes and parameters for the neural network
input_layer_size = size(validTrainSpike, 1);
feature_size = size(validTrainSpike, 2);
hidden_layer_size = 2 * input_layer_size;
output_layer_size = 2;
learning_rate = 0.01;
max_epochs = 100;
batch_size = 64;
num_folds = 5;

% Set the random number generator seed for reproducibility
rng(3810);

% Convert the training state to categorical data
validTrainState = categorical(validTrainState);

% Define the history lengths for the model
history_options = [5, 6, 7, 8, 9, 10, 11, 12];

% Initialize arrays to store the best models and accuracies for each fold and history length
bestModels = cell(num_folds, numel(history_options));
bestAccuracies = zeros(num_folds, numel(history_options));

% Perform k-fold cross-validation
for fold = 1:num_folds
    disp(['Running Fold: ' num2str(fold)]);
    
    % Split the data into training and validation sets
    validationIndices = randperm(size(validTrainSpike, 2), round(0.2 * size(validTrainSpike, 2)));
    trainingIndices = setdiff(1:size(validTrainSpike, 2), validationIndices);

    % Train the model for each history length
    for history_index = 1:numel(history_options)
        history = history_options(history_index);

        % Define the layers of the neural network
        layers = [
            sequenceInputLayer(input_layer_size, 'Name', 'inputLayer', 'MinLength', input_layer_size)
            batchNormalizationLayer('Name', 'batch1')
            flattenLayer('Name', 'flatten1')
            fullyConnectedLayer(hidden_layer_size, 'Name', 'hiddenLayer1')
            reluLayer('Name', 'relu1')
            dropoutLayer(0.1, 'Name', 'dropout1')
            fullyConnectedLayer(hidden_layer_size, 'Name', 'hiddenLayer2')
            reluLayer('Name', 'relu2')
            fullyConnectedLayer(hidden_layer_size, 'Name', 'hiddenLayer3')
            reluLayer('Name', 'relu3')
            fullyConnectedLayer(hidden_layer_size, 'Name', 'hiddenLayer4')
            reluLayer('Name', 'relu4')
            fullyConnectedLayer(hidden_layer_size, 'Name', 'hiddenLayer5')
            reluLayer('Name', 'relu5')
            fullyConnectedLayer(hidden_layer_size, 'Name', 'hiddenLayer6')
            reluLayer('Name', 'relu6')
            fullyConnectedLayer(output_layer_size, 'Name', 'outputLayer')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'classOutput')
        ];

        % Define the training options for the neural network
        options = trainingOptions('rmsprop', ...
            'MaxEpochs', max_epochs, ...
            'MiniBatchSize', batch_size, ...
            'InitialLearnRate', learning_rate, ...
            'Verbose', false, ...
            %{
            'ValidationData', {validTrainSpike(:, validationIndices), validTrainState(validationIndices)}, ...
            'ValidationFrequency', 10, ...
            'Plots', 'training-progress', ...
            %}
            'Shuffle', 'every-epoch', ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.01, ...
            'LearnRateDropPeriod', 100, ...
            'L2Regularization', 0.00001, ...
            'ValidationPatience', 10);

        % Train the neural network
        net = trainNetwork(validTrainSpike(:, trainingIndices), validTrainState(trainingIndices), layers, options);

        % Validate the model on the validation set
        validationPred = classify(net, validTrainSpike(:, validationIndices));
        validationAccuracy = sum(validationPred == validTrainState(validationIndices)) / numel(validationPred);

        % Store the best model and accuracy for each fold and history length
        bestModels{fold, history_index} = net;
        bestAccuracies(fold, history_index) = validationAccuracy;

        disp(['Validation Accuracy (Fold ' num2str(fold) ', History ' num2str(history) '): ' num2str(validationAccuracy)]);
    end
end

% Find the best model and its corresponding accuracy
[bestAccuracy, bestFoldIndex] = max(bestAccuracies(:));
[bestFold, bestHistoryIndex] = ind2sub(size(bestAccuracies), bestFoldIndex);
bestModel = bestModels{bestFold, bestHistoryIndex};
bestHistory = history_options(bestHistoryIndex);

% Validate the best model on the training set
trainingPred = classify(bestModel, validTrainSpike(:, trainingIndices));
trainingAccuracy = sum(trainingPred == validTrainState(trainingIndices)) / numel(trainingPred);

% Classify the test spike data using the best model
decodedState = classify(bestModel, testSpike);

% Calculate the confusion matrix for the training data
C = confusionmat(double(validTrainState(trainingIndices)), double(trainingPred));
TP = C(1, 1);    % True Positive
FN = C(1, 2);    % False Negative
FP = C(2, 1);    % False Positive
TN = C(2, 2);    % True Negative                                    

% Calculate sensitivity, specificity, G-Mean, precision, recall, and F1 score
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);
gMean = sqrt(sensitivity * specificity);

precision = TP / (TP + FP);
recall = TP / (TP + FN);
F1 = 2 * (precision * recall) / (precision + recall);

% Display the best training and validation accuracies
disp(['Best Training Accuracy: ' num2str(trainingAccuracy)]);
disp(['Best Validation Accuracy: ' num2str(bestAccuracy)]);

% Display the best history length
disp(['Best History Length: ' num2str(bestHistory)]);

% Calculate and display the G-Mean and F1 score
disp(['G-Mean: ' num2str(gMean)]);
disp(['F1 Score: ' num2str(F1)]);

% Calculate and display the confusion matrix for the training data
disp('Confusion Matrix (Training Data):'); 
disp(C);

% Export decodedState to a MAT file
matFileName = 'result.mat';
save(matFileName, 'decodedState');
disp('Decoded state exported');