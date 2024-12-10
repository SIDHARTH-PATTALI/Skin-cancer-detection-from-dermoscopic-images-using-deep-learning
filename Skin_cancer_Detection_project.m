% Skin Cancer Detection using Deep Learning, Fuzzy C-means, and GUI

%% Transfer Learning Using AlexNet

% Load training and validation data
imds = imageDatastore('/MATLAB Drive/data/train', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

% Visualize sample images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 16);
figure
for i = 1:16
    subplot(4, 4, i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end

% Load pre-trained AlexNet model
net = alexnet;
analyzeNetwork(net);

% Input size for AlexNet
inputSize = net.Layers(1).InputSize;

% Transfer layers
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

% Modify the layers to fit our number of classes
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

% Data augmentation and image pre-processing
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter('RandXReflection', true, ...
    'RandXTranslation', pixelRange, 'RandYTranslation', pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

% Training options
options = trainingOptions('sgdm', 'MiniBatchSize', 10, 'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, 'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, 'ValidationFrequency', 3, ...
    'Verbose', false, 'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(augimdsTrain, layers, options);

% Prediction on validation set
[YPred, scores] = classify(netTransfer, augimdsValidation);
idx = randperm(numel(imdsValidation.Files), 4);

% Display predicted images
figure
for i = 1:4
    subplot(2, 2, i)
    I = readimage(imdsValidation, idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

% Evaluate accuracy
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);

% Confusion Matrix
confMat = confusionmat(YValidation, YPred);
disp(confMat);
figure;
heatmap(categories(YValidation), categories(YValidation), confMat);
title('Confusion Matrix');
xlabel('Predicted Labels');
ylabel('True Labels');

% Save model
save('skin_disease_alxnet.mat', 'netTransfer');

%% Transfer Learning Using ResNet50

% Load ResNet50 and pre-process data
net = resnet50;
analyzeNetwork(net);

inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

% Modify layers for the new task
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

% Train the network
options = trainingOptions('sgdm', 'MiniBatchSize', 10, 'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, 'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, 'ValidationFrequency', 3, ...
    'Verbose', false, 'Plots', 'training-progress');
netTransfer = trainNetwork(augimdsTrain, layers, options);

% Prediction and performance evaluation
[YPred, scores] = classify(netTransfer, augimdsValidation);
accuracy = mean(YPred == YValidation);
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);

% Confusion Matrix for ResNet50
confMat = confusionmat(YValidation, YPred);
disp(confMat);
figure;
heatmap(categories(YValidation), categories(YValidation), confMat);
title('Confusion Matrix for ResNet50');
xlabel('Predicted Labels');
ylabel('True Labels');

% Save model
save('skin_disease_resnet50.mat', 'netTransfer');

%% Fuzzy C-Means Image Segmentation

% Fuzzy C-means clustering for skin cancer detection in images
originalImage = imread('C:\Users\asus\Documents\MATLAB\skin_cancer_2\test\malignant.jpg');
grayImage = rgb2gray(originalImage);
data = double(grayImage(:));

% Apply Fuzzy C-Means clustering
numClusters = 2;  % Background and skin cancer
[centers, U] = fcm(data, numClusters);

% Determine the cluster representing the skin cancer region
[~, maxIndex] = max(U);
clusteredImage = reshape(maxIndex, size(grayImage));

% Create a binary mask for the skin cancer region
skinCancerMask = clusteredImage == 2; % Adjust based on your cluster result
figure;
imshow(skinCancerMask);
title('Segmented Skin Cancer Area using Fuzzy C-Means');

% Save the segmented mask
imwrite(skinCancerMask, 'segmented_skin_cancer_mask.jpg');

%% GUI Application for Skin Cancer Detection

classdef skinCancerApp < matlab.apps.AppBase
    properties (Access = public)
        UIFigure matlab.ui.Figure
        ImportButton matlab.ui.control.Button
        AnalyzeButton matlab.ui.control.Button
        ImageAxes matlab.ui.control.UIAxes
        ResultLabel matlab.ui.control.Label
        ResultField matlab.ui.control.EditField
        ModelFileLabel matlab.ui.control.Label
        ModelFileField matlab.ui.control.EditField
        ClassLabel matlab.ui.control.Label
        ClassField matlab.ui.control.EditField
        ScoreLabel matlab.ui.control.Label
        ScoreField matlab.ui.control.EditField
    end
    
    methods (Access = private)
        
        % Import button callback
        function ImportButtonPushed(app, event)
            global a;
            [filename, folderPath] = uigetfile('.', 'Pick an Image');
            filename = strcat(folderPath, filename);
            fullFilePath = fullfile(folderPath, filename);
            a = imread(filename);
            imshow(a, 'Parent', app.ImageAxes);
        end

        % Analyze button callback
        function AnalyzeButtonPushed(app, event)
            global a;
            
            % Load the trained network model
            loadedData = load('skin_disease_resnet50.mat');
            model = loadedData.netTransfer;

            % Pre-process the image
            resizedImage = imresize(a, [224, 224]);
            [predictedLabel, scores] = classify(model, resizedImage);

            % Display results
            app.ClassField.Value = char(predictedLabel);
            app.ScoreField.Value = num2str(max(scores) * 100);
            app.ModelFileField.Value = 'ResNet50 Model';
        end
        
    end
    
    methods (Access = public)
        
        % Create the app
        function createComponents(app)
            app.UIFigure = uifigure('Position', [100, 100, 500, 400]);
            
            % Import Button
            app.ImportButton = uibutton(app.UIFigure, 'push', ...
                'Position', [50, 350, 100, 30], 'Text', 'Import Image');
            app.ImportButton.ButtonPushedFcn = createCallbackFcn(app, @ImportButtonPushed, true);
            
            % Analyze Button
            app.AnalyzeButton = uibutton(app.UIFigure, 'push', ...
                'Position', [200, 350, 100, 30], 'Text', 'Analyze');
            app.AnalyzeButton.ButtonPushedFcn = createCallbackFcn(app, @AnalyzeButtonPushed, true);
            
            % UI Axes for displaying image
            app.ImageAxes = axes(app.UIFigure, 'Position', [0.1, 0.2, 0.4, 0.5]);
            
            % Result Labels and Fields
            app.ResultLabel = uilabel(app.UIFigure, 'Position', [350, 320, 100, 22], 'Text', 'Result:');
            app.ResultField = uieditfield(app.UIFigure, 'text', 'Position', [400, 320, 80, 22]);
            
            app.ClassLabel = uilabel(app.UIFigure, 'Position', [350, 270, 100, 22], 'Text', 'Class:');
            app.ClassField = uieditfield(app.UIFigure, 'text', 'Position', [400, 270, 80, 22]);
            
            app.ScoreLabel = uilabel(app.UIFigure, 'Position', [350, 220, 100, 22], 'Text', 'Score:');
            app.ScoreField = uieditfield(app.UIFigure, 'text', 'Position', [400, 220, 80, 22]);
            
            app.ModelFileLabel = uilabel(app.UIFigure, 'Position', [350, 170, 100, 22], 'Text', 'Model File:');
            app.ModelFileField = uieditfield(app.UIFigure, 'text', 'Position', [400, 170, 80, 22]);
        end
        
        % Run the app
        function runApp(app)
            createComponents(app);
            app.UIFigure.Visible = 'on';
        end
    end
end
