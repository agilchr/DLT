function CNN_cifar_gray(num_epochs)
    
    if ~exist('num_epochs','var')
        num_epochs = 1;
    end
    
    train_x = [];
    train_y = [];
    for batch = 1 : 5 % reduce to one batch for quick functional tests
        fprintf('Converting batch %d\n',batch);
        load(['data_batch_',num2str(batch)]); % gives us data and labels
        new_x = reshape(double(data'), 32, 32, 3, size(data,1));
        mx = mean(new_x,3);
        train_x = cat(4, train_x, cat(3, mx, cat(3, mx, mx)));
        % train_x = cat(3, train_x, squeeze(mean(new_x,3)));
        train_y = cat(2, train_y, convert_labels(labels));
    end
    fprintf('Converting test batch\n');
    load('test_batch'); % also gives us data and labels
    test_x = reshape(data', 32, 32, 3, size(data,1));
    % test_x = squeeze(mean(test_x, 3));
    mx = mean(test_x, 3);
    test_x = cat(3, mx, cat(3, mx, mx));
    size(train_x)
    size(test_x)
    test_y = convert_labels(labels);
        

    %% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
    %will run 1 epoch in about 200 second and get around 11% error. 
    %With 100 epochs you'll get around 1.2% error
    rand('state',0)
    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 7) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 4) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 4) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
                 };
    cnn = cnnsetup(cnn, train_x, train_y);

    opts.alpha = 1;
    opts.batchsize = 50;
    opts.numepochs = num_epochs;

    cnn = cnntrain(cnn, train_x, train_y, opts);

    [er, bad] = cnntest(cnn, test_x, test_y);
    
    % let's look at the cnn
    cnn

    %plot mean squared error
    figure; plot(cnn.rL);
    
    fprintf('Testing Error: %4f\n', er);

end

function one_hot = convert_labels(labels)
    one_hot = zeros(10, numel(labels));
    fprintf('           ');
    for i = 1 : numel(labels)
        fprintf('\b\b\b\b\b\b\b\b\b\b\b%5d/%5d', i, 10000);
        one_hot(labels(i) + 1, i) = 1; % +1 to convert from 0-base to 1-base
    end
    fprintf('\n');
end