function cnn = CNN_cifar(num_epochs)
    
    if ~exist('num_epochs','var')
        num_epochs = 1;
    end
    
    if exist('cifar_train_data.mat','file')
        load('cifar_train_data');
    else
        train_x = [];
        train_y = [];
        for batch = 1 : 5 % reduce to one batch for quick functional tests
            fprintf('Converting batch %d\n',batch);
            load(['data_batch_',num2str(batch)]); % gives us data and labels
            new_x = reshape(double(data'), 32, 32, 3, size(data,1));
            train_x = cat(4, train_x, new_x);
            train_y = cat(2, train_y, convert_labels(labels));
        end
        save('cifar_train_data','train_x','train_y');
    end
    
    if exist('cifar_test_data.mat','file')
        load('cifar_test_data');
    else
        fprintf('Converting test batch\n');
        load('test_batch'); % also gives us data and labels
        test_x = reshape(data', 32, 32, 3, size(data,1));
        test_y = convert_labels(labels);
        save('cifar_test_data', 'test_x', 'test_y');
    end        
    
        

    %% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
    %will run 1 epoch in about 200 second and get around 11% error. 
    %With 100 epochs you'll get around 1.2% error
    rand('state',0)
    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 15, 'kernelsize', 5, 'padded', true) %convolution layer
        struct('type', 'r') %ReLU layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 10, 'kernelsize', 5, 'padded', false) %convolution layer
        struct('type', 'r')
        struct('type', 's', 'scale', 2) %subsampling layer
        struct('type', 'c', 'outputmaps', 10, 'kernelsize', 4, 'padded', false) ...
    %convolution layer
        struct('type', 'r')
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
    
    visualize(cnn.layers(2).a);

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