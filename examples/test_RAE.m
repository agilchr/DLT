function test_RAE(num_epochs, mask_fracs)
    
    if ~exist('num_epochs', 'var') || ~(numel(num_epochs) == 2)
        num_epochs = [3 5];
    end
    
    if ~exist('mask_fracs', 'var')
        mask_fracs = [.25 .5];
    end
    
    load mnist_uint8;

    train_x = double(train_x)/255;
    test_x  = double(test_x)/255;
    train_y = double(train_y);
    test_y  = double(test_y);

    %%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
    %  Setup and train a stacked denoising autoencoder (SDAE)
    rand('state',0)
    ae_opts.activation_function       = 'sigm';
    ae_opts.learningRate              = 1;
    ae_opts.lossFunction              = 'crossEnt';
    rae_opts.aeOpts = ae_opts;
    rae_opts.dims = [784 100];
    rae_opts.maskFractions = mask_fracs;
    rae = raesetup(rae_opts);
    opts.numepochs = num_epochs(1);
    opts.batchsize = 100;
    rae = raetrain(rae, train_x, opts);
    for i = 1 : numel(mask_fracs)
        visualize(rae.ae{i}{1}.W{1}(:,2:end)')
        title(['Weights of ' num2str(mask_fracs(i)) ' deleted']);
    end

    fprintf(['Training neural network with weights initialized from ' ...
             'RAE\n']);
    % Use the RAE to initialize a FFNN
    nn = nnsetup([784 100*numel(mask_fracs) 10]);
    nn.activation_function              = 'sigm';
    nn.lossFunction                     = 'crossEnt';
    nn.learningRate                     = 1;
    nn.W{1} = [];
    for i = 1 : numel(mask_fracs)
        nn.W{1} = [nn.W{1}; rae.ae{i}{1}.W{1}];
    end

    % Train the FFNN
    opts.numepochs = num_epochs(2);
    opts.batchsize = 100;
    nn = nntrain(nn, train_x, train_y, opts);
    [er, bad] = nntest(nn, test_x, test_y);
    fprintf('Testing error: %.4f%%\n', er*100);
    assert(er < 0.16, 'Too big error');
end
