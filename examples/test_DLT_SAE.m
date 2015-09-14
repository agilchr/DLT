function test_DLT_SAE(num_epochs, mask_frac)
    
    if ~exist('num_epochs', 'var')
        num_epochs = 5; % 0.1
    end
    
    if ~exist('mask_frac', 'var') || mask_frac < 0 || ...
            mask_frac > 1
        mask_frac = 0.2; % 0.5;
    end
    
    load mnist_uint8;

    train_x = double(train_x)/255;
    test_x  = double(test_x)/255;
    train_y = double(train_y);
    test_y  = double(test_y);

    %%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
    %  Setup and train a stacked denoising autoencoder (SDAE)
    rand('state',0)
    sae = saesetup([784 100]);
    sae.ae{1}.activation_function       = 'sigm';
    sae.ae{1}.learningRate              = 1;
    sae.ae{1}.inputZeroMaskedFraction   = mask_frac;
    sae.ae{1}.lossFunction              = 'crossEnt';
    opts.numepochs = num_epochs;
    opts.batchsize = 100;
    sae = saetrain(sae, train_x, opts);
    visualize(sae.ae{1}.W{1}(:,2:end)')

    % Use the SDAE to initialize a FFNN
    nn = nnsetup([784 100 10]);
    nn.activation_function              = 'sigm';
    nn.lossFunction                     = 'crossEnt';
    nn.learningRate                     = 1;
    nn.W{1} = sae.ae{1}.W{1};

    % Train the FFNN
    opts.numepochs = num_epochs;
    opts.batchsize = 100;
    nn = nntrain(nn, train_x, train_y, opts);
    [er, bad] = nntest(nn, test_x, test_y);
    fprintf('Testing error: %.4f%%\n', er*100);
    assert(er < 0.16, 'Too big error');
end
