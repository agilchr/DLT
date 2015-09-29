function net = cnnsetup(net, x, y)
    assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    sx = size(x);
    if numel(sx) == 3 % black and white images
        inputmaps = 1;
        mapsize = size(squeeze(x(:, :, 1)));        
    else % color images
        inputmaps = sx(3);
        mapsize = size(squeeze(x(:, :, 1, 1)));
    end
    
    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's')
            mapsize = mapsize / net.layers{l}.scale;
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps
                net.layers{l}.b{j} = 0;
            end
        end
        if strcmp(net.layers{l}.type, 'r') % ReLU layer
            net.layers{l}.scale = 1;
            net.layers{l}.outputmaps = net.layers{l-1}.outputmaps;
            for j = 1 : inputmaps
                net.layers{l}.b{j} = 0;
            end
        end
        if strcmp(net.layers{l}.type, 'c')
            net.layers{l}.scale = 1;
            if isfield(net.layers{l}, 'padded') && net.layers{l}.padded
                net.layers{l}.convtype = 'full';
                mapsize = mapsize + net.layers{l}.kernelsize - 1;
            else
                net.layers{l}.convtype = 'valid';
                mapsize = mapsize - net.layers{l}.kernelsize + 1;
            end
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  %  input map
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                net.layers{l}.b{j} = 0;
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);

    net.ffb = zeros(onum, 1);
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end
