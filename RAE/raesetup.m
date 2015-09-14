function rae = raesetup(raeOpts)
    optFields = fieldnames(raeOpts.aeOpts);
    for maskIndex = 1 : numel(raeOpts.maskFractions)
        rae.ae{maskIndex} = {};
        dims = raeOpts.dims;
        for u = 2 : numel(dims)
            % create an autoencoder for this layer
            rae.ae{maskIndex}{u-1} = nnsetup([dims(u-1) dims(u) dims(u-1)]);
            
            % assign it the appropriate masking fraction
            rae.ae{maskIndex}{u-1}.inputZeroMaskedFraction = ...
                raeOpts.maskFractions(maskIndex);
            
            % Copy the option fields into the new nn
            for fieldIndex = 1 : numel(optFields)
                rae.ae{maskIndex}{u-1}.(optFields{fieldIndex}) = ...
                    raeOpts.aeOpts.(optFields{fieldIndex});
            end
        end
    end
end
