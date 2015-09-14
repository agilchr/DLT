function rae = raetrain(rae, x, opts)
    input_copy = x;
    for i = 1 : numel(rae.ae);
        % grab the original input, not the result of the last run
        % with the previous masking fraction
        x = input_copy;
        for j = 1 : numel(rae.ae{i})
            disp(['Training AE ' num2str(j) '/' num2str(numel(rae.ae{i})) ...
                  ' with masking percent ' num2str(rae.ae{i}{j}.inputZeroMaskedFraction)]);
            rae.ae{i}{j} = nntrain(rae.ae{i}{j}, x, x, opts);
            t = nnff(rae.ae{i}{j}, x, x);
            x = t.a{2};
            %remove bias term
            x = x(:,2:end);
        end
    end
end
