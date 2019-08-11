function []=detect_keypoints(impath, kptpath, maxkpts, net_name)
    %% Detect the features

    net = dagnn.DagNN.loadobj(load(fullfile('nets', net_name)));

    % Uncomment the following lines to compute on a GPU
    % (works only if MatConvNet compiled with GPU support)
    % gpuDevice(1);  net.move('gpu');

    detector = DDet(net, 'thr', 4);
    im = imread(impath);
    [frames, ~, info] = detector.detect(im); 
    [~, order] = sort(info.peakScores);
    order = fliplr(order);
    [~, num] = size(order);

    cutoff = maxkpts;
    if num < cutoff
        cutoff = num;
    end
    
    A = [frames(1,order(1:cutoff));frames(2,order(1:cutoff))];

    %% Plot the results
    writematrix(A.',kptpath);

end
