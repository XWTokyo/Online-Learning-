% SOINN - Self-Organizing Incremental Neural Network
% ver. 2.8.0
%
% Copyright (c) 2013 Yoshihiro Nakamura
% This software is released under the MIT License.
% http://opensource.org/licenses/mit-license.php

classdef SOINN < handle

    properties
        % Signal's dimension.
        dimension;
        % A period deleting nodes. The nodes that doesn't satisfy some
        % condition are deleted every this period.
        deleteNodePeriod;
        % The maximum of edges' ages. If an edge's age is more than this,
        % the edge is deleted.
        maxEdgeAge;
        % The minimum of degrees. If the degree of a node is less than
        % this, the node is considered as a noise and deleted.
        minDegree = 2;
        % The number of inputed signals
        signalNum;
        % The matrix whose rows correspoind with signals of nodes.
        nodes;
        % The winning times of nodes in competitive learning
        winningTimes;
        % the matrix expressing which nodes are connected and each edge age.
        adjacencyMat;
    end

    methods
        function obj = SOINN(varargin)
            % constractor
            % Optoins:
            % 'lambda' (default: 300)
            %       A period deleting nodes. The nodes that doesn't satisfy
            %       some condition are deleted every this period.
            % 'ageMax' (default: 50)
            %       The maximum of edges' ages. If an edge's age is more
            %       than this, the edge is deleted.
            % 'dim' (default: 2)
            %       signal's dimension
            o = OptionHandler(varargin);
            obj.deleteNodePeriod = o.get('lambda', 300);
            obj.maxEdgeAge = o.get('ageMax', 50);
            obj.dimension = o.get('dim', 2);
            obj.nodes = [];
            obj.winningTimes = [];
            obj.adjacencyMat = sparse([]);
            obj.signalNum = 0;
        end

        function inputSignal(obj, signal)
            % @param {row vector} signal - new input signal
            signal = obj.checkSignal(signal);
            obj.signalNum = obj.signalNum + 1;

            if size(obj.nodes, 1) < 3
                obj.addNode(signal);
                return;
            end

            [winner, dists] = obj.findNearestNodes(2,signal);
            simThresholds = obj.calculateSimiralityThresholds(winner);
            if any(dists > simThresholds)
                obj.addNode(signal);
            else
                obj.addEdge(winner);
                obj.incrementEdgeAges(winner(1));
                winner(1) = obj.deleteOldEdges(winner(1));
                obj.updateWinner(winner(1), signal);
                obj.updateAdjacentNodes(winner(1), signal);
            end

            if mod(obj.signalNum, obj.deleteNodePeriod) == 0
                obj.deleteNoiseNodes();
            end
        end

        function show(obj, varargin)
            % Display SOINN's network in 2D.
            % Options:
            % 'winningTimes' a flag to show winning time of each node
            % 'dim' which dimensions to show (default: [1, 2])
            % 'data' data showed with KDESOINN status (default: [])
            % 'cluster' cluster labels which is a row vector outputed by clustring()
            o = OptionHandler(varargin);
            dims = o.get('dim', [1 2]);
            winningTimeFlag = o.exist('winnCLCingTimes');
            data = o.get('data', []);
            hold on;
            %show data
            if size(data, 1) > 0
                plot(data(:,dims(1)), data(:,dims(2)), 'xb');
            end
            % show edges
            for j = 1:size(obj.adjacencyMat,2)
                for k = j:size(obj.adjacencyMat, 1)
                    if obj.adjacencyMat(k,j) > 0
                        nk = obj.nodes(k,:);
                        nj = obj.nodes(j,:);
                        plot([nk(dims(1)), nj(dims(1))], [nk(dims(2)), nj(dims(2))], 'k');
                    end
                end
            end
             clusterLabels = o.get('cluster', obj.clustering(2));
            if ~isempty(clusterLabels)
                clusterNum = max(clusterLabels);
                
                colors = obj.getRgbVectors(clusterNum+1);
                
                idx = clusterLabels == -1;
                
                plot(obj.nodes(idx, dims(1)), obj.nodes(idx, dims(2)), '.', 'Markersize', 20, 'MarkerEdgeColor', colors(1,:));
             
                for i = 1:clusterNum
                    idx = clusterLabels == i;
                    
                if isequal( colors(i+1, :), [1 1 1])
                        plot(obj.nodes(idx, dims(1)), obj.nodes(idx, dims(2)), '.', 'Markersize', 20, 'MarkerEdgeColor', [0.3333    1.0000    0.6667]);
                else
                        plot(obj.nodes(idx, dims(1)), obj.nodes(idx, dims(2)), '.', 'Markersize', 20, 'MarkerEdgeColor', colors(i+1, :));
                end
                end
            else
                plot(obj.nodes(:,dims(1)), obj.nodes(:,dims(2)), '.r','Markersize',20);
            end
            if winningTimeFlag
                delta = (max(max(obj.nodes(:,dims))) - min(min(obj.nodes(:,dims)))) * 0.005;
                for i = 1:size(obj.nodes,1)
                    text(obj.nodes(i,dims(1)) + delta, obj.nodes(i,dims(2)) + delta, num2str(obj.winningTimes(i)), 'BackgroundColor', [1,1,1]);
                end
            end
            set(gca,'XGrid','on','YGrid','on');
            hold off
        end

        function labels = clustering(obj, minClusterSize)
            % get cluster labels of nodes
            % @pram {int} minClusterSize
            %   The miminum of cluster size.
            %   Clusers whose size is smaller than minClusterSize, they are cnosidered as noise.
            %   Nodes belonging to these clusters are assinged -1 labels.
            % return
            %   Cluster labels of nodes which is a row vector.
            n = size(obj.nodes, 1);
            labels = zeros(1, n);
            currentClusterLabel = 1;
            for i = 1:n
                if labels(i) == 0
                    queue = i;
                    [labels , clusterSize] = obj.labelWithBreadthFirstSearch(labels, queue, currentClusterLabel);
                    if clusterSize < minClusterSize
                        labels(labels==currentClusterLabel) = -1;
                    else
                        currentClusterLabel = currentClusterLabel + 1;
                    end
                end
            end
        end
    end

    methods(Hidden=true)
        function rgbs = getRgbVectors(obj, num)
            rgbs = [];
            maxBase = ceil(power(num, 1/3));
            for base = 2:maxBase
                candi = zeros(base^3, 3);
                for i = 0:base^3-1
                    candi(i+1,:) = obj.str2numArry(dec2base(i,base,3));
                end
                candi = candi / (base-1);
                for i = 1:size(candi, 1)
                    tf = true;
                    for k = 1:size(rgbs, 1)
                        if all(candi(i,:) == rgbs(k,:))
                            tf = false;
                            break;
                        end
                    end
                    if tf
                        rgbs = cat(1, rgbs, candi(i, :));
                    end
                end
            end
            rgbs = rgbs(1:num, :);
        end

        function arry = str2numArry(~, str)
            n = length(str);
            arry = zeros(1, n);
            for i = 1:n
                arry(i) = str2num(str(i));
            end
        end

        function [labels, num] = labelWithBreadthFirstSearch(obj, labels, queue, currentClusterLabel)
            num = 0;
            while ~isempty(queue)
                idx = queue(1);
                queue(1) = [];
                if labels(idx) == 0
                    num = num + 1;
                    labels(idx) = currentClusterLabel;
                    queue = cat(2, queue, find(obj.adjacencyMat(idx, :) > 0));
                end
            end
        end

        function signal = checkSignal(obj, signal)
            s = size(signal);
            if s(1) == 1
                if s(2) == obj.dimension
                    return;
                else
                    error('Soinn:checkSignal:dimError', 'The dimension of input signal is not valid.');
                end
            else
                if s(2) == 1
                    signal = obj.checkSignal(signal');
                else
                    error('Soinn:checkSignal:notVector', 'Input signal have to be a vector.');
                end
            end
        end

        function addNode(obj, signal)
            num = size(obj.nodes, 1);
            obj.nodes(num+1,:) = signal;
            obj.winningTimes(num+1) = 1;
            if num == 0
                obj.adjacencyMat(1,1) = 0;
            else
                obj.adjacencyMat(num+1,:) = zeros(1, num);
                obj.adjacencyMat(:,num+1) = zeros(num+1, 1);
            end
        end

        function [indexes, sqDists] = findNearestNodes(obj, num, signal)
            indexes = zeros(num, 1);
            sqDists = zeros(num, 1);
            D = sum(((obj.nodes - repmat(signal, size(obj.nodes, 1), 1)).^2), 2);
            for i = 1:num
                [sqDists(i), indexes(i)] = min(D);
                D(indexes(i)) = inf;
            end
        end

        function simThresholds = calculateSimiralityThresholds(obj, nodeIndexes)
            simThresholds = zeros(length(nodeIndexes), 1);
            for i = 1: length(nodeIndexes)
                simThresholds(i) = obj.calculateSimiralityThreshold(nodeIndexes(i));
            end
        end

        function threshold = calculateSimiralityThreshold(obj, nodeIndex)
            if any(obj.adjacencyMat(:,nodeIndex))
                pals = obj.nodes(obj.adjacencyMat(:,nodeIndex) > 0,:);
                D = sum(((pals - repmat(obj.nodes(nodeIndex,:), size(pals, 1), 1)).^2), 2);
                threshold = max(D);
            else
                [~, sqDists] = obj.findNearestNodes(2, obj.nodes(nodeIndex, :));
                threshold = sqDists(2);
            end
        end

        function addEdge(obj, nodeIndexes)
            obj.adjacencyMat(nodeIndexes(1), nodeIndexes(2)) = 1;
            obj.adjacencyMat(nodeIndexes(2), nodeIndexes(1)) = 1;
        end

        function updateWinner(obj, winnerIndex, signal)
            % @param {int} winnerIndex - hte index of winner
            % @param {row vector} signal - inputted new signal
            obj.winningTimes(winnerIndex) = obj.winningTimes(winnerIndex) + 1;
            w = obj.nodes(winnerIndex,:);
            obj.nodes(winnerIndex, :) = w + (signal - w)./obj.winningTimes(winnerIndex);
        end

        function updateAdjacentNodes(obj, winnerIndex, signal)
            pals = find(obj.adjacencyMat(:,winnerIndex) > 0);
            for i = 1:length(pals)
                w = obj.nodes(pals(i),:);
                obj.nodes(pals(i), :) = w + (signal - w)./(100 * obj.winningTimes(pals(i)));
            end
        end

        function incrementEdgeAges(obj, winnerIndex)
            indexes = find(obj.adjacencyMat(:,winnerIndex) > 0);
            for i = 1: length(indexes)
                obj.incrementEdgeAge(winnerIndex, indexes(i));
            end
        end

        function incrementEdgeAge(obj, i, j)
            obj.adjacencyMat(i, j) = obj.adjacencyMat(i, j) + 1;
            obj.adjacencyMat(j, i) = obj.adjacencyMat(j, i) + 1;
        end

        function setEdgeAge(obj, i, j, value)
            obj.adjacencyMat(i, j) = value;
            obj.adjacencyMat(j, i) = value;
        end

        function winnerIndex = deleteOldEdges(obj, winnerIndex)
            indexes = find(obj.adjacencyMat(:,winnerIndex) > obj.maxEdgeAge + 1); % 1 expresses that there is an edge.
            deletedNodeIndexes = [];
            for i = 1: length(indexes)
                obj.setEdgeAge(indexes(i), winnerIndex, 0);
                if ~any(obj.adjacencyMat(:,indexes(i)))
                    deletedNodeIndexes = cat(1, deletedNodeIndexes, indexes(i));
                end
            end
            winnerIndex = winnerIndex - sum(deletedNodeIndexes < winnerIndex);
            obj.deleteNodes(deletedNodeIndexes);
        end

        function deleteNodes(obj, indexes)
            obj.nodes(indexes,:) = [];
            obj.winningTimes(indexes) = [];
            obj.adjacencyMat(indexes, :) = [];
            obj.adjacencyMat(:, indexes) = [];
        end

        function deleteNoiseNodes(obj)
            noises = sum(obj.adjacencyMat > 0) < obj.minDegree;
            obj.deleteNodes(noises);
        end
    end
end
