close all; clear all; clc
warning off;
addpath(genpath('ClusteringMeasure'));
addpath(genpath('function'));
ResSavePath = 'Res/';
MaxResSavePath = 'maxRes/';
SetParas;
for dataIndex = 10:length(datasetName) - (length(datasetName) - 10)
    dataName = [dataPath datasetName{dataIndex} '.mat'];
    load(dataName, 'KH', 'Y');
    ResBest = zeros(1, 8);
    ResStd = zeros(1, 8);
    %% Data Preparation
    c = max(Y);
    V = size(KH, 3);
    N = size(KH, 1);
    KH = kcenter(KH);
    KH = knorm(KH);
    gamma0 = ones(V,1)/V;
    avgKer = mycombFun(KH,gamma0);
    %% parameters setting
    r1 = -4:2:4;
    r2 = -4:2:4;
    r3 = 0.01:0.01:0.09;
    acc = zeros(length(r1), length(r2), length(r3));
    nmi = zeros(length(r1), length(r2), length(r3));
    purity = zeros(length(r1), length(r2), length(r3));
    idx = 1;
    for r1Index = 1:length(r1)
        r1Temp = r1(r1Index);
        for r2Index = 1:length(r2)
            r2Temp = r2(r2Index);
            for r3Index = 1:length(r3)
                r3Temp = r3(r3Index);
                tic;
                % initialize the dimension of each kernel partition
                evs = zeros(V,N);
                for i = 1:V
                    evs(i,:) = eigs(KH(:,:,i),N)';
                end
                esr = zeros(V,N);
                for i = 1:V
                    for j = 1:N
                        temp = evs(i,:);
                        esr(i,j) = sum(temp(1:j))/sum(evs(i,:));
                    end
                end
                esry = zeros(V,N);
                for i = 1:V
                    for j = 2:N
                        temp = esr(i,:);
                        esry(i,j) = (temp(j) - temp(j-1))/temp(j-1);
                    end
                end
                indexx = zeros(1,V);
                for i = 1:V
                    for tempi = 1:(N-1)
                        temp = esry(i,:);
                        if (abs((temp(tempi)-temp(tempi+1))/temp(tempi))<r3Temp)
                            break;
                        end
                    end
                    indexx(i) = tempi;
                end
                % initialize the dimension of centroids
                evsavg = eigs(avgKer,N);
                esravg = zeros(1,N);
                for i = 1:N
                    esravg(i) = sum(evsavg(1:i))/sum(evsavg);
                end
                esryavg = zeros(1,N);
                for i = 2:N
                    esryavg(i) = (esravg(i)-esravg(i-1))/esravg(i-1);
                end
                for indexxavg = 1:(N-1)
                    if (abs((esryavg(indexxavg)-esryavg(indexxavg+1))/esryavg(indexxavg))<r3Temp)
                        break;
                    end
                end
%                 indexxavg = min(indexx);  
                % initialize centroids
                [Favg] = mykernelkmeans(avgKer, indexxavg);
                [~,C] = kmeans(Favg, c);
                time1 = toc;    
                % Main algorithm
                fprintf('Please wait a few minutes\n');
                disp(['Dataset: ', datasetName{dataIndex}, ...
                ', --r1--: ', num2str(r1Temp), ', --r2--: ', num2str(r2Temp), ', --r3--: ', num2str(r3Temp)]);
                F = cell(V,1);
                for i = 1:V
                    [F{i}] = mykernelkmeans(KH(:, :, i), indexx(i));
                end
                [Fstar, gamma, obj, Z] = ERMKC(F, c, 5.^r1Temp, 2.^r2Temp, N, V, indexx, indexxavg, Favg); 
                time2 = toc;
                tic;
                [res] = myNMIACC(real(Fstar), Y, c);
                time3 = toc;
                Runtime(idx) = time1 + time2 + time3/20;
                disp(['runtime: ', num2str(Runtime(idx))]);
                idx = idx + 1;
                tempResBest(1, :) = res(1, :);
                tempResStd(1, :) = res(2, :);
            
                acc(r1Index, r2Index, r3Index) = tempResBest(1, 7);
                nmi(r1Index, r2Index, r3Index) = tempResBest(1, 4);
                purity(r1Index, r2Index, r3Index) = tempResBest(1, 8);
            
                resFile = [ResSavePath datasetName{dataIndex}, '-ACC=', num2str(tempResBest(1, 7)), ...
                '-r1=', num2str(r1Temp), '-r2=', num2str(r2Temp), '.mat'];
                save(resFile, 'tempResBest', 'tempResStd', 'F');
            
                for tempIndex = 1:8
                    if tempResBest(1, tempIndex) > ResBest(1, tempIndex)
                        if tempIndex == 7
                            newF = Fstar;
                            newgamma = gamma;
                            newobj = obj;
                            newZ = Z;
                        end
                        ResBest(1, tempIndex) = tempResBest(1, tempIndex);
                        ResStd(1, tempIndex) = tempResStd(1, tempIndex);
                    end
                end
            end
        end
    end
    aRuntime = mean(Runtime);
    resFile2 = [MaxResSavePath datasetName{dataIndex}, '-ACC=', num2str(ResBest(1, 7)), '.mat'];
    save(resFile2, 'ResBest', 'ResStd', 'acc', 'nmi', 'purity', 'aRuntime', 'newF', 'newgamma', 'newobj', 'Y', 'newZ');
end

