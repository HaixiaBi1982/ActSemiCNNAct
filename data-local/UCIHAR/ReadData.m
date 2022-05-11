clear all;
close all;

RootDir = 'C:\research\HealthCare\Datasets\UCI-HAR\UCI_HAR_Dataset\UCI_HAR_Dataset';

Fs = 100;
T = 1/Fs;                

% 14 participants
AllData = zeros(14*12*5*4000, 9);
currNum = 1;
DataNum = 0;
for par = 1:14
    par
    Dir = strcat(RootDir, num2str(par), '\');
    % 12 activities
    for act = 1:12
        % five trials
        for tr = 1:5
            File = strcat(Dir, 'a', num2str(act), 't', num2str(tr), '.mat');
            load(File);
            LineNum = size(sensor_readings, 1);
            t = (0:LineNum-1)*T;         
            AllData(currNum:currNum+LineNum-1, 1) = t;
            AllData(currNum:currNum+LineNum-1, 2) = DataNum;            
            AllData(currNum:currNum+LineNum-1, 3:8) = sensor_readings;
            AllData(currNum:currNum+LineNum-1, 9) = str2num(activity_number)-1;            
            currNum = currNum+LineNum;     
            DataNum = DataNum + 1;
        end
    end
end

currNum = currNum - 1;
% DataNum = DataNum - 1;

Data = zeros(currNum, 8);
Data = AllData(1:currNum,:);

save Data.mat Data;
save Data.txt -ascii Data;
