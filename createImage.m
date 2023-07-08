
%% 创建训练集
clearvars;
% =========================================================================
% SIMULATION
% =========================================================================

% load the initial pressure distribution from an image and scale

PML_size = 20;              % size of the PML in grid points
Nx = 128 - 2 * PML_size;    % number of grid points in the x direction
Ny = 128 - 2 * PML_size;    % number of grid points in the y direction
x = 10e-3;                  % total grid size [m]
y = 10e-3;                  % total grid size [m]
dx = x / Nx;                % grid point spacing in the x direction [m]
dy = y / Ny;                % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);


p0 = zeros(Nx,Ny);
p0(1,1) = 1;
% smooth the initial pressure distribution and restore the magnitude
p0 = smooth(p0, true);

% assign to the source structure
source.p0 = p0;

% define the properties of the propagation medium
medium.sound_speed = 1500;  % [m/s]

% define a centered Cartesian circular sensor
sensor_radius = 4.9e-3;     % [m]
sensor_angle = 2*pi;      % [rad]
sensor_pos = [0, 0];        % [m]
num_sensor_points = 64;
cart_sensor_mask = makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle);
% assign to sensor structure
mask = cart2grid(kgrid,cart_sensor_mask);
sensor.mask = mask;

% create the time array
kgrid.makeTime(medium.sound_speed);

% set the input options
input_args = {'Smooth', false, 'PMLInside', false, 'PlotPML', false, 'CreateLog', false};

% run the simulation
sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});

tstart = tic;
k = 0;
for i = 2:Nx-1
    for j = 2:Ny-1
        k = k+1;
        p0 = zeros(Nx,Ny);
        p0(i,j) = 1;
        % smooth the initial pressure distribution and restore the magnitude
        p0 = smooth(p0, true);
        save("./P0/P"+sprintf('%04d',k),'p0')

        % assign to the source structure
        source.p0 = p0;

        % run the simulation
        sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});
        save("./L0/L"+sprintf('%04d',k),'sensor_data')
        
        str = sprintf('第%d次迭代',k);
        disp('**********************')
        disp(str);
        disp('**********************')

    end

end
tend=toc(tstart);


%% 创建测试集

clearvars;
% =========================================================================
% SIMULATION
% =========================================================================

% load the initial pressure distribution from an image and scale

PML_size = 20;              % size of the PML in grid points
Nx = 128 - 2 * PML_size;    % number of grid points in the x direction
Ny = 128 - 2 * PML_size;    % number of grid points in the y direction
x = 10e-3;                  % total grid size [m]
y = 10e-3;                  % total grid size [m]
dx = x / Nx;                % grid point spacing in the x direction [m]
dy = y / Ny;                % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium
medium.sound_speed = 1500;  % [m/s]

% define a centered Cartesian circular sensor
sensor_radius = 4.9e-3;     % [m]
sensor_angle = 2*pi;      % [rad]
sensor_pos = [0, 0];        % [m]
num_sensor_points = 64;
cart_sensor_mask = makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle);
% assign to sensor structure
mask = cart2grid(kgrid,cart_sensor_mask);
sensor.mask = mask;

% create the time array
kgrid.makeTime(medium.sound_speed);

% set the input options
input_args = {'Smooth', false, 'PMLInside', false, 'PlotPML', false, 'CreateLog', false};

tstart = tic;
folder = './TestPitch/CHASEDB1/';  % 文件夹路径
filePattern = fullfile(folder, '*.png');
pngFiles = dir(filePattern);

for i = 1:length(filePattern)
    filename = fullfile(folder, pngFiles(i).name);
    img = imread(filename);
    p0 = imresize(img,[Nx, Ny]);

    % assign to the source structure
    p0 = 5*smooth(double(p0));
    save(folder+"../P0/P"+sprintf('%04d',i),'p0')

    % assign to the source structure
    source.p0 = p0;
    
    % run the simulation
    sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});
    save(folder+"../L0/L"+sprintf('%04d',i),'sensor_data')
    
    str = sprintf('第%d次迭代',i);
    disp('**********************')
    disp(str);
    disp('**********************')
%     break
end


tend=toc(tstart);