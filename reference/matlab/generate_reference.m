function generate_reference(output_path)
%GENERATE_REFERENCE Deterministic LTE Toolbox OFDM cross-check fixture.
if nargin < 1
    output_path = fullfile('artifacts', 'matlab_reference', 'lte_toolbox_reference.mat');
end
rng(20260922, 'twister');
mods = {'QPSK', '16QAM', '64QAM'};

target_bits = randi([0 1], 300, 14, 2);
b0 = double(target_bits(:,:,1)); b1 = double(target_bits(:,:,2));
target_grid = ((1-2*b0) + 1i*(1-2*b1))/sqrt(2);

interference_grid = complex(zeros(1200,14));
for block = 1:4
    modulation = mods{mod(block-1,3)+1};
    width = 2 * block;
    if block == 4, width = 2; modulation = 'QPSK'; end
    bits = randi([0 1], 300, 14, width);
    b0 = double(bits(:,:,1)); b1 = double(bits(:,:,2));
    if strcmp(modulation, 'QPSK')
        symbols = ((1-2*b0) + 1i*(1-2*b1))/sqrt(2);
    elseif strcmp(modulation, '16QAM')
        b2 = double(bits(:,:,3)); b3 = double(bits(:,:,4));
        symbols = ((1-2*b0).*(2-(1-2*b2)) + 1i*(1-2*b1).*(2-(1-2*b3)))/sqrt(10);
    else
        b2 = double(bits(:,:,3)); b3 = double(bits(:,:,4));
        b4 = double(bits(:,:,5)); b5 = double(bits(:,:,6));
        symbols = ((1-2*b0).*(4-(1-2*b2).*(2-(1-2*b4))) + ...
            1i*(1-2*b1).*(4-(1-2*b3).*(2-(1-2*b5))))/sqrt(42);
    end
    rows = (block-1)*300 + (1:300);
    interference_grid(rows,:) = symbols;
end

target_enb.NDLRB = 25; target_enb.CyclicPrefix = 'Normal';
target_enb.Nfft = 2048; target_enb.Windowing = 0;
interference_enb.NDLRB = 100; interference_enb.CyclicPrefix = 'Normal';
interference_enb.Nfft = 2048; interference_enb.Windowing = 0;
[target_waveform, target_info] = lteOFDMModulate(target_enb, target_grid, 0, 2048);
[interference_waveform, interference_info] = lteOFDMModulate(interference_enb, interference_grid, 0, 2048);
matlab_version = version;
toolbox = ver('lte'); lte_toolbox_version = toolbox.Version;
parent = fileparts(output_path); if ~exist(parent, 'dir'), mkdir(parent); end
save(output_path, 'target_grid', 'interference_grid', 'target_waveform', ...
    'interference_waveform', 'target_info', 'interference_info', ...
    'matlab_version', 'lte_toolbox_version', '-v7');
end
