function SeisPlot(M, fig, scale, varargin)
%Input : 
% M     - 需要绘制的地震波形，m * n，m表示波形道号， n表示时间ms。
% fig   - 是否使用新窗口， 'old' 'new';
% scale - 是否对波形进行缩放;

% Author : Chirl Chen

if ~exist('fig', 'var')
    fig = {'figure', 'new'};
end
if ~exist('scale', 'var')
    scale = {'scale', 'no'};
end
corruptedSlice = squeeze(M)';
seisCorr = s_convert(corruptedSlice, 0, 1);
s_wplot(seisCorr, fig, scale, varargin{:});
end

