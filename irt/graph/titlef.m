 function h = titlef(varargin)
%function h = titlef(varargin)
%|
%| version of title with built-in sprintf
%| also supports default font size from ir_fontsize()

if nargin < 1, help(mfilename), fail(mfilename), end

opt = {'fontsize', ir_fontsize('title')};
%opt = {opt{:}, 'fontname', 'times'};
if ir_is_octave && 1
	opt = {opt{:}, 'fontname', 'Helvetica'};
end

varargin{1} = ir_strrep_tex(varargin{1}); % trick

if ir_is_octave
	tex = {};
	str = varargin{1};
	str = strrep(str, '$', '');
	varargin{1} = str;
else
	tex = {'interpreter', 'latex'};
	str = varargin{1};
	str = strrep(str, '\', '\\'); % because of sprintf below
	% trick to replace "|" with "$|$" in strings
	if ~isempty(strfind(str, '|')) && isempty(strfind(str, '$'))
		str = strrep(str, '|', '$|$');
	end
	varargin{1} = str;
end


if isfreemat
	for ii=1:length(varargin)
		if streq(varargin{ii}, 'interpreter') % not supported by freemat
			varargin{ii} = {};
			varargin{ii+1} = {};
		end
	end
end

if im
	tmp = sprintf(varargin{:});
	hh = title(tmp, tex{:}, opt{:});
else
	hh = [];
end

if nargout
	h = hh;
end
