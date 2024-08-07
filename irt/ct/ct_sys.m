 function [sys, f] = ct_sys(varargin)
% function [sys, f] = ct_sys([options])
% This routine returns the system geometry parameters
% for a fan-beam x-ray CT system, by default the GE LightSpeed system.
% The GE-specific parameters are available online, e.g.:
% http://www.hinnovation.com/ourcompany/radiology.pdf
% http://www.pasa.nhs.uk/evaluation/docs/diagnostic_imaging_evaluation/Report_05015.pdf
% Caller can override any of the parameters using named options.

% defaults
f.type = 'lightspeed';
f.nx = 512;
f.ny = [];
f.pixel_size = 500 / 512; % = 0.9765625 = fov/nx
f.ray_spacing = 1.0;    % Approximate detector pitch
f.strip_width = [];
f.support = 'all';
f.center_x = 0;
f.center_y = 0;
f.orbit = 360;
f.orbit_start = 0;
f.dis_src_det = 949.;   % Approximate
f.dis_iso_det = 408.;   % Approximate
f.channel_offset = 1.25;    % fraction of a channel
f.source_offset = 0;
f.flip_x = 1;       % 1 means no flip
f.flip_y = 1;       % 1 means no flip
f.scale = 0;        % transmission units.  leave untouched.

f.nb = 888; % detector channels
f.na = 984; % angular samples

f = vararg_pair(f, varargin);

% more defaults
if isempty(f.ny); f.ny = f.nx; end
if isempty(f.strip_width); f.strip_width = f.ray_spacing; end

if ~streq(f.type, 'lightspeed'); error 'type not implemented'; end

sys = arg_pair('system', 14, ...
    'nx', f.nx, 'ny', f.ny, ...
    'nb', f.nb, 'na', f.na, ...
    'support',  f.support, ...
    'pixel_size',   f.pixel_size, ...
    'ray_spacing',  f.ray_spacing, ...
    'strip_width',  f.strip_width, ...
    'orbit',    f.orbit, ...
    'orbit_start',  f.orbit_start, ...
    'src_det_dis',  f.dis_src_det, ...
    'obj2det_x',    f.dis_iso_det, ...
    'obj2det_y',    f.dis_iso_det, ...
    'center_x', f.center_x, ...
    'center_y', f.center_y, ...
    'channel_offset', f.channel_offset, ...
    'source_offset', f.source_offset, ...
    'flip_x', f.flip_x, ...
    'flip_y', f.flip_y, ...
    'scale', f.scale);
