# Python 3.9 compatible version of wfm2read (complete, fixed to read full waveform)
import os
import struct
import numpy as np
import sys
import array
import matplotlib.pyplot as plt

def rd_unpack(f, format_str, byteorder):
    return struct.unpack(byteorder+format_str, f.read(struct.calcsize(format_str)))[0]

def rd_unpack_a(f, num_bytes, format_str, byteorder):
    return [rd_unpack(f, format_str, byteorder) for _ in range(num_bytes)]

def get_edim(fid, byteorder, wfm_version):
    dim = {}
    dim['dim_scale'] = rd_unpack(fid,'d',byteorder)
    dim['dim_offset'] = rd_unpack(fid,'d',byteorder)
    dim['dim_size'] = rd_unpack(fid,'I',byteorder)
    dim['units'] = rd_unpack(fid, '20s', byteorder).decode(errors='ignore').split('\x00')[0]
    dim['dim_extent_min'] = rd_unpack(fid,'d',byteorder)
    dim['dim_extent_max'] = rd_unpack(fid,'d',byteorder)
    dim['dim_resolution'] = rd_unpack(fid,'d',byteorder)
    dim['dim_ref_point'] = rd_unpack(fid,'d',byteorder)
    dim['format'] = rd_unpack_a(fid,4,'b',byteorder)
    dim['storage_type'] = rd_unpack_a(fid,4,'b',byteorder)
    dim['n_value'] = rd_unpack(fid,'i',byteorder)
    dim['over_range'] = rd_unpack(fid,'i',byteorder)
    dim['under_range'] = rd_unpack(fid,'i',byteorder)
    dim['high_range'] = rd_unpack(fid,'i',byteorder)
    dim['low_range'] = rd_unpack(fid,'i',byteorder)
    dim['user_scale'] = rd_unpack(fid,'d',byteorder)
    dim['user_units'] = rd_unpack(fid,'20s',byteorder).decode(errors='ignore').split('\x00')[0]
    dim['user_offset'] = rd_unpack(fid,'d',byteorder)
    dim['point_density'] = rd_unpack(fid,'d',byteorder) if wfm_version >= 3 else rd_unpack(fid,'I',byteorder)
    dim['href'] = rd_unpack(fid,'d',byteorder)
    dim['trig_delay'] = rd_unpack(fid,'d',byteorder)
    return dim

def get_idim(fid, byteorder, wfm_version):
    dim = {}
    dim['dim_scale'] = rd_unpack(fid,'d',byteorder)
    dim['dim_offset'] = rd_unpack(fid,'d',byteorder)
    dim['dim_size'] = rd_unpack(fid,'I',byteorder)
    dim['units'] = rd_unpack(fid, '20s', byteorder).decode(errors='ignore').split('\x00')[0]
    dim['dim_extent_min'] = rd_unpack(fid,'d',byteorder)
    dim['dim_extent_max'] = rd_unpack(fid,'d',byteorder)
    dim['dim_resolution'] = rd_unpack(fid,'d',byteorder)
    dim['dim_ref_point'] = rd_unpack(fid,'d',byteorder)
    dim['spacing'] = rd_unpack(fid,'I',byteorder)
    dim['user_scale'] = rd_unpack(fid,'d',byteorder)
    dim['user_units'] = rd_unpack(fid,'20s',byteorder).decode(errors='ignore').split('\x00')[0]
    dim['user_offset'] = rd_unpack(fid,'d',byteorder)
    dim['point_density'] = rd_unpack(fid,'d',byteorder) if wfm_version >= 3 else rd_unpack(fid,'I',byteorder)
    dim['href'] = rd_unpack(fid,'d',byteorder)
    dim['trig_delay'] = rd_unpack(fid,'d',byteorder)
    return dim

def wfm2read(filename, datapoints=None, step=1, startind=0, verbose=False):
    if not filename.endswith('.wfm') or not os.path.exists(filename):
        raise FileNotFoundError(f"Invalid file name: {filename}")

    with open(filename, 'rb') as fid:
        info = {}
        info['byte_order_verification'] = '{:04X}'.format(struct.unpack('H', fid.read(2))[0])
        byteorder = '<' if info['byte_order_verification'] == '0F0F' else '>'
        version_str = struct.unpack(byteorder + '8s', fid.read(8))[0].decode(errors='ignore')
        wfm_version = int(version_str.split('#')[1])

        info['versioning_number'] = version_str
        info['num_digits_in_byte_count'] = rd_unpack(fid,'B',byteorder)
        info['num_bytes_to_EOF'] = rd_unpack(fid,'i',byteorder)
        info['num_bytes_per_point'] = rd_unpack(fid,'B',byteorder)
        info['byte_offset_to_beginning_of_curve_buffer'] = rd_unpack(fid,'I',byteorder)
        info['horizontal_zoom_scale_factor'] = rd_unpack(fid,'i',byteorder)
        info['horizontal_zoom_position'] = rd_unpack(fid,'f',byteorder)
        info['vertical_zoom_scale_factor'] = rd_unpack(fid,'d',byteorder)
        info['vertical_zoom_position'] = rd_unpack(fid,'f',byteorder)
        info['waveform_label'] = rd_unpack(fid,'32s',byteorder).decode(errors='ignore').split('\x00')[0]
        info['N'] = rd_unpack(fid,'I',byteorder)
        info['size_of_waveform_header'] = rd_unpack(fid,'H',byteorder)

        fid.read(4)  # setType
        fid.read(4)  # wfmCnt
        fid.read(24)
        fid.read(4)
        fid.read(4)
        fid.read(4)
        fid.read(4)
        fid.read(16)
        fid.read(4)
        fid.read(4)
        fid.read(4)

        if wfm_version >= 2:
            fid.read(2)  # summary_frame_type

        fid.read(4)
        fid.read(8)

        info['ed1'] = get_edim(fid, byteorder, wfm_version)
        info['ed2'] = get_edim(fid, byteorder, wfm_version)
        info['id1'] = get_idim(fid, byteorder, wfm_version)
        info['id2'] = get_idim(fid, byteorder, wfm_version)

        fid.read(4)  # tb1 spacing
        fid.read(4)
        fid.read(4)
        fid.read(4)  # tb2 spacing
        fid.read(4)
        fid.read(4)

        info['real_point_offset'] = rd_unpack(fid,'I',byteorder)
        info['tt_offset'] = rd_unpack(fid,'d',byteorder)
        info['frac_sec'] = rd_unpack(fid,'d',byteorder)
        info['GMT_sec'] = rd_unpack(fid,'i',byteorder)
        info['state_flags'] = rd_unpack(fid,'I',byteorder)
        fid.read(4)
        fid.read(2)
        info['precharge_start_offset'] = rd_unpack(fid,'I',byteorder)
        info['data_start_offset'] = rd_unpack(fid,'I',byteorder)
        info['postcharge_start_offset'] = rd_unpack(fid,'I',byteorder)
        fid.read(4)
        info['end_of_curve_buffer_offset'] = rd_unpack(fid,'I',byteorder)

        data_format_map = {
            0: 'h', 1: 'i', 2: 'I', 3: 'Q',
            4: 'f', 5: 'd', 6: 'B', 7: 'b'
        }
        dfmt = info['ed1']['format'][0]
        if dfmt not in data_format_map or (dfmt in [6, 7] and wfm_version < 3):
            raise ValueError("Invalid data format or file version")

        data_format = data_format_map[dfmt]

        offset = info['byte_offset_to_beginning_of_curve_buffer'] + \
                 info['data_start_offset'] + \
                 startind * info['num_bytes_per_point']
        fid.seek(offset)

        nop_all = (info['postcharge_start_offset'] - info['data_start_offset']) // info['num_bytes_per_point']
        nop = nop_all - startind
        Nframes = info['N']
        pts_per_frame = nop_all if Nframes > 0 else nop_all

        # print(f'len V = {nop_all}, Nframes = {Nframes}, pts_per_frame = {pts_per_frame}')
        if  Nframes > 1:
            # interpret fractional step as "read all frames"
            datapoints = nop_all * Nframes
        # if datapoints is None:
        #     datapoints = int(nop // step)
        # else:
        #     if datapoints > nop:
        #         datapoints = int(nop // step)

        if verbose:
            print(f"Reading {datapoints} data points from {filename} starting at index {startind} with step {step}")
        values = array.array(data_format)
        values.frombytes(fid.read(struct.calcsize(data_format) * datapoints))
        values = np.array(values)

        t = info['id1']['dim_offset'] + info['id1']['dim_scale'] * np.arange(startind, startind + datapoints * step, step)
        y = info['ed1']['dim_offset'] + info['ed1']['dim_scale'] * values
        
        if Nframes > 1:
            points_per_frame = datapoints // Nframes
            # print(f"Reshaping V to ({Nframes}, {points_per_frame})")
            y = y.reshape((Nframes, -1))
            t = t[:points_per_frame]

        ind_over = np.where(values == info['ed1']['over_range'])[0]
        ind_under = np.where(values <= -info['ed1']['over_range'])[0]

        info['yunit'] = info['ed1']['units']
        info['tunit'] = info['id1']['units']
        info['yres'] = info['ed1']['dim_resolution']
        info['samplingrate'] = 1 / info['id1']['dim_scale']
        info['nop'] = datapoints

        return y, t, info, ind_over, ind_under

if __name__ == "__main__":
    file_wfm = sys.argv[1]
    y, t, info, ind_over, ind_under = wfm2read(file_wfm)
    plt.plot(t, y)
    plt.xlabel(info['tunit'])
    plt.ylabel(info['yunit'])
    plt.title("Tektronix WFM Waveform")
    plt.show()
