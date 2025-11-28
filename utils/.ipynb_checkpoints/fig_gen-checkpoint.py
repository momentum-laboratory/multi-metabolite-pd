from skimage.measure import regionprops
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.data_preprocessing import *
from utils.inference import *
from utils.colormaps import *

def phantom_fig_gen(dtype, device):
    fig = make_subplots(rows=2, cols=9, horizontal_spacing=0.001, vertical_spacing=0.2,
                        subplot_titles=(
                            '5 mM',
                            '6 mM',
                            '9 mM',
                            '10 mM',
                            '12 mM',
                            '12 mM',
                            '15 mM',
                            '16 mM',
                            '20 mM',)
                        )

    s = 26
    top_gap = 3
    top_gap = 0

    col_location_lists = [[6, 8, 9], [1, 4, 7], [2, 3, 5]]
    for phantom_i, col_location_list in enumerate(col_location_lists):
        # raw data preperation
        glu_scan = np.load(os.path.join('data', f'phantom_{phantom_i + 1}', f'raw_glu.npy'))
        acquired_data_glu = phantom_data_preparation(np.transpose(glu_scan, (0, 2, 1)), dtype, device)

        # inference and quantification
        glu_nn_path = os.path.join('nn_weights', 'in_vitro', 'glu', 'trained_nn.pt')
        array_shape = glu_scan[0, :, :].shape
        quant_maps_glu = phantom_glu_inference(glu_nn_path, acquired_data_glu, array_shape, device)

        for vial_i, col_i in enumerate(col_location_list):
            vial_mask = np.transpose(
                np.load(os.path.join('data', f'phantom_{phantom_i + 1}', f'mask_{vial_i + 1}.npy')))
            rows, cols = np.where(vial_mask == 1)
            x_loc = np.mean(cols).astype(int)
            y_loc = np.mean(rows).astype(int)

            fs_flat = np.zeros([s + top_gap, s])
            ksw_flat = np.zeros([s + top_gap, s])
            masked_fs = quant_maps_glu['fs'] * vial_mask * 110e3 / 3
            masked_ksw = quant_maps_glu['ksw'] * vial_mask

            fs_flat[top_gap:s + top_gap, 0:s] = masked_fs[
                int(y_loc - (s / 2)):int(y_loc + (s / 2)), int(x_loc - (s / 2)):int(x_loc + (s / 2))]
            ksw_flat[top_gap:s + top_gap, 0:s] = masked_ksw[
                int(y_loc - (s / 2)):int(y_loc + (s / 2)), int(x_loc - (s / 2)):int(x_loc + (s / 2))]

            # Add heatmaps for the three arrays
            heatmap_fs = go.Heatmap(z=fs_flat, colorscale=custom_viridis, coloraxis='coloraxis1')
            heatmap_ksw = go.Heatmap(z=ksw_flat, colorscale=custom_plasma, coloraxis='coloraxis2')
            fig.add_trace(heatmap_fs, row=1, col=col_i)
            fig.add_trace(heatmap_ksw, row=2, col=col_i)

            for row_i in [1, 2]:
                # Add individual titles and separate colorbars
                fig.update_xaxes(row=row_i, col=col_i, showgrid=False, showticklabels=False)
                fig.update_yaxes(row=row_i, col=col_i, showgrid=False, showticklabels=False, autorange='reversed')

    # Manually add separate colorbars
    f_const = 3 / 110000
    colorbar_fs = {'colorscale': custom_viridis, 'cmin': 0, 'cmax': 20}
    colorbar_ksw = {'colorscale': custom_plasma, 'cmin': 5000, 'cmax': 10000}

    fig.update_layout(
        template='plotly_white',  # Set the theme to plotly dark
        title_text=f"",
        title_font=dict(size=24),
        showlegend=False,  # Hide legend
        height=300,
        width=850,  # Set a width based on your preference
        margin=dict(l=10, r=10, t=30, b=0),  # Adjust top and bottom margins
        title=dict(x=0.2, y=0.97),  # Adjust the title position
        coloraxis1=colorbar_fs,
        coloraxis2=colorbar_ksw,
        coloraxis1_colorbar=dict(x=0.5, y=0.38, len=1, thickness=18, orientation='h'),
        coloraxis2_colorbar=dict(x=0.5, y=-0.228, len=1, thickness=18, orientation='h'),
    )

    fig.update_yaxes(title_text='Glu (mM)', title_font=dict(size=18), row=1, col=1)
    fig.update_yaxes(title_text='k<sub>sw</sub> (s<sup>-1</sup>)', title_font=dict(size=18), row=2, col=1)

    fig.show()


def single_rep_fig_gen(dtype, device):
    n_rows = 5
    n_cols = 5
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        horizontal_spacing=0.002, vertical_spacing=0.005,
                        subplot_titles=['T₁ (ms)', 'MT f<sub>ss</sub> (%)', 'rNOE f<sub>s</sub> (%)',
                                        'Amide f<sub>s</sub> (%)', 'Glu (mM)',
                                        '', '', '', '', '',
                                        '', '', '', '', '',
                                        'T₂ (ms)', 'MT k<sub>ssw</sub> (s<sup>-1</sup>)',
                                        'rNOE k<sub>sw</sub> (s<sup>-1</sup>)', 'Amide k<sub>sw</sub> (s<sup>-1</sup>)',
                                        'Glu k<sub>sw</sub> (s<sup>-1</sup>)',
                                        '', '', '', '', ''],
                        row_heights=[0.2, 0.2, 0.15, 0.2, 0.2],
                        )

    # Iterate over pre-mptp and post-mptp
    scan_array = {}
    for time_point_i, time_point in enumerate(['pre_mptp', 'post_mptp']):
        print(f'Processing {time_point}:')
        for scan_type in ['mask', 'highres', 't1_map', 't2_map', 'raw_mt', 'raw_rnoe', 'raw_amide', 'raw_glu']:
            cur_scan = np.load(os.path.join('data', 'mouse_9', time_point, f'{scan_type}.npy'))
            scan_array[scan_type] = cur_scan

        # crop image
        mask = scan_array['mask']
        mask_cen = regionprops(mask)[0].centroid
        # cropped_mask = img_cropper(mask_cen, scan_array['mask'])

        array_shape = scan_array['t1_map'].shape
        prepared_array = data_preparation(scan_array, dtype, device)

        mt_nn_path = os.path.join('nn_weights', 'in_vivo', 'mt', 'trained_nn.pt')
        input_mt_param, quant_maps_mt = mt_inference(mt_nn_path, prepared_array, array_shape, device)
        rnoe_nn_path = os.path.join('nn_weights', 'in_vivo', 'rnoe', 'trained_nn.pt')
        quant_maps_rnoe = rnoe_inference(rnoe_nn_path, prepared_array, input_mt_param, array_shape, device)
        amide_glu_nn_path = os.path.join('nn_weights', 'in_vivo', 'amide_glu', 'trained_nn.pt')
        quant_maps_amide, quant_maps_glu = amide_glu_inference(amide_glu_nn_path, prepared_array, input_mt_param,
                                                               array_shape, device)

        # Create maps with masked values set to NaN
        maps = [
            np.where(mask == 0, np.nan, scan_array['t1_map'] * mask),
            np.where(mask == 0, np.nan, scan_array['t2_map'] * mask),
            np.where(mask == 0, np.nan, quant_maps_mt['fs'] * 100 * mask),
            np.where(mask == 0, np.nan, quant_maps_mt['ksw'] * mask),
            np.where(mask == 0, np.nan, quant_maps_rnoe['fs'] * 100 * mask),
            np.where(mask == 0, np.nan, quant_maps_rnoe['ksw'] * mask),
            np.where(mask == 0, np.nan, quant_maps_amide['fs'] * 100 * mask),
            np.where(mask == 0, np.nan, quant_maps_amide['ksw'] * mask),
            np.where(mask == 0, np.nan, quant_maps_glu['fs'] * (110000 / 3) * mask),
            np.where(mask == 0, np.nan, quant_maps_glu['ksw'] * mask),
        ]

        cropped_highres = img_cropper(mask_cen, scan_array['highres'], highres_flag=True)
        hr_heatmap = go.Heatmap(z=cropped_highres, coloraxis='coloraxis11',
                                x=np.arange(0, cropped_highres.shape[1]) / 2,
                                y=np.arange(0, cropped_highres.shape[0]) / 2)
        for c_i in np.arange(1, n_cols + 1):
            fig.add_trace(hr_heatmap, row=1 + 1 * time_point_i, col=c_i)
            fig.add_trace(hr_heatmap, row=4 + 1 * time_point_i, col=c_i)

        heatmap_t1 = go.Heatmap(z=img_cropper(mask_cen, maps[0]), showscale=False, colorscale=plotly_lipari,
                                coloraxis='coloraxis1')
        heatmap_t2 = go.Heatmap(z=img_cropper(mask_cen, maps[1]), showscale=False, colorscale=plotly_navia,
                                coloraxis='coloraxis2')
        fig.add_trace(heatmap_t1, row=1 + 1 * time_point_i, col=1)
        fig.add_trace(heatmap_t2, row=4 + 1 * time_point_i, col=1)

        heatmap_mt_fs = go.Heatmap(z=img_cropper(mask_cen, maps[2]), colorscale=custom_viridis, coloraxis='coloraxis3')
        heatmap_mt_ksw = go.Heatmap(z=img_cropper(mask_cen, maps[3]), colorscale=custom_plasma, coloraxis='coloraxis4')
        fig.add_trace(heatmap_mt_fs, row=1 + 1 * time_point_i, col=2)
        fig.add_trace(heatmap_mt_ksw, row=4 + 1 * time_point_i, col=2)

        heatmap_noe_fs = go.Heatmap(z=img_cropper(mask_cen, maps[4]), colorscale=custom_viridis, coloraxis='coloraxis5')
        heatmap_noe_ksw = go.Heatmap(z=img_cropper(mask_cen, maps[5]), colorscale=custom_plasma, coloraxis='coloraxis6')
        fig.add_trace(heatmap_noe_fs, row=1 + 1 * time_point_i, col=3)
        fig.add_trace(heatmap_noe_ksw, row=4 + 1 * time_point_i, col=3)

        heatmap_amide_fs = go.Heatmap(z=img_cropper(mask_cen, maps[6]), colorscale=custom_viridis,
                                      coloraxis='coloraxis7')
        heatmap_amide_ksw = go.Heatmap(z=img_cropper(mask_cen, maps[7]), colorscale=custom_plasma,
                                       coloraxis='coloraxis8')
        fig.add_trace(heatmap_amide_fs, row=1 + 1 * time_point_i, col=4)
        fig.add_trace(heatmap_amide_ksw, row=4 + 1 * time_point_i, col=4)

        heatmap_glu_fs = go.Heatmap(z=img_cropper(mask_cen, maps[8]), colorscale=custom_viridis, coloraxis='coloraxis9')
        heatmap_glu_ksw = go.Heatmap(z=img_cropper(mask_cen, maps[9]), colorscale=custom_plasma,
                                     coloraxis='coloraxis10')
        fig.add_trace(heatmap_glu_fs, row=1 + 1 * time_point_i, col=5)
        fig.add_trace(heatmap_glu_ksw, row=4 + 1 * time_point_i, col=5)

        cb_len = (1 / n_cols) * 0.9
        start = 1 / (n_cols * 2)
        step = (1) / n_cols
        c_bar_y = 0.48
        fig.update_layout(
            coloraxis1=color_bar_dict['colorbar_t1w'],
            coloraxis2=color_bar_dict['colorbar_t2w'],
            coloraxis3=color_bar_dict['colorbar_fs_mt'],
            coloraxis4=color_bar_dict['colorbar_ksw_mt'],
            coloraxis5=color_bar_dict['colorbar_fs_noe'],
            coloraxis6=color_bar_dict['colorbar_ksw_noe'],
            coloraxis7=color_bar_dict['colorbar_fs_amide'],
            coloraxis8=color_bar_dict['colorbar_ksw_amide'],
            coloraxis9=color_bar_dict['colorbar_fs'],
            coloraxis10=color_bar_dict['colorbar_ksw'],
            coloraxis11=color_bar_dict['colorbar_highres'],
            coloraxis1_colorbar=dict(orientation='h', x=start, y=c_bar_y, len=cb_len, thickness=18),
            coloraxis3_colorbar=dict(orientation='h', x=start + step, y=c_bar_y, len=cb_len, thickness=18),
            coloraxis5_colorbar=dict(orientation='h', x=start + 2 * step, y=c_bar_y, len=cb_len, thickness=18),
            coloraxis7_colorbar=dict(orientation='h', x=start + 3 * step, y=c_bar_y, len=cb_len, thickness=18),
            coloraxis9_colorbar=dict(orientation='h', x=start + 4 * step, y=c_bar_y, len=cb_len, thickness=18),
            coloraxis2_colorbar=dict(orientation='h', x=start, y=-0.1, len=cb_len, thickness=18),
            coloraxis4_colorbar=dict(orientation='h', x=start + step, y=-0.1, len=cb_len, thickness=18),
            coloraxis6_colorbar=dict(orientation='h', x=start + 2 * step, y=-0.1, len=cb_len, thickness=18),
            coloraxis8_colorbar=dict(orientation='h', x=start + 3 * step, y=-0.1, len=cb_len, thickness=18),
            coloraxis10_colorbar=dict(orientation='h', x=start + 4 * step, y=-0.1, len=cb_len, thickness=18),
            coloraxis11_showscale=False
        )

        print('\n')

    for title_i in [1,4]:
        fig.update_yaxes(title_text='Pre-MPTP', showgrid=False, row=title_i, col=1,
                         title=dict(font=dict(size=18), standoff=0))
        fig.update_yaxes(title_text='Post-MPTP', showgrid=False, row=title_i+1, col=1,
                         title=dict(font=dict(size=18), standoff=0))

    # Add individual titles and separate colorbars
    for c_i in np.arange(1, n_cols + 1):
        for r_i in [1, 2, 4, 5]:
            fig.update_xaxes(row=r_i, col=c_i, showgrid=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, row=r_i, col=c_i, showticklabels=False,
                             autorange='reversed')  # Reverse the y-axis

            fig.update_yaxes(scaleanchor=f'x{(r_i - 1) * n_cols + c_i}', scaleratio=1, row=r_i, col=c_i)

    fig.update_layout(
        template='plotly_white',
        title_text='',
        showlegend=False,  # Hide legend
        height=600,
        width=920,  # Set a width based on your preference
        margin=dict(l=40, r=40, t=60, b=60),  # Adjust top and bottom margins
        title=dict(x=0.02, y=0.97)  # Adjust the title position
    )

    fig.show()

def three_rep_fig_gen(dtype, device):
    n_rows = 8
    n_cols = 4
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        horizontal_spacing=0.002, vertical_spacing=0.005,
                        subplot_titles=['MT f<sub>ss</sub> (%)', 'rNOE f<sub>s</sub> (%)', 'Amide f<sub>s</sub> (%)',
                                        'Glu (mM)'],
                        row_heights=[0.2, 0.2, 0.015, 0.2, 0.2, 0.015, 0.2, 0.2],
                        )
    mt_nn_path = os.path.join('nn_weights', 'in_vivo', 'mt', 'trained_nn.pt')
    rnoe_nn_path = os.path.join('nn_weights', 'in_vivo', 'rnoe', 'trained_nn.pt')
    amide_glu_nn_path = os.path.join('nn_weights', 'in_vivo', 'amide_glu', 'trained_nn.pt')

    # Iterate over pre-mptp and post-mptp
    scan_array = {}
    for time_point_i, time_point in enumerate(['pre_mptp', 'post_mptp']):
        print(f'Processing {time_point}:')
        for fig_mouse_i, mouse_i in enumerate([17, 18, 19]):
            for scan_type in ['mask', 'highres', 't1_map', 't2_map', 'raw_mt', 'raw_rnoe', 'raw_amide', 'raw_glu']:
                cur_scan = np.load(os.path.join('data', f'mouse_{mouse_i}', time_point, f'{scan_type}.npy'))
                scan_array[scan_type] = cur_scan

            # crop image
            mask = scan_array['mask']
            mask_cen = regionprops(mask)[0].centroid
            # cropped_mask = img_cropper(mask_cen, scan_array['mask'])

            array_shape = scan_array['t1_map'].shape
            prepared_array = data_preparation(scan_array, dtype, device)

            input_mt_param, quant_maps_mt = mt_inference(mt_nn_path, prepared_array, array_shape, device)
            quant_maps_rnoe = rnoe_inference(rnoe_nn_path, prepared_array, input_mt_param, array_shape, device)
            quant_maps_amide, quant_maps_glu = amide_glu_inference(amide_glu_nn_path, prepared_array, input_mt_param,
                                                                   array_shape, device)

            # Create maps with masked values set to NaN
            maps = [
                quant_maps_mt['fs'] * 100 * mask,
                quant_maps_rnoe['fs'] * 100 * mask,
                quant_maps_amide['fs'] * 100 * mask,
                quant_maps_glu['fs'] * (110000 / 3) * mask,
            ]

            heatmap_mt_fs = go.Heatmap(z=img_cropper(mask_cen, maps[0]), colorscale=custom_viridis,
                                       coloraxis='coloraxis1')
            fig.add_trace(heatmap_mt_fs, row=1 + fig_mouse_i * 3 + 1 * time_point_i, col=1)

            heatmap_noe_fs = go.Heatmap(z=img_cropper(mask_cen, maps[1]), colorscale=custom_viridis,
                                        coloraxis='coloraxis2')
            fig.add_trace(heatmap_noe_fs, row=1 + fig_mouse_i * 3 + 1 * time_point_i, col=2)

            heatmap_amide_fs = go.Heatmap(z=img_cropper(mask_cen, maps[2]), colorscale=custom_viridis,
                                          coloraxis='coloraxis3')
            fig.add_trace(heatmap_amide_fs, row=1 + fig_mouse_i * 3 + 1 * time_point_i, col=3)

            heatmap_glu_fs = go.Heatmap(z=img_cropper(mask_cen, maps[3]), colorscale=custom_viridis,
                                        coloraxis='coloraxis4')
            fig.add_trace(heatmap_glu_fs, row=1 + fig_mouse_i * 3 + 1 * time_point_i, col=4)

            cb_len = (1 / n_cols) * 0.9
            start = 1 / (n_cols * 2)
            step = (1) / n_cols
            c_bar_y = -0.08
            fig.update_layout(
                coloraxis1=color_bar_dict['colorbar_fs_mt'],
                coloraxis2=color_bar_dict['colorbar_fs_noe'],
                coloraxis3=color_bar_dict['colorbar_fs_amide'],
                coloraxis4=color_bar_dict['colorbar_fs'],
                coloraxis1_colorbar=dict(orientation='h', x=start, y=c_bar_y, len=cb_len, thickness=18),
                coloraxis2_colorbar=dict(orientation='h', x=start + step, y=c_bar_y, len=cb_len, thickness=18),
                coloraxis3_colorbar=dict(orientation='h', x=start + 2 * step, y=c_bar_y, len=cb_len, thickness=18),
                coloraxis4_colorbar=dict(orientation='h', x=start + 3 * step, y=c_bar_y, len=cb_len, thickness=18),
            )

            print('\n')

    for title_i in [1, 4, 7]:
        fig.update_yaxes(title_text='Pre-MPTP', showgrid=False, row=title_i, col=1, title_font=dict(size=16))
        fig.update_yaxes(title_text='Post-MPTP', showgrid=False, row=title_i + 1, col=1, title_font=dict(size=16))

    # Add individual titles and separate colorbars
    for c_i in np.arange(1, n_cols + 1):
        for r_i in [1, 2, 4, 5, 7, 8]:
            fig.update_xaxes(row=r_i, col=c_i, showgrid=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, row=r_i, col=c_i, showticklabels=False,
                             autorange='reversed')  # Reverse the y-axis

            fig.update_yaxes(scaleanchor=f'x{(r_i - 1) * n_cols + c_i}', scaleratio=1, row=r_i, col=c_i)

    existing_annotations = list(fig.layout.annotations)
    exm_gap = 0.4
    base_y = 0.1
    labels = ['Example 1', 'Example 2', 'Example 3']

    custom_annotations = [
        dict(
            text=label,
            x=-0.09,
            y=base_y + (len(labels) - 1 - i) * exm_gap,
            xref='paper', yref='paper',
            showarrow=False,
            textangle=-90,
            font=dict(size=18)
        )
        for i, label in enumerate(labels)
    ]
    fig.update_layout(annotations=existing_annotations + custom_annotations)

    fig.update_layout(
        template='plotly_white',
        title_text='',
        showlegend=False,  # Hide legend
        height=800,
        width=840,  # Set a width based on your preference
        margin=dict(l=80, r=40, t=60, b=60),  # Adjust top and bottom margins
        title=dict(x=0.02, y=0.97)  # Adjust the title position
    )

    fig.show()

