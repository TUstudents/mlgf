import os
import shutil

import numpy as np
import h5py
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

import joblib
import argparse
import matplotlib.pyplot as plt

# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.figure_factory as ff
# from plotly.subplots import make_subplots

from mlgf.lib.helpers import get_rsq

# import rdkit
# from rdkit.Chem.Draw import IPythonConsole

def plot_eval_dos(freqs_dos, dos_true, dos_ml, dos_hf = None):
    
    # mae_pred = np.mean(np.abs(dos_ml-dos_true))
    mae_pred = np.sum(np.abs(dos_ml-dos_true))/np.sum(dos_true)
    fig = go.Figure()
    if not dos_hf is None:
        mae_hf = np.sum(np.abs(dos_hf-dos_true))/np.sum(dos_true)
        fig.add_trace(go.Scatter(x = np.real(freqs_dos), y = dos_hf,  mode = 'lines', name = f'Hartree Fock<br>MAE: {mae_hf:0.2f}', line = dict(dash = 'dot')))
    fig.add_trace(go.Scatter(x = np.real(freqs_dos), y = dos_true,  mode = 'none', name = 'True', fill='tozeroy', fillcolor = 'rgba(0, 204, 150, 0.6)'))
    fig.add_trace(go.Scatter(x = np.real(freqs_dos), y = dos_ml,  mode = 'lines', name = f'ML Prediction<br>MAE: {mae_pred:0.2f}', line=dict(color='blue')))
    

    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='white'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='white',
        ticksuffix = ' (A.U.)', showticksuffix = 'last'
    )
    fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(
        showgrid=False,
    ),
    yaxis=dict(
        showgrid=False,
    )
    )
    
    return fig

def plot_dos_hf_ml_true(freqs_dos, dos_hf, dos_ml, dos_true, plotter='plotly', verbose = True,
              interactive=False, title = 'DOS Comparison'):
    """Plot true, predicted, and HF DOS.

    Args:
        freqs_dos (numpy array): x-axis of real frequency points
        dos_hf (numpy array): HF DOS
        dos_ml (numpy array): ML-predicted DOS
        dos_true (numpy array): True DOS (GW level, CCGF level, FCI level)
        plotter (str) : plotly or matplotlib
        verbose (bool): prints MAE for HF and ML DOS to stdout
        interactive (bool): whether plotly or matplot shows the plot 
        title (str): plot title
    """
    
    if plotter == 'plotly':
        plt = plot_eval_dos(freqs_dos, dos_true, dos_ml, dos_hf)
        plt.update_layout(title = f'DOS Comparison', title_x = 0.5)
        if interactive:
            plt.show()
        else: 
            plt.write_image('DOS_pred.png')   
    elif plotter == 'matplotlib':
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(np.real(freqs_dos), dos_ml, label = 'ML Target')
        ax.plot(np.real(freqs_dos), dos_true, label = 'True')
        ax.plot(np.real(freqs_dos), dos_hf, label = 'HF')

        ax.set_xlim(-1.0, 1.0)
        ax.legend(loc = 'upper right')
        ax.set_xlabel('Frequency (Ha)')
        ax.set_title(title)
        if interactive:
            plt.show()
        else:
            plt.savefig('DOS_pred.png')

    mae_pred = np.mean(np.abs(dos_ml-dos_true))
    mae_hf = np.mean(np.abs(dos_hf-dos_true))
    if verbose:
        print('MAE for ML-predicted DOS: ', mae_pred)
        print('MAE for HF DOS: ', mae_hf)

    return plt

def plot_model_perf(df_rsq, re_im = None):
    df = df_rsq.copy()
    if not re_im is None: df = df.loc[df['ReIm'] == re_im]
    else: re_im = 'ii, ij'
    min_true, max_true = np.min(df['True']), np.max(df['True']) 
    min_pred, max_pred = np.min(df['Pred']), np.max(df['Pred']) 
    fig = px.scatter(df, x="Pred", y="True", animation_frame="iw", color="ReIm", hover_name="Elements", hover_data= ['Rsq'],
               log_x=False, range_x=[min_pred, max_pred], range_y=[min_true, max_true])

    fig.update_layout(width = 700, height = 700, title = f'Model Performance {re_im}')
    fig.add_trace(go.Scatter(x = [-10**2, 10**2], y = [-10**2, 10**2],  mode = 'lines', name = 'y = x'))
    return fig

def qp_xy_plot(qpe_true, qpe_ml, mf_mo_energy):    
    rsq_ml = get_rsq(qpe_ml, qpe_true)
    rsq_mo = get_rsq(qpe_ml, mf_mo_energy)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(qpe_true, qpe_true, label = 'y=x')
    ax.scatter(qpe_true, qpe_ml, label = f'ML QP Energies (rsq = {rsq_ml:0.5f})', s=3, c = '#3CB043')
    ax.scatter(qpe_true, mf_mo_energy, label = f'MO Energies (rsq = {rsq_mo:0.5f})', s=3 , c = '#000000')
    

    ax.legend(loc = 'upper left')
    ax.set_xlabel('True QP Energies')
    ax.set_ylabel('ML QPE or MF MO Energies')
    ax.set_title('QPE prediction compared to True from ML-predicted Sigma')

    mae_pred = np.mean(np.abs(qpe_true-qpe_ml))
    mae_hf = np.mean(np.abs(qpe_true-mf_mo_energy))


    print(f'MAE for ML-predicted QPE (n = {len(qpe_true)}): ', mae_pred)
    print(f'MAE for mean field mo energies (n = {len(qpe_true)}): ', mae_hf)
    print('ML QPE: ',  qpe_ml)
    print('True QPE: ',  qpe_true)
    print(f'rsq ML, MO: {rsq_ml:0.5f}, {rsq_mo:0.5f}')

    return plt

def qp_energy_level_plot(qpe_true, qpe_ml):    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    
    for i in range(len(qpe_true)):
        x = [-1, 1]
        t = [qpe_true[i]]*2
        ml = [qpe_ml[i]]*2
        if i == 0: 
            ax.plot(x, t, label = 'True QP Energies',c = '#000000')
            ax.plot(x, ml, label = 'ML QP Energies', linestyle = 'dashed', c = '#3CB043')
        else:
            ax.plot(x, t, c = '#000000')
            ax.plot(x, ml, linestyle = 'dashed' , c = '#3CB043')

    

    ax.legend(loc = 'upper left')
    ax.set_xlabel('True QP Energies')
    ax.set_ylabel('Energy (a.u.)')
    ax.xaxis.set_ticklabels([])
    ax.set_title('True QPE vs. ML-predicted Energy Levels')

    mae_pred = np.mean(np.abs(qpe_true-qpe_ml))
    print(f'MAE for ML-predicted QPE (n = {len(qpe_true)}): ', mae_pred)
    print('ML QPE: ',  qpe_ml)
    print('True QPE: ',  qpe_true)

    return plt


def plot_by_omega_ind(i, y_pred, y_test_plot, real = True):
    if real: omega_points = 0
    else: omega_points = y_pred.shape[-1]//2
    test_omega_ind = i + omega_points
    if real and i == 0: sl = True
    else: sl = False

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = [-10**10, 10**10], y = [0, 0],  mode = 'lines', showlegend = False, line_color = 'lightgrey'))
    fig.add_trace(go.Scatter(y = [-10**10, 10**10], x = [0, 0],  mode = 'lines', showlegend = False, line_color = 'lightgrey'))
    fig.add_trace(go.Scatter(x = [-10**10, 10**10], y = [-10**10, 10**10],  mode = 'lines', name = 'y = x', line_color = 'blue', showlegend = sl)
    )
    fig.add_trace(go.Scatter(x = y_test_plot[:,test_omega_ind], y = y_pred[:,test_omega_ind],  mode = 'markers',  name ='true, pred', line_color = 'black', showlegend = sl)
    )
   

    return fig

def clean_up_fig_by_omega(fig, y_pred, y_test_plot, i, real, omega_fit = None):
    c = 2 - real
    r = i + 1
    
    if real: omega_points = 0
    else: omega_points = y_pred.shape[-1]//2

    if omega_fit is None:
        omega_fit = np.array([0.+7.15785522e-05j, 0.+4.06337059e-03j, 0.+1.72533649e-02j,
        0.+3.59014622e-02j, 0.+6.25474232e-02j, 0.+9.87009331e-02j,
        0.+1.46580843e-01j, 0.+1.95459911e-01j, 0.+2.73624248e-01j,
        0.+3.53912569e-01j, 0.+4.84611522e-01j, 0.+6.22572834e-01j,
        0.+8.02596369e-01j, 0.+1.04270642e+00j, 0.+1.37189377e+00j,
        0.+1.83919186e+00j, 0.+2.53290412e+00j, 0.+3.62538052e+00j])
        
    test_omega_ind = i + omega_points
    r1 = get_rsq(y_pred[:,test_omega_ind], y_test_plot[:,test_omega_ind])
    omega_plot = np.imag(omega_fit[i])
    ymin_test, ymax_test = min(y_test_plot[:,test_omega_ind]), max(y_test_plot[:,test_omega_ind])
    ymin_pred, ymax_pred = min(y_pred[:,test_omega_ind]), max(y_pred[:,test_omega_ind])
    
    xpos = 0.8*(ymax_test - ymin_test) + ymin_test
    ypos = 0.10*(ymax_pred - ymin_pred) + ymin_pred
    
    fig.add_annotation(text=f"Rsq = {r1:0.4f}", xref=f"x{r+omega_points}", yref=f"y{r+omega_points}", x=xpos, y=ypos, showarrow=False, row = r, col = c)
    
    if real: rt = 'Real'
    else: rt = 'Imaginary'
    fig.update_xaxes(
    title = f'True {rt} part on omega = {omega_plot:0.2f}i', range=[ymin_test, ymax_test],
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey', row = r, col = c
    )
    if real: rt = 'Predicted'
    else: rt = ''
    fig.update_yaxes(
        range=[ymin_pred, ymax_pred], title = rt, 
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        zeroline= True, row = r, col = c
    )
    return fig
    
    
def make_multifig_by_omega(y_pred, y_test_plot):
    nrows = y_pred.shape[-1] // 2
    ncols = 2
    fig = make_subplots(rows=nrows, cols=ncols)
    fig.update_layout(width = 700, height = 7000, font_family = 'Serif')
    
    for i in range(nrows):
        f = plot_by_omega_ind(i, y_pred, y_test_plot, real = True)
        for t in f.data: 
            fig.append_trace(t, row=i+1, col=1)
            fig = clean_up_fig_by_omega(fig, y_pred, y_test_plot, i, True)
            
        f = plot_by_omega_ind(i, y_pred, y_test_plot, real = False)
        for t in f.data:
            fig.append_trace(t, row=i+1, col=2)
            fig = clean_up_fig_by_omega(fig, y_pred, y_test_plot, i, False)
    fig.update_layout(title_x = 0.5, title = r'$\text{Comparison of } \Sigma_{ii}(i\omega) \text{ in MO basis}$')
    fig.update_layout(plot_bgcolor='white', font_family = 'Serif')

    return fig

from plotly.subplots import make_subplots

def make_multifig_by_orb_homo_lumo(y_pred, y_test_plot, homo_ind = 0, lumo_ind = 1, yp_error = None, omega_fit = None):
    ncols = 2
    fig = make_subplots(rows=1, cols=ncols)
    
    ypi = np.imag(y_pred)
    yti = np.imag(y_test_plot)
    ypr = np.real(y_pred)
    ytr = np.real(y_test_plot)
    
    if not yp_error is None:
        ypr_error = np.real(yp_error)
        ypi_error = np.imag(yp_error)
    else:
        ypr_error = None
        ypi_error = None
        
    f = plot_by_orb_ind(homo_ind, ypr, ytr, color = 'red', showlegend = False, yp_error = ypr_error, omega_fit = omega_fit)
    row, col = 1, 1
    for t in f.data: 
        fig.append_trace(t, row=1, col=1)
        fig = clean_up_fig_by_orb(fig, ypr, ytr, [homo_ind, lumo_ind], row, col, omega_fit = omega_fit)
        
    f = plot_by_orb_ind(homo_ind, ypi, yti, color = 'red', showlegend = True, names = ['HOMO Predicted', 'HOMO True'], yp_error = ypi_error, omega_fit = omega_fit)
    row, col = 1, 2
    for t in f.data:
        fig.append_trace(t, row=1, col=2)
        fig = clean_up_fig_by_orb(fig, ypi, yti, [homo_ind, lumo_ind], row, col, omega_fit = omega_fit)

    f = plot_by_orb_ind(lumo_ind, ypr, ytr, color = 'blue', showlegend = False, yp_error = ypr_error, omega_fit = omega_fit)
    row, col = 1, 1
    for t in f.data: 
        fig.append_trace(t, row=1, col=1)
        fig = clean_up_fig_by_orb(fig, ypr, ytr, [homo_ind, lumo_ind], row, col, omega_fit = omega_fit)
        
    f = plot_by_orb_ind(lumo_ind, ypi, yti, color = 'blue', showlegend = True, names = ['LUMO Predicted', 'LUMO True'], yp_error = ypi_error, omega_fit = omega_fit)
    row, col = 1, 2
    for t in f.data:
        fig.append_trace(t, row=1, col=2)
        fig = clean_up_fig_by_orb(fig, ypi, yti, [homo_ind, lumo_ind], row, col, omega_fit = omega_fit)

    # fig.update_layout(title_x = 0.5, title = r'$\text{Comparison of } \Sigma_{ii}(i\omega) \text{ in MO basis (Left: Real Part, Right: Imaginary Part)}$')
    fig.update_layout(plot_bgcolor='white', font_family = 'Serif')

    return fig

def plot_by_orb_ind(i, yp, yt, omega_fit = None, color = 'red', showlegend = False, names = ['Predicted', 'True'], yp_error = None):
    nomega = yp.shape[-1] // 2

    if omega_fit is None:
        omega_fit = np.array([0.+7.15785522e-05j, 0.+4.06337059e-03j, 0.+1.72533649e-02j,
       0.+3.59014622e-02j, 0.+6.25474232e-02j, 0.+9.87009331e-02j,
       0.+1.46580843e-01j, 0.+1.95459911e-01j, 0.+2.73624248e-01j,
       0.+3.53912569e-01j, 0.+4.84611522e-01j, 0.+6.22572834e-01j,
       0.+8.02596369e-01j, 0.+1.04270642e+00j, 0.+1.37189377e+00j,
       0.+1.83919186e+00j, 0.+2.53290412e+00j, 0.+3.62538052e+00j])

    fig = go.Figure()
    x = list(np.imag(omega_fit))
    fig.add_trace(go.Scatter(x = x, y = yp[i,:],  mode = 'lines',  name =names[0], line_color = color, showlegend = showlegend)
    )

    fig.add_trace(go.Scatter(x = x, y = yt[i,:],  mode = 'lines',  name =names[1], line_color = color, showlegend = showlegend, line = dict(dash = 'dot'))
    )

    if not yp_error is None:
        y_upper = list(yp[i,:] + yp_error[i,:])
        y_lower = list(yp[i,:] - yp_error[i,:])
        if color == 'blue':
            fillcolor = 'rgba(0,0,255,0.2)'
        else:
            fillcolor = 'rgba(255,0,0,0.2)'
        fig.add_trace(go.Scatter(
            x=x+x[::-1], # x, then x reversed
            y=y_upper+y_lower[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor=fillcolor,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
                     )

    return fig  

def clean_up_fig_by_orb(fig, y_pred, y_test_plot, i, row, col, omega_fit = None):    
    nomega = y_pred.shape[-1] // 2

    if omega_fit is None:
        omega_fit = np.array([0.+7.15785522e-05j, 0.+4.06337059e-03j, 0.+1.72533649e-02j,
       0.+3.59014622e-02j, 0.+6.25474232e-02j, 0.+9.87009331e-02j,
       0.+1.46580843e-01j, 0.+1.95459911e-01j, 0.+2.73624248e-01j,
       0.+3.53912569e-01j, 0.+4.84611522e-01j, 0.+6.22572834e-01j,
       0.+8.02596369e-01j, 0.+1.04270642e+00j, 0.+1.37189377e+00j,
       0.+1.83919186e+00j, 0.+2.53290412e+00j, 0.+3.62538052e+00j])
        
    norbs = y_pred.shape[0]
        
    ymin_test, ymax_test = np.min(y_test_plot[i, :]), np.max(y_test_plot[i,:])
    ymin_pred, ymax_pred = np.min(y_pred[i,:]), np.max(y_pred[i,:])
    
    ymin = np.min([ymin_test, ymin_pred])
    ymax = np.max([ymax_test, ymax_pred])
    xmin, xmax = np.min(np.imag(omega_fit)), np.max(np.imag(omega_fit))
    
    fig.update_xaxes(
    title = 'omega', range=[xmin, xmax],
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey', row = row, col = col
    )
    fig.update_yaxes(
        range=[ymin, ymax],
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        zeroline= True, row = row, col = col
    )
    return fig
    
    
def make_multifig_by_orb(y_pred, y_test_plot):
    nrows = y_pred.shape[0]
    ncols = 2
    fig = make_subplots(rows=nrows, cols=ncols)

    ypi = np.imag(y_pred)
    yti = np.imag(y_test_plot)
    ypr = np.real(y_pred)
    ytr = np.real(y_test_plot)
    
    for i in range(nrows):
        f = plot_by_orb_ind(i, ypr, ytr)
        for t in f.data: 
            fig.append_trace(t, row=i+1, col=1)
            fig = clean_up_fig_by_orb(fig, ypr, ytr, i)
            
        f = plot_by_orb_ind(i, ypi, yti)
        for t in f.data:
            fig.append_trace(t, row=i+1, col=2)
            fig = clean_up_fig_by_orb(fig, ypi, yti, i)
        print(i)
    fig.update_layout(title_x = 0.5, title = r'$\text{Comparison of } \Sigma_{ii}(i\omega) \text{ in MO basis (Left: Real Part, Right: Imaginary Part)}$')
    fig.update_layout(plot_bgcolor='white', font_family = 'Serif')

    return fig



def qp_energy_level_plot(df, unit_errors = 'eV', au_to_ev = 27.2114):
    '''
    prediction is one of ML or PBE, PBE, will not show ML
    '''
    smiles_list = list(df['SMILES'])
    homos_ml = df['HOMO ML'].to_numpy()
    lumos_ml = df['LUMO ML'].to_numpy()
    homos_true =  df['HOMO True'].to_numpy()
    lumos_true =  df['LUMO True'].to_numpy()
    homos_pbe =  df['HOMO PBE'].to_numpy()
    lumos_pbe =  df['LUMO PBE'].to_numpy()

    nfiles = len(df)
    fig = go.Figure()
    width = 0.7/nfiles
    spacer = 0.15/nfiles
    current_x = 0
    x_mid = []
    y_homos, y_lumos = [], []
    homo_errors, lumo_errors = [], []
    homo_pbe_errors, lumo_pbe_errors = [], []
    tmp_folder = os.getcwd() + '/tmp_imgs'
    os.mkdir(tmp_folder)
    for j in range(len(df)):
        smiles = smiles_list[j]
        smiles2mol(smiles, filename = f'{tmp_folder}/{j}.png')
        qpe_ml, qpe_true, qpe_pbe = np.array([homos_ml[j], lumos_ml[j]]), np.array([homos_true[j], lumos_true[j]]), np.array([homos_pbe[j], lumos_pbe[j]])
        if unit_errors == 'eV': qpe_true, qpe_ml, qpe_pbe = au_to_ev*qpe_true, au_to_ev*qpe_ml, au_to_ev*qpe_pbe
        lw = 1
        x = np.array([current_x + spacer, current_x + spacer+ width])
        x_mid.append(np.mean(x))
        y_homos.append(np.min([qpe_true[0], qpe_ml[0]]))
        y_lumos.append(np.max([qpe_true[1], qpe_ml[1]]))
        homo_errors.append(qpe_ml[0] - qpe_true[0])
        lumo_errors.append(qpe_ml[1] - qpe_true[1])
        for i in range(len(qpe_true)):
            t = [qpe_true[i]]*2
            ml = [qpe_ml[i]]*2
            pbe = [qpe_pbe[i]]*2
            
            if i == 0 and j == 0: 
                fig.add_trace(go.Scatter(x = x, y = t, name = 'True QP Energies', text = smiles, customdata = [smiles]*2, hoverinfo = 'all', line = dict(color='black', width=lw, dash='3px'), mode='lines'))
                fig.add_trace(go.Scatter(x = x, y = ml, name = 'ML QP Energies', text = smiles, customdata = [smiles]*2,hoverinfo = 'all', line = dict(color='royalblue', width=lw),  mode='lines'))
                fig.add_trace(go.Scatter(x = x, y = pbe, name = 'DFT@PBE0 KS Energies', text = smiles, customdata = [smiles]*2,hoverinfo = 'all', line = dict(color='rgba(255, 0, 0, 0.5)', width=lw),  mode='lines'))
            else:
                fig.add_trace(go.Scatter(x = x, y = t, text = smiles, customdata = [smiles]*2,hoverinfo = 'all', line = dict(color='black', width=lw, dash='3px'), mode='lines', showlegend = False))
                fig.add_trace(go.Scatter(x = x, y = ml, text = smiles, customdata = [smiles]*2, hoverinfo = 'all', line = dict(color='royalblue', width=lw), mode='lines', showlegend = False))
                fig.add_trace(go.Scatter(x = x, y = pbe, text = smiles, customdata = [smiles]*2,hoverinfo = 'all', line = dict(color='rgba(255, 0, 0, 0.5)', width=lw),  mode='lines', showlegend = False))
        current_x = current_x + spacer + spacer + width
    fig.update_layout(yaxis_title = f'Energy ({unit_errors})', xaxis_range = [0,1], title_x = 0.5)
    fig.update_layout(xaxis_visible=True, xaxis_showticklabels=False, yaxis_visible = True)
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='white',
        ticksuffix = f' {unit_errors}', showticksuffix = 'last'
    )
    fig.update_xaxes(
        mirror=True,
        showline=True,
        linecolor='black',
        gridcolor='white'
    )
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)'}) # , width = 1000
    
    k = 0
    for x, yh, yl, yh_lab, yl_lab in zip(x_mid, y_homos, y_lumos, homo_errors, lumo_errors):
        fig.add_annotation(text=f"{yh_lab:0.2f}",
                        xref="x", yref="y",
                        x=x, y=yh-1, showarrow=False)
        fig.add_annotation(text=f"{yl_lab:0.2f}",
                    xref="x", yref="y",
                    x=x, y=yl+1, showarrow=False)
        k += 1

    for i in range(len(smiles_list)):
        img = Image.open(f'{tmp_folder}/{i}.png') # image path
        fig.add_layout_image(
            source=img,
            xref="xref",
            yref="yref",
            x=x_mid[i],
            y=0.2,
            xanchor="center",
            yanchor="center",
            sizex=0.1,
            sizey=0.1
        )

    shutil.rmtree(tmp_folder)
    return fig, x_mid, smiles_list, y_homos, y_lumos, homo_errors, lumo_errors
