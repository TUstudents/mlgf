import argparse
from mlgf.data import Dataset
from mlgf.lib.doshelper import calc_dos, NDAAAFit
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from mlgf.utils.xyz_traj import water_get_geometry, SmallMolecule
import json

def draw_water(ax, md):
    mol = md.data['mol']
    (r1, r2), theta = water_get_geometry(SmallMolecule.from_pyscf_mol(mol))

    ax.add_patch(patches.Circle((0, 0), 0.12, color='red', fill=True))
    thetaby2 = theta/2
    
    h1 = (r1*np.cos(thetaby2), r1*np.sin(thetaby2))
    h2 = (r2*np.cos(thetaby2), -r2*np.sin(thetaby2))
    ax.plot([0, h1[0]], [0, h1[1]], color='black', zorder=0)
    ax.plot([0, h2[0]], [0, h2[1]], color='black', zorder=0)
    ax.add_patch(patches.Circle(h1, 0.1, color='blue', fill=True))
    ax.add_patch(patches.Circle(h2, 0.1, color='blue', fill=True))
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    xmin=0
    xmax=max(r1*np.cos(thetaby2), r2*np.cos(thetaby2))
    ymin=-r2*np.sin(thetaby2)
    ymax=r1*np.sin(thetaby2)
    maxdim = 3
    box_ctr = ((xmax+xmin)/2, (ymax+ymin)/2)
    padding = 0.1
    ax.set_xlim(box_ctr[0] - maxdim/2 - padding, box_ctr[0] + maxdim/2 + padding)
    ax.set_ylim(box_ctr[1] - maxdim/2 - padding, box_ctr[1] + maxdim/2 + padding)
    ax.set_aspect('equal')
    ax.text(box_ctr[0]+maxdim/2-0.3, 
            box_ctr[1]+maxdim/2-0.3,
            r'$\theta={:.2f}^\circ$'.format(np.degrees(theta)) + "\n" + r'$r_1={:.2f}\, \AA$'.format(r1) + "\n" + r'$r_2={:.2f}\, \AA$'.format(r2))

     


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot DOS from water FCI calculations, draw geometry, and compute negative DOS area')
    parser.add_argument('-d', '--diag', action='store_true', help='Use diagonal approximation')
    parser.add_argument('-f', '--format', default='png', help='Output format for matplotlib')
    parser.add_argument('-o', '--output', default=os.getcwd(), help='Output directory')
    parser.add_argument('-p', '--prefix', default='dos', help='Prefix for output files')
    parser.add_argument('-n', '--npts', default=201, type=int, help='# points in frequency grid for DOS')
    parser.add_argument('-b', '--broadening', default=1e-3, type=float, help='Broadening (eta) for DOS')
    parser.add_argument('--is-dft', action='store_true', help='Do not assume Vxc = Vk')
    parser.add_argument('-t', '--title', default='', help='Title for plot')
    parser.add_argument('--noplot', action='store_true', help='Do not plot')
    
    parser.add_argument('files', metavar='FILE', nargs='+', help='Joblib files to plot')
    args = parser.parse_args()
    
    dset = Dataset(args.files)
    grid = np.linspace(-1,1, args.npts)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    negative_dos_area = []
    
    for name in dset.fnames:
        basename = pathlib.Path(name).stem
        md = dset.get_by_fname(name)
        
        vk_minus_vxc = None
        if args.is_dft:
            vk_minus_vxc = md.data['vk'] - md.data['vxc']
        
        dos = calc_dos(grid, args.broadening, md.data['coeff'], md.data['omega_fit'], md.data['mo_energy'],
                       vk_minus_vxc=vk_minus_vxc, diag=args.diag)
        aaafit = NDAAAFit(md.data['sigmaI'], md.data['omega_fit'], md.data['mo_energy'], vk_minus_vxc=vk_minus_vxc,
                          diag=args.diag, fitGF=False, eta=args.broadening,
                          fitDOS=False)
        aaafit.kernel(mmax=4)
        dos_aaa = aaafit.getdos(grid)
        #dos_aaa2 = aaafit.getdos2(grid, 1e-2, md.data['mo_energy'], vk_minus_vxc=vk_minus_vxc)
        if not args.noplot:
            fig = plt.figure(figsize=(12,5))
            ax = fig.add_subplot(1,2,1)
            ax_2 = fig.add_subplot(1,2,2)
            ax.plot(grid, dos, color='black', linewidth=1, label='Pade-Thiele')
            ax.fill_between(grid, np.zeros_like(grid), dos_aaa, facecolor='green', alpha=0.2, label='AAA')
            #ax.fill_between(grid, np.zeros_like(grid), dos_aaa2, facecolor='blue', alpha=0.2, label='AAA2')
            #ax.plot(grid, dos_aaa, color='green')
            ax.set_xlabel('Frequency (Ha)')
            ax.set_ylabel('DOS')
            ax.set_title(args.title.format(basename))
            #ax.set_xlim(right=1.2)
            ax.legend(loc='upper right')
            draw_water(ax_2, md)
            plt.savefig(os.path.join(args.output, args.prefix + '_' + basename + '.' + args.format))
            plt.close()

        mesh_size=grid[1]-grid[0]
        nar = np.sum(dos[dos<0])*mesh_size
        negative_dos_area.append((nar, basename))
    
    negative_dos_area.sort()
    if args.noplot:
        with open(args.prefix + '_negative_dos_area.json', 'w') as f:
            json.dump(negative_dos_area, f)
    
        areas = np.array([x[0] for x in negative_dos_area])
        minus_areas = areas[areas<0]
        nonneg_areas = areas[areas>=0]
        plt.plot(minus_areas)
        plt.plot(np.arange(len(nonneg_areas))+len(minus_areas), nonneg_areas)
        plt.xlabel('Index')
        plt.ylabel('Negative DOS area')
        plt.savefig(args.prefix + '_negative_dos_area.' + args.format)
    
    with open(args.prefix + '_good', 'w') as f:
        for area, basename in negative_dos_area:
            if area == 0:
                f.write(basename + '.chk\n')
                f.write(basename + '.joblib\n')