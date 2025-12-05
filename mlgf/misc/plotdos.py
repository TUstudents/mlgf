import argparse
from mlgf.data import Dataset
from mlgf.lib.doshelper import calc_dos, NDAAAFit
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch plot DOS from sigmaI using analytic continuation')
    parser.add_argument('-d', '--diag', action='store_true', help='Use diagonal approximation')
    parser.add_argument('-f', '--format', default='png', help='Output format for matplotlib')
    parser.add_argument('-o', '--output', default=os.getcwd(), help='Output directory')
    parser.add_argument('-p', '--prefix', default='dos', help='Prefix for output files')
    parser.add_argument('-n', '--npts', default=201, type=int, help='# points in frequency grid for DOS')
    parser.add_argument('-b', '--broadening', default=1e-3, type=float, help='Broadening (eta) for DOS')
    parser.add_argument('--is-dft', action='store_true', help='Do not assume Vxc = Vk')
    parser.add_argument('-t', '--title', default='', help='Title for plot')
    
    parser.add_argument('files', metavar='FILE', nargs='+', help='Joblib files to plot')
    args = parser.parse_args()
    
    dset = Dataset(args.files)
    grid = np.linspace(-1, 1, args.npts)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    for name in dset.fnames:
        basename = pathlib.Path(name).stem
        md = dset.get_by_fname(name)
        
        vk_minus_vxc = None
        if args.is_dft:
            vk_minus_vxc = md.data['vk'] - md.data['vxc']
        
        if args.diag:
            coeffs = np.diagonal(md.data['coeff'], axis1=1, axis2=2)
        else:
            coeffs = md.data['coeff']
        #dos = calc_dos(grid, args.broadening, coeffs, md.data['omega_fit'], md.data['mo_energy'],
        #               vk_minus_vxc=vk_minus_vxc, diag=args.diag)
        
        aaafit = NDAAAFit(md.data['sigmaI'], md.data['omega_fit'], diag=False)
        aaafit.kernel()
        dos = aaafit.getdos(grid, args.broadening, md.data['mo_energy'], vk_minus_vxc=vk_minus_vxc)
        fig, ax = plt.figure(), plt.gca()
        ax.plot(grid, dos, color='black', linewidth=1)
        ax.set_xlabel('Frequency (Ha)')
        ax.set_ylabel('DOS')
        ax.set_title(args.title.format(basename))
        plt.savefig(os.path.join(args.output, args.prefix + '_' + basename + '.' + args.format))
        plt.close()
    
