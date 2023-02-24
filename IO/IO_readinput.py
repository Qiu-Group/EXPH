

path = '../'

def readin(path):
    keywords = keyword_list()
    # (1) read lines
    file = open(path + 'exph.in')
    lines = file.readlines()
    # (2) get rid of '#', '\n' and ''
    lines = [x.strip() for x in lines]
    new_lines = {}
    for line in lines:
        if line == '' or line[0] == '#':
            continue
        else:

            if line.split('=')[0].strip() in keywords:

                if len(line.split('=')) == 2 :
                    pass
                else:
                    raise Exception(line.split()[0]+ ' can not be parsed')
                # new_lines.append(line.split()[0])
                # Three type input
                # (1) int
                # (2) path
                # (3) Bool
                if "'" in  line.split('=')[-1].strip():
                    new_lines[line.split('=')[0].strip()] = line.split('=')[-1].strip().strip("'")
                elif "[" in line.split('=')[-1].strip():
                    new_lines[line.split('=')[0].strip()] = list(map(int, line.split('=')[-1].strip().strip('[').strip(']').split(',')))
                elif line.split('=')[-1].strip() == 'True':
                    new_lines[line.split('=')[0].strip()] = True
                elif line.split('=')[-1].strip() == 'False':
                    new_lines[line.split('=')[0].strip()] = False
                else:
                    new_lines[line.split('=')[0].strip()] = float(line.split('=')[-1].strip())
            else:
                raise Exception(line + " is an unexpected keyword")
    return new_lines

# define parse function



def keyword_list():
    return ['valence_start_gkk','conduction_end_gkk','save_path','h5_path','T', 'degaussian',
            'initialize', 'xct_scattering_rate', 'xct_scattering_rate_Q_kmap', 'xct_scattering_rate_nS',
            'xct_scattering_rate_interpolation', 'xct_lifetime_all_BZ', 'xct_lifetime_all_BZ_nS',
            'xct_lifetime_all_BZ_interpolation', 'plot_xt_band', 'plot_xt_band_number',
            'plot_xt_interpolation', 'plot_ph_band', 'plot_ph_band_number', 'plot_ph_band_interpolation',
            'plot_xct_ph_mat', 'plot_xct_ph_mat_Q', 'plot_xct_ph_mat_n', 'plot_xct_ph_mat_m',
            'plot_xct_ph_mat_ph', 'plot_xct_ph_mat_interpolation', 'plot_xct_lifetime', 'plot_xct_lifetime_interposize_for_Lifetime',
            'plot_xct_data','exph_mat_write','nS_initial','nS_final'

            ,'Diffusion_PDE' , 'delta_degaussian_occupation', 'delta_T', 'T_total', 'delta_X', 'delta_Y', 'X', 'Y', 'initial_S_diffusion',
            'initial_Q_diffusion', 'initial_Gaussian_Broad', 'onGPU',
            'plot_diffusion', 'plot_diffusion_state', 'plot_diffusion_frame'
            ]