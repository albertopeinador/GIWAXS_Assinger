import re
#import matplotlib.pyplot as plt
import numpy as np
#from scipy.special import wofz
import streamlit as st
from scipy.optimize import linear_sum_assignment


#def pseudovoigt(x, a0, a1, a2, a3):
#    return a0 * ((1-a3) * np.exp(-np.log(2) * ( (x-a1) / a2) ** 2) + a3 / (1 + ((x-a1) / a2) ** 2))


def find_closest_value(list1, list2, target):
    # Find the index where list1's value is closest to the target
    closest_index = min(range(len(list1)), key=lambda i: abs(list1[i] - target))
    # Return the corresponding value from list2
    return list2[closest_index]

def compute_cost(dist_matrix, std_devs):
    # Convert distances to a cost matrix using a Gaussian distribution
    # We assume a squared distance here for simplicity and scaling
    cost_matrix = dist_matrix / std_devs
    return cost_matrix

def assign_peaks(dist_matrix, std_devs):
    # Compute the cost matrix
    cost_matrix = compute_cost(dist_matrix, std_devs)
    
    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return row_ind, col_ind


def pseudovoigt(x, height, center, hwhm, shape):
    return height*((1-shape)*np.exp(-np.log(2)*((x-center)/hwhm)**2)+shape/(1+((x-center)/hwhm)**2))


#def voigt(x, amp, center, sigma, gamma):
#    """
#    Calculate the Voigt profile.
#    
#    Parameters:
#    - x: array-like, independent variable (e.g., frequency or energy)
#    - center: float, center of the peak (mean of the Gaussian)
#    - sigma: float, standard deviation of the Gaussian component
#    - gamma: float, half-width at half-maximum (HWHM) of the Lorentzian component
#    
#    Returns:
#    - array-like, Voigt profile values
#    """
#    # Calculate the real and imaginary parts of z
#    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2.0))
#    
#    # Compute the Voigt function using wofz
#    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
def gauss(x, h, c, w):
    return h * np.exp(-(x-c)**2 / (2 * (w * 2/ 2.355)**2))


def fit_read(ruta):
    index = 0
    names = {}
    data = {}
    VARS = {}
    funcs = {}
    models = {}
    big_vars_idx = []
    weird_funcs = []
    #with open(ruta) as file:
    TITLE_PATTERN = r"'(.*?)'"
    DATA_PATTERN = r"X\[\d+\]=([\d.]+),\s*Y\[\d+\]=([\d.]+)"
    VAR_PATTERN = r'= ~?[-+]?\d*\.?\d+'
    VAR_INDEX = r"\$_(\d+)"
    FUNC_PATTERN = r"(\w+)\(([^)]+)\)"
    FUNC_INDEX = r"%_(\d+)"
    for strline in ruta:
        line = strline.decode('utf-8').strip()
        if line.startswith('plot'):
            break
        if line.startswith('use @'):
            m = re.search(r'\d+$', line)
            index = int(m.group())
            data[index] = []
            #funcs[index] = []
        if line.startswith('title = '):
            matches = re.findall(TITLE_PATTERN, line)
            names[index] = matches[0]
        if line.startswith('X['):
            match = re.search(DATA_PATTERN, line)
            data[index].append((float(match.group(1)), float(match.group(2))))
        if line.startswith('$_'):
            index_match = re.search(VAR_INDEX, line)
            match = re.search(VAR_PATTERN, line)
            value = match.group(0).split('= ')[1]
            var_value = float(value[1:]) if value.startswith('~') else float(value)
            VARS[int(index_match.group(1))] = var_value
            if var_value > 1e2:
                big_vars_idx.append(int(index_match.group(1)))
                warning_txt = 'WARNING!!!! : Large variable found! - Variable ' + index_match.group(1) + ' = ' + str(var_value) 
                st.warning(warning_txt)
        if line.startswith('%_'):
            index_match = re.search(FUNC_INDEX, line)
            match = re.search(FUNC_PATTERN, line)
            func_name = [match.group(1)]
            integers_inside = re.findall(r"\d+", match.group(2))
            integers_inside = [int(num) for num in integers_inside]
            if any(item in integers_inside for item in big_vars_idx):
                weird_funcs.append(int(index_match.group(1)))
            funcs[int(index_match.group(1))] = func_name+integers_inside
        if line.startswith('@') and line[1].isdigit():
            model = re.findall(FUNC_INDEX, line)
            models[int(line[1])] = [int(j) for j in model]
            bad_model_check = [int(j) for j in model if int(j) in weird_funcs]
            if bad_model_check:
                warn_text = 'Bad Model: ' + names[int(line[1])]
                st.warning(warn_text)
    return names, data, VARS, funcs, models


#def plots(ruta, list_of_plots):
#    names, data, VARS, funcs, models = fit_read(ruta)
#    filtered_dict = {key: value for key, value in models.items() if key % 2 == 0}
#    filtered_names = {key: value for key, value in names.items() if key % 2 == 0}
#    #filter_names = [1, 3, 4]
#    if 'data' in list_of_plots:
#        for i in filtered_names:
#            X = [data[i][j][0] for j in range(len(data[i]))]
#            Y = [data[i][j][1] for j in range(len(data[i]))]
#            plt.plot(X, Y, '.', markersize = 3)
#            plt.xlim(0, 20)
#    if 'models' in list_of_plots:
#
#        x = np.linspace(0, 20, 500)
#        for model in filtered_dict:
#
#            mod = np.zeros(500)
#            for f in models[model]:
#
#                if funcs[f][0] == 'PseudoVoigt':
#                    mod += pseudovoigt(x, VARS[funcs[f][1]], VARS[funcs[f][2]], VARS[funcs[f][3]], VARS[funcs[f][4]])
#                if funcs[f][0] == 'Linear':
#                    mod += VARS[funcs[f][1]] + VARS[funcs[f][2]] * x
#                if funcs[f][0] == 'ExpDecay':
#                    mod += VARS[funcs[f][1]] * np.exp(-x / VARS[funcs[f][2]])
#                if funcs[f][0] == 'Gaussian':
#                    mod += gauss(x, VARS[funcs[f][1]], VARS[funcs[f][2]], VARS[funcs[f][3]])
#            plt.plot(x, mod)
#    plt.ylim((-0.5, 1))
#
#    plt.show()

def calcmodel(model, funcs, VARS):
    x = np.linspace(2, 20, 500)
    mod = np.zeros(500)
    for f in model:
        if funcs[f][0] == 'PseudoVoigt':
            mod += pseudovoigt(x, VARS[funcs[f][1]], VARS[funcs[f][2]], VARS[funcs[f][3]], VARS[funcs[f][4]])
        if funcs[f][0] == 'Linear':
            mod += VARS[funcs[f][1]] + VARS[funcs[f][2]] * x
        if funcs[f][0] == 'ExpDecay':
            mod += VARS[funcs[f][1]] * np.exp(-x / VARS[funcs[f][2]])
        if funcs[f][0] == 'Gaussian':
            mod += gauss(x, VARS[funcs[f][1]], VARS[funcs[f][2]], VARS[funcs[f][3]])
    return x, mod

def minus_asign(peaks, expected, s_expected = 0):
    '''distance = np.array([abs(q - left_q[exp_peak]) for exp_peak in left_q])
    closest_arg = distance.argmin()
    min_dist = distance.min()
    closest_peak = list(left_q.keys())[closest_arg]
    asign_d = left_s_q[closest_peak] * modifyer
    if min_dist <= asign_d and closest_arg in asigned:
        asigned[closest_arg] = (q, min_dist)
        return closest_arg, left.pop[q]
    elif min_dist >= asign_d and 'pseudo' + closest_arg not in asigned:
        asigned['pseudo' + closest_arg] = (q, min_dist)
        return 'pseudo' + closest_arg, left.pop[q]
    elif min_dist <= asign_d and closest_arg in asigned:
        warnings.warn(f'Two good asignations for plane {closest_arg}')
        if min_dist < asigned[closest_peak][1]:
            WORK IN PROGRESS'''
    #dist_mat = np.array()
    dist_mat = np.array([abs(peaks[0][1] - exp_peak) for exp_peak in expected])
    for peak in peaks[1:]:
        dist_mat = np.vstack([dist_mat, [abs(peak[1] - exp_peak) for exp_peak in expected]])
    row_ind, col_ind = assign_peaks(dist_mat, s_expected)
    assignation = {}
    #print("Optimal assignments:")
    for r, c in zip(row_ind, col_ind):
        #print(f"Real peak {r} is assigned to theoretical peak {c}")
        assignation[c] = r
    return assignation
#peaks = [[0.164653420641, 3.19965106851, 0.186057092487, 0.0], [0.0998563769559, 3.7574685947, 0.607934927211, 0.543387781314], [0.019024074596, 5.75209404892, 0.750384339279, 0.0]]
#expected = [3.196801868774,3.7397033025380004,11.32948957296,11.335287537240001]
#s_exp = [0.03786248821854892, 0.18356459828109675, 5.819486781334248, 5.571924196913774]
#
#minus_asign(peaks, expected, s_exp)


def get_stats (asigned_peaks):
    mean_q = {}
    mean_h = {}
    mean_w = {}
    s_q = {}
    s_h = {}
    s_w = {}
    params = {}
    for sample in asigned_peaks:
        for peak in asigned_peaks[sample]:
            if peak not in params:
                params[peak] = {'q': [asigned_peaks[sample][peak][1]], 'h': [asigned_peaks[sample][peak][0]], 'w': [asigned_peaks[sample][peak][2]]}
            else:
                params[peak]['q'].append(asigned_peaks[sample][peak][1])
                params[peak]['h'].append(asigned_peaks[sample][peak][0])
                params[peak]['w'].append(asigned_peaks[sample][peak][2])
    for peak in params:
        mean_q[peak] = np.mean(params[peak]['q'])
        s_q [peak] = np.std(params[peak]['q'])
        mean_h[peak] = np.mean(params[peak]['h'])
        s_h [peak] = np.std(params[peak]['h'])
        mean_w[peak] = np.mean(params[peak]['w'])
        s_w [peak] = np.std(params[peak]['w'])
    return mean_q, s_q, mean_h, s_h, mean_w, s_w

def asigner(ruta, list_of_peaks, assign_extra_peaks, list_of_functions):
    #modifyer = 1. / asign_strength if asign_strength != 0 else float('inf')
    names, data, VARS, funcs, models = fit_read(ruta)
    NUM_OF_EXPEC_PEAKS = len(list_of_peaks)
    NUM_OF_PEAKS = {}
    PEAKS = {}
    asigned_peaks = {}
    for model in models:
        PEAKS [model] = []
        NUM_OF_PEAKS [model] = 0
        for f in models[model]:
            if funcs[f][0] in list_of_functions:
                NUM_OF_PEAKS [model] += 1
                PEAKS[model].append(funcs[f])
        if NUM_OF_PEAKS [model] == NUM_OF_EXPEC_PEAKS:
            if asigned_peaks == {}:
                print('first assignation:', names[model])
            asigned_peaks [names[model]] = {}
            qs = []
            varias = list(PEAKS[model][val][1:] for val, vals in enumerate(PEAKS[model]))#, key = VARS[PEAKS[model][val][1:]]))
            #print(varias)
            sorted_PEAKS = []
            for pointer, peak in enumerate(list_of_peaks):
                qs.append(VARS[PEAKS[model][pointer][2]])
            
            sorted_PEAKS.append(list(x for _,x in sorted(zip(qs,varias))))
            #print(sorted_PEAKS)
            for pointer, peak in enumerate(list_of_peaks):
            #    print(VARS[PEAKS[model][pointer][2]])
                #print(pointer)
                #print(sorted_PEAKS[0][pointer][:])
                #sorte = sorted(PEAKS[model][pointer][1:], key=[VARS[val][1] for val in PEAKS[model][pointer][1:]])#sorted([VARS[val][1] for val in PEAKS[model][pointer][1:]])
                #print(sorted_PEAKS[pointer][1:][0])
                asigned_peaks [names[model]][peak] = [VARS[val] for val in sorted_PEAKS[0][pointer][:]]#PEAKS[model][pointer][1:]]

    
    mean_q, s_q, mean_h, s_h, mean_w, s_w = get_stats(asigned_peaks)

    for model in models:
        if NUM_OF_PEAKS[model] < NUM_OF_EXPEC_PEAKS:
            asigned_peaks [names[model]] = {}
            '''for pointer, peak in enumerate(PEAKS[model]):

                current_q = VARS[peak[2]]

                distance = np.array([abs(current_q - mean_q[exp_peak]) for exp_peak in mean_q])

                closest_arg = distance.argmin()
                min_dist = distance.min()
                closest_peak = list(mean_q.keys())[closest_arg]

                asign_d = s_q[closest_peak] * modifyer

                if min_dist <= asign_d and closest_peak not in asigned_peaks [names[model]]:
                    asigned_peaks [names[model]][closest_peak] = [VARS[val] for val in PEAKS[model][pointer][1:]]
                    mean_q, s_q, mean_h, s_h, mean_w, s_w = get_stats(asigned_peaks)

                elif min_dist >= asign_d and 'pseudo' + closest_peak not in asigned_peaks [names[model]]:
                    asigned_peaks [names[model]]['pseudo' + closest_peak] = [VARS[val] for val in PEAKS[model][pointer][1:]]
                elif closest_peak or 'pseudo' + closest_peak in asigned_peaks [names[model]]:
                    warnings.warn('Attempted to asign two peaks to the same plane')
                    if min_dist < abs(asigned_peaks [names[model]][closest_peak][1] - mean_q[closest_peak]) and min_dist <= asign_d:
                        removed = asigned_peaks [names[model]][closest_peak].copy()
                        asigned_peaks [names[model]][closest_peak] = [VARS[val] for val in PEAKS[model][pointer][1:]]
                        #reasign(removed, closest_peak)
                    elif min_dist < abs(asigned_peaks [names[model]][closest_peak][1] - mean_q[closest_peak]) and min_dist >= asign_d:
                        asigned_peaks [names[model]]['pseudo' + closest_peak] = [VARS[val] for val in PEAKS[model][pointer][1:]]    
            '''
            peaks = []
            for peak in PEAKS[model]:
                peaks.append([VARS[f] for f in peak[1:]])#PEAKS[model]
            expected = [q for q in mean_q.values()]
            s_exp = [sq for sq in s_q.values()]
            assignation = minus_asign(peaks, expected, s_exp)
            for asigned in assignation:
                asigned_peaks [names[model]][list(mean_q.keys())[asigned]] = peaks[assignation[asigned]]

        #if NUM_OF_EXPEC_PEAKS < NUM_OF_PEAKS:
        #    for ex_peak in list_of_peaks:
        #        for 
        if NUM_OF_EXPEC_PEAKS < NUM_OF_PEAKS[model]:
            asigned_peaks [names[model]] = {}
            exp_peaks = [q for q in mean_q.values()]
            peaks = []
            s_exp = [sq for sq in s_q.values()]
            for peak in PEAKS[model]:
                peaks.append([VARS[f] for f in peak[1:]])
            assignation = minus_asign(peaks, exp_peaks, s_exp)
            for asigned in assignation:
                asigned_peaks [names[model]][list(mean_q.keys())[asigned]] = peaks[assignation[asigned]]
            for idx, peak in enumerate(PEAKS[model]):
                vas = [VARS[val] for val in peak[1:]]
                if vas not in asigned_peaks[names[model]].values() and assign_extra_peaks:
                    asigned_peaks [names[model]][f'unknonw_{idx}'] = vas
    return names, asigned_peaks, data, models, funcs, VARS




def write_csv(asigned_peaks):
    outstring = ''
    for model in asigned_peaks:
        q_head = ','.join([f'q_{peak}' for peak in asigned_peaks[model]])
        A_head = ','.join([f'A_{peak}' for peak in asigned_peaks[model]])
        w_head = ','.join([f'FWHM_{peak}' for peak in asigned_peaks[model]])
        header = ',' + q_head + ',' + A_head + ',' + w_head + '\n'
        
        outstring += header
        qs = ','.join([str(asigned_peaks[model][peak][1]) for peak in asigned_peaks[model]])
        try:
            As = ','.join([str(asigned_peaks[model][peak][0] * asigned_peaks[model][peak][2] * ((1 - asigned_peaks[model][peak][3]) * np.sqrt(np.pi / np.log(2)) + asigned_peaks[model][peak][3] * np.pi)) for peak in asigned_peaks[model]])
        except IndexError:
            eta = 0
            As = ','.join([str(asigned_peaks[model][peak][0] * asigned_peaks[model][peak][2] * ((1 - eta) * np.sqrt(np.pi / np.log(2)) + eta * np.pi)) for peak in asigned_peaks[model]])

        ws = ','.join([str(asigned_peaks[model][peak][2] * 2) for peak in asigned_peaks[model]])
        outstring += model + ','+ qs + ',' + As + ',' + ws + '\n'
    return outstring

#print(asigner('/Users/albpeivei/Documents/Fits_nuevos/BTP_ip.fit', ['a1', 'a2', 'a3', 'a4'], True))