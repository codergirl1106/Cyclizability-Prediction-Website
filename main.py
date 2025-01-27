import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import io
import time
from tensorflow import keras

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import re
import py3Dmol
from stmol import showmol
import requests
import streamlit_scrollable_textbox as stx
import streamlit_js_eval
import functools
from itertools import chain
import string


swidth = streamlit_js_eval.streamlit_js_eval(js_expressions='screen.width', want_output = True, key = 'SCR')
try:
    swidth = int(swidth)
except TypeError:
    swidth = 1100

def my_cache(f):
    @st.cache_data(max_entries=5, ttl=600)
    @functools.wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)
    return inner

@st.cache_data
def load_model(modelnum: int):
    return keras.models.load_model(f"./adapter-free-Model/C{modelnum}free")

def pred(model, pool):
    input = np.zeros((len(pool), 200), dtype = np.single)
    temp = {'A':0, 'T':1, 'G':2, 'C':3}
    for i in range(len(pool)): 
        for j in range(50):
            input[i][j*4 + temp[pool[i][j]]] = 1
    A = model.predict(input, batch_size=128)
    A.resize((len(pool),))
    return A

def getFasta(pbdid): 
    sequencelink = f"https://www.rcsb.org/fasta/entry/{pbdid}"
    tt = requests.get(sequencelink).text
    return tt

def getCif(pdbid):
    link = f'https://files.rcsb.org/download/{pdbid}.cif'
    text = requests.get(link).text
    return text

def readCif(text, chains):
    out = []
    for line in text.split('\n'):
        l = line.split()
        if len(l) > 0:
            if l[0] == "ATOM":
                if l[6] in chains and l[5] in {'DA', 'DG', 'DT', 'DC'}:
                    out.append([l[3], l[5],l[6],l[8],l[10],l[11],l[12]]) # C8/C6, DA/DG/DT/DC, chain, seq_index, x, y, z

    return out

def readPDB(text, chains):
    out = []
    for line in text.split('\n'):
        l = line.split()
        if len(l) > 0:
            if l[0] == "ATOM":
                if l[4] in chains and l[3] in {'DA', 'DG', 'DT', 'DC'}:
                    out.append([l[2], l[3], l[4], l[5], l[6], l[7], l[8]])
    
    return out

def processFasta(alist):
    sequences = []
    current_sequence = ""
    for i in alist[1:]:
        if i[0] == ">":
            sequences.append(current_sequence)
            current_sequence = ""
        else:
            current_sequence += i
            
    if current_sequence != "":
        sequences.append(current_sequence)

    return sequences

def getSequence(fasta, text, ciforpdb):
    seq_and_chains = []
    fasta = fasta.split('\n')[:-1]
    for i in range(0, len(fasta), 2):
        seq_and_chains.append([re.findall(r'([A-Z]+)(?:\[auth [^\]]+\])?[,|]', fasta[i]), fasta[i+1]])
        
    seq_and_chains.sort()
    sequences = []
    cords = []

    qqq = dict()
    sqq = dict()
    for i in seq_and_chains:
        chains, seq = i[0], i[1]

        if ciforpdb == "cif":
            stuff = readCif(text, chains)
        else:
            stuff = readPDB(text, chains)

        if set(seq).issubset({'A', 'C', 'G', 'T', 'N', 'X'}):
            for line in stuff:
                if line[1] in ['DA', 'DG'] and line[0] == 'C8':
                    if line[2] in qqq:
                        qqq[line[2]].append([float(line[_]) for _ in range(4, 7)])
                    else:
                        qqq.update({line[2]: [[float(line[_]) for _ in range(4, 7)]]})
                        seqcords = [int(i[3]) for i in stuff if i[2] == line[2]]
                        sqq[line[2]] = [seq, min(seqcords)-1,max(seqcords)]
                    
                if line[1] in ['DT', 'DC'] and line[0] == 'C6':
                    if line[2] in qqq:
                        qqq[line[2]].append([float(line[_]) for _ in range(4, 7)])
                    else:
                        qqq.update({line[2]: [[float(line[_]) for _ in range(4, 7)]]})
                        seqcords = [int(i[3]) for i in stuff if i[2] == line[2]]
                        sqq[line[2]] = [seq, min(seqcords)-1,max(seqcords)]

    if len(qqq) % 2 == 0:
        while True:
            keyd = sorted(qqq.keys())

            if len(keyd) == 0:
                break

            if sqq[keyd[0]][1] <= 0:
                sqq[keyd[0]][2] += 1-sqq[keyd[0]][1]
                sqq[keyd[0]][1] += 1-sqq[keyd[0]][1]

            if sqq[keyd[1]][1] <= 0:
                sqq[keyd[1]][2] += 1-sqq[keyd[1]][1]
                sqq[keyd[1]][1] += 1-sqq[keyd[1]][1]
                
            startindex = max(len(sqq[keyd[1]][0])-sqq[keyd[1]][2], sqq[keyd[0]][1])
            endindex = min(len(sqq[keyd[1]][0])-sqq[keyd[1]][1], sqq[keyd[0]][2])

            sequences.append(sqq[keyd[0]][0][startindex:endindex])

            qqq[keyd[0]] = qqq[keyd[0]][startindex-sqq[keyd[0]][1]:endindex-startindex+startindex-sqq[keyd[0]][1]]+qqq[keyd[1]][len(sqq[keyd[1]][0])-endindex-sqq[keyd[1]][1]:endindex-startindex+len(sqq[keyd[1]][0])-endindex-sqq[keyd[1]][1]]
            del qqq[keyd[1]]
            qwer, asdf = np.array(qqq[keyd[0]][:len(qqq[keyd[0]])//2], dtype = np.single), np.array(qqq[keyd[0]][-(len(qqq[keyd[0]])//2):][::-1], dtype = np.single)

            del qqq[keyd[0]]
            otemp = (qwer+asdf)/2
            o = np.array([(otemp[i+1]+otemp[i])/2 for i in range(len(otemp)-1)], dtype = np.single)

            ytemp = qwer - asdf
            z = np.array([otemp[i+1]-otemp[i] for i in range(len(otemp)-1)], dtype = np.single)
            ytemp = [(ytemp[i+1]+ytemp[i])/2 for i in range(len(ytemp)-1)]
            x = np.array([np.cross(ytemp[i], z[i]) for i in range(len(z))], dtype = np.single)
            y = np.array([np.cross(z[i], x[i]) for i in range(len(z))], dtype = np.single)
            x = -np.array([x[i]/np.linalg.norm(x[i]) for i in range(len(x))], dtype = np.single) # direction of minor groove
            y = np.array([y[i]/np.linalg.norm(y[i]) for i in range(len(y))], dtype = np.single)
            cords.append((o, x, y, z))
    elif len(qqq) == 3:
        main_seq = sorted(sqq.items(), key=lambda x: len(x[1]))[-1][0]
        for i in qqq:
            if i != main_seq:
                sequences.append(sqq[i])
                qwer, asdf = np.array(qqq[i][:len(qqq[i])//2], dtype = np.single), np.array(qqq[i][-(len(qqq[i])//2):][::-1], dtype = np.single)
                otemp = (qwer+asdf)/2
                o = np.array([(otemp[i+1]+otemp[i])/2 for i in range(len(otemp)-1)], dtype = np.single)
                ytemp = qwer - asdf
                z = np.array([otemp[i+1]-otemp[i] for i in range(len(otemp)-1)], dtype = np.single)
                ytemp = [(ytemp[i+1]+ytemp[i])/2 for i in range(len(ytemp)-1)]
                x = np.array([np.cross(ytemp[i], z[i]) for i in range(len(z))], dtype = np.single)
                y = np.array([np.cross(z[i], x[i]) for i in range(len(z))], dtype = np.single)
                x = -np.array([x[i]/np.linalg.norm(x[i]) for i in range(len(x))], dtype = np.single) # direction of minor groove
                y = np.array([y[i]/np.linalg.norm(y[i]) for i in range(len(y))], dtype = np.single)
                cords.append((o, x, y, z))
    else:
        for i in qqq:
            sequences.append(sqq[i])
            qwer, asdf = np.array(qqq[i][:len(qqq[i])//2], dtype = np.single), np.array(qqq[i][-(len(qqq[i])//2):][::-1], dtype = np.single)
            otemp = (qwer+asdf)/2
            o = np.array([(otemp[i+1]+otemp[i])/2 for i in range(len(otemp)-1)], dtype = np.single)
            ytemp = qwer - asdf
            z = np.array([otemp[i+1]-otemp[i] for i in range(len(otemp)-1)], dtype = np.single)
            ytemp = [(ytemp[i+1]+ytemp[i])/2 for i in range(len(ytemp)-1)]
            x = np.array([np.cross(ytemp[i], z[i]) for i in range(len(z))], dtype = np.single)
            y = np.array([np.cross(z[i], x[i]) for i in range(len(z))], dtype = np.single)
            x = -np.array([x[i]/np.linalg.norm(x[i]) for i in range(len(x))], dtype = np.single) # direction of minor groove
            y = np.array([y[i]/np.linalg.norm(y[i]) for i in range(len(y))], dtype = np.single)
            cords.append((o, x, y, z))

    return sequences, cords

def envelope2(x, y):
    x, y = list(x), list(y)
    uidx, ux, uy = [0], [x[0]], [y[0]]
    lidx, lx, ly = [0], [x[0]], [y[0]]

    # local extremas
    for i in range(1, len(x)-1):
        if (y[i] == max(y[max(0, i-3):min(i+4, len(y))])):
            uidx.append(i)
            ux.append(x[i])
            uy.append(y[i])
        if (y[i] == min(y[max(0, i-3):min(i+4, len(y))])):
            lidx.append(i)
            lx.append(x[i])
            ly.append(y[i])

    uidx.append(len(x)-1)
    ux.append(x[-1])
    uy.append(y[-1])
    lidx.append(len(x)-1)
    lx.append(x[-1])
    ly.append(y[-1])

    ubf = interp1d(ux, uy, kind=3, bounds_error=False)
    lbf = interp1d(lx, ly, kind=3, bounds_error=False)
    ub = np.array([y, ubf(x)]).max(axis=0)
    lb = np.array([y, lbf(x)]).min(axis=0)

    return ub, lb, ub-lb

def envelope1(fity):
    ux, uy = [0], [fity[0]]
    lx, ly = [0], [fity[0]]

    # local extremas
    for i in range(1, len(fity)-1):
        if (fity[i] == max(fity[max(0, i-3):min(i+4, len(fity))])):
            ux.append(i)
            uy.append(fity[i])
        if (fity[i] == min(fity[max(0, i-3):min(i+4, len(fity))])):
            lx.append(i)
            ly.append(fity[i])

    ux.append(len(fity)-1)
    uy.append(fity[-1])
    lx.append(len(fity)-1)
    ly.append(fity[-1])

    ub = np.array([fity, interp1d(ux, uy, kind=3, bounds_error=False)(range(len(fity)))]).max(axis=0)
    lb = np.array([fity, interp1d(lx, ly, kind=3, bounds_error=False)(range(len(fity)))]).min(axis=0)
    return ub-lb

def trig(x, *args): # x = [C0, amp, psi]
    return [args[0][0] - x[0] - x[1]**2*math.cos((34.5/args[0][-1]-3)*2*math.pi-math.pi*2/3 - x[2]),
            args[0][1] - x[0] - x[1]**2*math.cos((31.5/args[0][-1]-3)*2*math.pi-math.pi*2/3 - x[2]),
            args[0][2] - x[0] - x[1]**2*math.cos((29.5/args[0][-1]-2)*2*math.pi-math.pi*2/3 - x[2])]

def show_st_3dmol(pdb_code,original_pdb,cartoon_style="oval",
                  cartoon_radius=0.2,cartoon_color="lightgray",zoom=1,spin_on=False):

    if swidth >= 1000:
        view = py3Dmol.view(width=int(swidth/2), height=int(swidth/3))
    else:
        view = py3Dmol.view(width=int(swidth), height=int(swidth))

    view.addModelsAsFrames(pdb_code)
    view.addModelsAsFrames(original_pdb)
    view.setStyle({"cartoon": {"style": cartoon_style,"color": cartoon_color,"thickness": cartoon_radius}})

    style_lst = []
    surface_lst = []

    view.addStyle({'chain':'X'}, {'stick': {"color": "blue"}})

    for i in string.ascii_uppercase:
        view.addStyle({'chain':i}, {'line': {}})

    view.zoomTo()
    view.spin(spin_on)
    view.zoom(zoom)

    if swidth >= 1000:
        showmol(view, height=int(swidth/3), width=int(swidth/2))
    else:
        showmol(view, height=int(swidth), width=int(swidth))
    return 0

def helpercode(model_num: int, seqlist: dict, pool, sequence):
    prediction = pred(load_model(model_num), tuple(seqlist.keys()))

    result_array = np.zeros((len(pool), len(pool[0]) - 49), dtype = np.single)

    for i in range(len(pool)):
        for j in range(49):
            result_array[i, j] = prediction[seqlist[pool[i][j:j + 50]]]
        for j in range(len(pool[i])-98, len(pool[i])-49):
            result_array[i, j] = prediction[seqlist[pool[i][j:j + 50]]]

    result_array = result_array.mean(axis=0)
    if len(sequence) >= 50:
        seqlist = [sequence[i:i+50] for i in range(len(sequence)-49)]
        result_array[49:-49] = pred(load_model(model_num), seqlist).reshape(-1, )

    result_array -= result_array.mean()
    return result_array

def pdb_out(psi, amp, factor, cords):
    counterhetatm = 0
    counterconect = 0

    pdb_final_output = "HEADER    output from spatial analysis\n"
    hetatmf = ''
    conectf = ''
    for i in range(len(cords)):
        arrow = []
        for j in range(len(cords[i][1])):
            arrow.append(np.cos(-psi[i][j])*cords[i][1][j] + np.sin(-psi[i][j])*cords[i][2][j])

        arrow = np.array(arrow)

        for j in range(len(arrow)):
            arrow[j] /= np.linalg.norm(arrow[j])
            arrow[j] *= factor*amp[i][j]

        e = cords[i][0] + arrow
        o = np.around(cords[i][0], 3)
        e = np.around(e, 3)

        hetatm = ''
        conect = ''
        for j in range(len(o)):
            hetatm += 'HETATM' + str(counterhetatm+j+1).rjust(5) + ' C    AXI ' + f'X{i+1}' +' '*(3-len(str(i+1)))+'1    ' # '    1' 5d
            for k in range(3):
                hetatm += str(o[j][k]).rjust(8)
            hetatm += '\n'
        for j in range(len(e)):
            hetatm += 'HETATM' + str(counterhetatm+j+1+len(o)).rjust(5) + ' C    AXI ' + f'X{i+1}' +' '*(3-len(str(i+1)))+'1    ' # '    1' 5d
            for k in range(3):
                hetatm += str(round(e[j][k], 2)).rjust(8)
            hetatm += '\n'
        for j in range(len(o)-1):
            conect += 'CONECT' + str(counterconect+j+1).rjust(5) + str(counterconect+j+2).rjust(5) + '\n' # CONECT    1    2
        for j in range(len(o)):
            conect += 'CONECT' + str(counterconect+j+1).rjust(5) + str(counterconect+j+1+len(o)).rjust(5) + '\n' # CONECT    1    2

        counterhetatm += 2*len(o)
        counterconect += 2*len(o)
        hetatmf += hetatm
        conectf += conect

    return pdb_final_output + hetatmf + conectf

def spatial(seq):
    def func(x): # x = [C0, amp, psi]
        return [c26_ - x[0] - x[1]**2*math.cos((34.5/10.3-3)*2*math.pi-math.pi*2/3 - x[2]),
                c29_ - x[0] - x[1]**2*math.cos((31.5/10.3-3)*2*math.pi-math.pi*2/3 - x[2]),
                c31_ - x[0] - x[1]**2*math.cos((29.5/10.3-2)*2*math.pi-math.pi*2/3 - x[2])]

    base = ['A','T','G','C']
    left = ''.join([random.choice(base) for i in range(49)])
    right = ''.join([random.choice(base) for i in range(49)])
    seq = left + seq + right
    seqlist = [seq[i:i+50] for i in range(len(seq)-50+1)]

    zxcv26 = pred(load_model(26), seqlist)
    zxcv29 = pred(load_model(29), seqlist)
    zxcv31 = pred(load_model(31), seqlist)
    zxcv26 -= zxcv26.mean()
    zxcv29 -= zxcv29.mean()
    zxcv31 -= zxcv31.mean()

    u26, l26, caf26 = envelope2(np.arange(len(zxcv26)), zxcv26)
    u29, l29, caf29 = envelope2(np.arange(len(zxcv29)), zxcv29)
    u31, l31, caf31 = envelope2(np.arange(len(zxcv31)), zxcv31)
    amp = (caf26 + caf29 + caf31)/3
    psi = []
    for i in range(len(amp)):
        c26_, c29_, c31_ = zxcv26[i], zxcv29[i], zxcv31[i]
        root = fsolve(func, [1, 1, 1])
        psi.append(root[2])
        if(psi[-1] > math.pi): psi[-1] -= 2*math.pi
    psi = np.array(psi)
    amp = amp[25:-25]
    psi = psi[25:-25]
    return amp, psi

def similarity(seq, cord):
    amp, psi = spatial(seq)

    arrow = []
    for j in range(len(cord[1])): arrow.append(np.cos(-psi[j])*cord[1][j] + np.sin(-psi[j])*cord[2][j])
    arrow = np.array(arrow)
    for j in range(len(arrow)):
        arrow[j] /= np.linalg.norm(arrow[j])
        arrow[j] *= amp[j]

    similar = []
    for i in range(len(arrow)):
        _1, _2 = max(0, i-24), min(len(arrow), i+24+1)
        similar.append(np.inner(cord[3][_2-1]-cord[3][_1], arrow[i]))
    return np.array(similar)

@my_cache
def longcode(sequence, helical_turn):
    pool = []
    base = ['A','T','G','C']

    for i in range(200):
        left = ''.join([random.choice(base) for i in range(49)])
        right = ''.join([random.choice(base) for i in range(49)])
        pool.append(left + sequence + right)

    seqlist = dict()
    indext = 0
    for i in range(len(pool)):
        for j in range(len(pool[i])-49):
            tt = pool[i][j:j+50]
            if tt not in seqlist:
                seqlist.update({tt: indext})
                indext += 1

    models = dict.fromkeys((26, 29, 31))
    for modelnum in models.keys():
        models[modelnum] = helpercode(modelnum, seqlist, pool, sequence)

    amp = sum(envelope1(m) for m in models.values()) / len(models)

    psi = []
    for i in range(len(amp)):
        root = fsolve(trig, [1, 1, 1], args=[m[i] for m in models.values()]+[helical_turn])
        psi.append(root[2])
        if(psi[-1] > math.pi): psi[-1] -= 2*math.pi

    psi = np.array(psi, dtype = np.single)

    amp,psi = amp[25:-25],psi[25:-25]
    return amp, psi

def spatial_analysis_ui(imgg, sequence, texttt, cords):
    st.markdown("***")
    st.header(f"Spatial Visualization")

    factor = st.text_input('vector length scale factor','30')
    try:
        factor = int(factor)
    except:
        factor = 30
    helical_turn = st.text_input('bp/helical turn','10.3')
    try:
        helical_turn = float(helical_turn)
    except:
        helical_turn = 10.3

    # remove reverse complementary strands

    amp, psi, sim = [], [], []
    for seq in range(len(sequence)):
        a, p = longcode(sequence[seq], helical_turn)
        amp.append(a)
        psi.append(p)
        sim.append(similarity(sequence[seq], cords[seq]))

    pdb_output = pdb_out(psi, amp, factor, cords)

    file_nameu = st.text_input('file name', 'spatial_visualization.pdb')
    show_st_3dmol(pdb_output, texttt)
    st.download_button('Download .pdb', pdb_output, file_name=f"{file_nameu}")

    st.markdown("***")

    for v in range(len(amp)):
        st.header(f"Sequence {v+1}")
        figg, axx = plt.subplots()
        figgg, axxx = plt.subplots()
        figggg, axxxx = plt.subplots()

        plt.figure(figsize=(10, 3))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        axx.spines[['right', 'top']].set_visible(False)
        axxx.spines[['right', 'top']].set_visible(False)
        axxxx.spines[['right', 'top']].set_visible(False)

        axx.plot(amp[v], color='black')
        axxx.plot(psi[v], color='black')
        axxxx.plot(sim[v], color='black')

        col1, col2, col3 = st.columns([0.333, 0.333, 0.333])
        with col1:
            st.subheader(f"Amplitude Graph")
            st.pyplot(figg)

            filetype = st.selectbox(f'amplitude graph {v+1} file type', ('svg', 'png', 'jpeg'))

            figg.savefig(imgg, format=filetype)
            file_name3 = st.text_input('file name', f'amplitude_graph{v+1}.{filetype}')
            btn3 = st.download_button(label="Download graph",data=imgg,file_name=f"{file_name3}",mime=f"image/{filetype}")

        with col2:  
            st.subheader(f"Phase Graph")
            st.pyplot(figgg)

            filetype2 = st.selectbox(f'phase graph {v+1} file type', ('svg', 'png', 'jpeg'))

            figgg.savefig(imgg, format=filetype2)
            file_name4 = st.text_input('file name', f'phase_graph{v+1}.{filetype2}')
            btn4 = st.download_button(label="Download graph",data=imgg,file_name=f"{file_name4}",mime=f"image/{filetype2}")

        with col3:
            st.subheader(f"Similarity Graph")
            st.pyplot(figggg)

            filetype3 = st.selectbox(f'similarity graph {v+1} file type', ('svg', 'png', 'jpeg'))

            figgg.savefig(imgg, format=filetype3)
            file_name5 = st.text_input('file name', f'similarity_graph{v+1}.{filetype3}')
            btn5 = st.download_button(label="Download graph",data=imgg,file_name=f"{file_name5}",mime=f"image/{filetype3}")

        st.markdown("***")

    st.header(f"Amplitude Graph Data")
    long_text11 = "\n".join([','.join(map(lambda x: format(x, '.4f'), i)) for i in amp])

    file_name11 = st.text_input('file name', f'amplitude_data.txt')

    st.download_button('Download data', long_text11, file_name=f"{file_name11}")

    st.markdown("data format w/ one sequence per line")
    stx.scrollableTextbox(long_text11, height = 300)

    st.markdown("***")

    st.header(f"Phase Graph Data")

    long_text22 = "\n".join([','.join(map(lambda x: format(x, '.4f'), i)) for i in psi])

    file_name22 = st.text_input('file name', f'phase_data.txt')

    st.download_button('Download data', long_text22, file_name=f"{file_name22}")

    st.markdown("data format w/ one sequence per line")
    stx.scrollableTextbox(long_text22, height = 300)

    st.header(f"Similarity Graph Data")

    long_text33 = "\n".join([','.join(map(lambda x: format(x, '.4f'), i)) for i in sim])

    file_name33 = st.text_input('file name', f'similarity_data.txt')

    st.download_button('Download data', long_text33, file_name=f"{file_name33}")

    st.markdown("data format w/ one sequence per line")
    stx.scrollableTextbox(long_text33, height = 300)

def sequence_ui(imgg, seqs, option):
    modelnum = int(re.findall(r'\d+', option)[0])

    cNfree_predictions = []
    count = 1

    for seq in seqs:
        if len(seq) >= 50:
            list50 = [seq[i:i+50] for i in range(len(seq)-50+1)]

            cNfree = list(pred(load_model(modelnum), list50))
            cNfree_predictions.append(cNfree)

            fig, ax = plt.subplots()
            plt.figure(figsize=(10, 3))
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            ax.spines[['right', 'top']].set_visible(False)

            ax.plot(cNfree, color='black')

            st.markdown("***")
            st.header(f"C{modelnum}corr prediction for Sequence {count}")
            filetype5 = st.selectbox(f'C{modelnum}corr sequence {count} file type', ('png', 'svg', 'jpeg'))

            fig.savefig(imgg, format=filetype5)

            file_name1 = st.text_input(f'C{modelnum}corr sequence {count} file name', f'C{modelnum}corr_sequence{count}_prediction.{filetype5}')
            btn = st.download_button(
                label="Download graph",
                data=imgg,
                file_name=f"{file_name1}",
                mime=f"image/{filetype5}"
            )

            st.pyplot(fig)
        else:
            st.markdown("***")
            st.header(f"C{modelnum}corr prediction for Sequence {count}")
            st.markdown(f"Sequence < 50bp; C{modelnum}corr prediction cannot be run on this sequence.")

        count += 1

    st.markdown("***")

    st.header(f"Data of C{modelnum}corr prediction")

    long_text = "\n".join([','.join(map(lambda x: format(x, '.4f'), i)) for i in cNfree_predictions])

    file_name = st.text_input('file name', f'C{modelnum}corr_prediction_data.txt')

    st.download_button('Download data', long_text, file_name=f"{file_name}")

    stx.scrollableTextbox(long_text, height = 300)

def main():
    st.title("Cyclizability Prediction\n")
    st.markdown("---")
    st.subheader("website guide")
    st.markdown("***functions*** + ***parameters***")
    st.markdown("***:blue[spatial visualization function]***: ***:green[pdb id]*** OR ***:violet[custom sequence + pdb file]***")
    st.markdown("***:blue[C0, C26, C29, or C31 predictions]***: ***:green[pdb id (fasta sequence >= 50bp)]*** OR ***:violet[custom sequence (>= 50bp)]***")
    st.markdown("the [github](%s) code!" % "https://github.com/codergirl1106/Cyclizability-Prediction-Website/")
    st.markdown("---")
    
    seq = ''
    seq1 = ''
    seq2 = ''
    cords = ''
    pdbcif = ''

    imgg = io.BytesIO()
    
    st.subheader("Analysis Type")
    option = st.selectbox('', ('Spatial analysis', 'C0corr prediction', 'C26corr prediction', 'C29corr prediction', 'C31corr prediction'))

    if option == 'Spatial analysis':

        st.subheader("Input Option 1: RCSB PDB IDs")
        pdbid = st.text_input('PDB ID','', placeholder="7OHC").upper()
        if pdbid != '' and seq == '':
            try:
                seq, cords = getSequence(getFasta(pdbid), getCif(pdbid), "cif")
            except:
                pass

        st.markdown("---")

        st.subheader("Input Option 2: Custom Structure")

        col1, col2, col3 = st.columns([0.45, 0.1, 0.45])
        with col1:
            try:
                st.markdown("[example fasta file](%s)" % "https://drive.google.com/file/d/1mcLi6EMX7xjKzD4gqQUrHAEQJFGz2rey/view?usp=sharing")
                fasta = st.file_uploader("upload a fasta file").getvalue().decode("utf-8")
                t1 = st.markdown("[example fasta file](%s)" % "https://drive.google.com/file/d/1mcLi6EMX7xjKzD4gqQUrHAEQJFGz2rey/view?usp=sharing")
            except:
                pass

        with col2:
            st.subheader("AND")

        with col3:
            try:
                st.markdown("[example cif file](%s)" % "https://drive.google.com/file/d/15QZako2huyhmpRuoyXIgJzG9oz72UUk4/view?usp=sharing")
                pdbcif = st.file_uploader("upload a pdb/cif file")
                fileextension = pdbcif.name.split('.')[-1]
                pdbcif = pdbcif.getvalue().decode("utf-8")
                seq, cords = getSequence(fasta, pdbcif, fileextension)
            except:
                pass

        st.markdown("NOTE: We recommend running the website using .cif files from RCSB PDB rather than .pdb, when possible.")
        st.markdown("---")

        if len(seq) != 0 and len(cords) != 0 and pdbid != '':
            spatial_analysis_ui(imgg, seq, getCif(pdbid), cords)
        else:
            st.subheader(":red[Incomplete Information to Visualize]")
    else:
        st.subheader("Input")
        col1, col2, col3 = st.columns([0.45, 0.1, 0.45])
        with col1:
            try:
                st.markdown("[example fasta file](%s)" % "https://drive.google.com/file/d/13ZLmQs49wROgKT4M1vbdZ61odyqPzoan/view?usp=sharing")
                fasta = st.file_uploader("upload a fasta file (DNA sequences only)").getvalue().decode("utf-8")
                seq1 = processFasta(fasta.upper().rstrip().split('\n'))
            except:
                pass

        with col2:
            st.subheader("OR")

        with col3:
            try:
                fasta = st.text_area("custom DNA sequence input (one sequence per line)",
                                     placeholder="ATCAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCGAT\nATCGATGTATATATCTGACACGTGCCTGGAGACTAGGGAGTAATCCCCTTGGCGGTTAAAACGCGGGGGACAGCGCGTACGTGCGTTTAAGCGGTGCTAGAGCTGTCTACGACCAATTGAGCGGCCTCGGCACCGGGATTCTGAT")
                seq2 = fasta.upper().split()
            except:
                pass
        
        if len(seq1) != 0:
            if len(seq1[0]) < 50:
                st.subheader(":red[Please provide a sequence (>= 50bp) or a pdb id/file]")
            else:
                sequence_ui(imgg, seq1, option)      
        elif len(seq2) != 0:
            if len(seq2[0]) < 50:
                st.subheader(":red[Please provide a sequence (>= 50bp) or a pdb id/file]")
            else:
                sequence_ui(imgg, seq2, option)
    return 0

main()
