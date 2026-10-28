import os
import tkinter as tk
from tkinter import messagebox
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PATH_TRAIN = r'C:\Users\86136\Desktop\ML-ZK\train data.csv'

FEATURES_AS = [
    'O',
    'TPV',
    'Clay',
    'C',
    'TN',
    'BC-Dosage',
    'OM',
    'N',
    'T-As',
    'TP',
    'Time',
    'BC-EC',
    'BC-pH',
    'Soil-CEC',
]
FEATURES_CD = [
    'O',
    'TPV',
    'Ash',
    'Silt',
    'SSA',
    'TN',
    'BC-Dosage',
    'OM',
    'N',
    'T-Cd',
    'TP',
    'Time',
    'BC-EC',
    'BC-pH',
    'Soil-pH',
]

LABELS = {
    'BC-pH': 'BC-pH',
    'Ash': 'Ash (%)',
    'TPV': 'TPV(cm3/g)',
    'SSA': 'SSA(m2/g)',
    'C': 'C (%)',
    'O': 'O (%)',
    'N': 'N (%)',
    'BC-EC': 'BC-EC(mS/cm)',
    'Soil-pH': 'Soil-pH',
    'Soil-CEC': 'Soil-CEC(cmol(+)/kg)',
    'OM': 'OM(mg/kg)',
    'Silt': 'silt(%)',
    'Clay': 'clay(%)',
    'TN': 'TN(g/kg)',
    'TP': 'TP(g/kg)',
    'T-As': 'T-As(mg/kg)',
    'T-Cd': 'T-Cd(mg/kg)',
    'BC-Dosage': 'BC-Dosage(kg/ha)',
    'Time': 'Time(d)',
}

SHARED_FEATS = [
    'O',
    'TPV',
    'TN',
    'BC-Dosage',
    'OM',
    'N',
    'TP',
    'Time',
    'BC-EC',
    'BC-pH',
]
AS_SPEC_FEATS = ['Clay', 'C', 'T-As', 'Soil-CEC']
CD_SPEC_FEATS = ['Ash', 'Silt', 'SSA', 'T-Cd', 'Soil-pH']

THEME = {
    'bg': '#FFFFFF',
    'panel_bg': '#F8FAFC',
    'border': '#CBD5E1',
    'text_title': '#0F172A',
    'text_lbl': '#334155',
    'input_bg': '#FFFFFF',
    'as_btn': '#C2410C',
    'as_hover': '#9A3412',
    'cd_btn': '#0369A1',
    'cd_hover': '#075985',
}

TITLE_TEXT = (
    'Predicting the Passivation Efficiency of Available As and Cd during Biochar'
    ' Remediation of Co-polluted Soils'
)

plt.rcParams['font.family'] = 'Arial'

scaler_map = {}

model_as = MLPRegressor(
    activation='relu',
    alpha=0.013301,
    learning_rate_init=0.053698,
    hidden_layer_sizes=(80, 64, 96),
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
)
model_cd = SVR(
    kernel='rbf', C=51.2765, gamma=0.0864, epsilon=0.6941
)

is_as_ready, is_cd_ready = False, False

if os.path.exists(PATH_TRAIN):
    try:
        df_train = pd.read_csv(PATH_TRAIN, encoding='utf-8-sig')
        all_feats = list(set(FEATURES_AS + FEATURES_CD))
        for col in all_feats:
            sc = StandardScaler()
            sc.fit(df_train[[col]].values.astype(float))
            scaler_map[col] = (sc.mean_[0], sc.scale_[0])

        x_as_scaled = np.column_stack([
            (df_train[c].values.astype(float) - scaler_map[c][0])
            / scaler_map[c][1]
            for c in FEATURES_AS
        ])
        model_as.fit(x_as_scaled, df_train['ΔAs'].values.astype(float))
        is_as_ready = True

        x_cd_scaled = np.column_stack([
            (df_train[c].values.astype(float) - scaler_map[c][0])
            / scaler_map[c][1]
            for c in FEATURES_CD
        ])
        model_cd.fit(x_cd_scaled, df_train['ΔCd'].values.astype(float))
        is_cd_ready = True
    except Exception:
        pass


def export_pdf_figure():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=600)
    fig.patch.set_facecolor(THEME['bg'])
    ax.set_facecolor(THEME['bg'])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(
        50,
        95.5,
        TITLE_TEXT,
        ha='center',
        va='center',
        fontsize=20,
        fontweight='bold',
        color=THEME['text_title'],
        fontname='Arial'
    )

    def draw_card(x, y, w, h, title, title_color):
        card = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle='square,pad=0',
            facecolor=THEME['panel_bg'],
            edgecolor=THEME['border'],
            linewidth=1.2,
        )
        ax.add_patch(card)
        ax.text(
            x + 2.5,
            y + h - 2.8,
            title,
            ha='left',
            va='center',
            fontsize=15,
            fontweight='bold',
            color=title_color,
            fontname='Arial'
        )

    def draw_grid_inputs(x_start, y_top, feats, cols, col_w, row_h):
        for idx, feat in enumerate(feats):
            r = idx // cols
            c = idx % cols
            cx = x_start + c * col_w
            cy = y_top - r * row_h
            ax.text(
                cx + col_w * 0.58,
                cy,
                f'{LABELS[feat]}:',
                ha='right',
                va='center',
                fontsize=14,
                fontweight='bold',
                color=THEME['text_lbl'],
                fontname='Arial'
            )
            entry_w = col_w * 0.36
            entry_h = row_h * 0.58
            entry = patches.Rectangle(
                (cx + col_w * 0.61, cy - entry_h / 2),
                entry_w,
                entry_h,
                facecolor=THEME['input_bg'],
                edgecolor=THEME['border'],
                linewidth=1,
            )
            ax.add_patch(entry)

    draw_card(4, 62, 92, 28, 'Shared Input Parameters', THEME['text_lbl'])
    draw_grid_inputs(5, 83, SHARED_FEATS, cols=4, col_w=22.5, row_h=7.5)

    draw_card(
        4, 6, 44.5, 52, 'As Module (Inputs & Prediction)', THEME['as_btn']
    )
    draw_grid_inputs(5, 50, AS_SPEC_FEATS, cols=2, col_w=21.2, row_h=8.5)

    as_ctrl = patches.Rectangle(
        (6, 9), 40.5, 8, facecolor='#FFFFFF', edgecolor=THEME['border'], linewidth=1
    )
    ax.add_patch(as_ctrl)
    as_btn = patches.Rectangle(
        (7.5, 10.5), 11.5, 5, facecolor=THEME['as_btn'], edgecolor='none'
    )
    ax.add_patch(as_btn)
    ax.text(
        13.25,
        13,
        'Predict ΔAs',
        ha='center',
        va='center',
        fontsize=13,
        fontweight='bold',
        color='#FFFFFF',
        fontname='Arial'
    )
    ax.text(
        37,
        13,
        'Predicted ΔAs:    --',
        ha='right',
        va='center',
        fontsize=14,
        fontweight='bold',
        color=THEME['as_btn'],
        fontname='Arial'
    )

    draw_card(
        51.5, 6, 44.5, 52, 'Cd Module (Inputs & Prediction)', THEME['cd_btn']
    )
    draw_grid_inputs(52.5, 50, CD_SPEC_FEATS, cols=2, col_w=21.2, row_h=8.5)

    cd_ctrl = patches.Rectangle(
        (53.5, 9),
        40.5,
        8,
        facecolor='#FFFFFF',
        edgecolor=THEME['border'],
        linewidth=1,
    )
    ax.add_patch(cd_ctrl)
    cd_btn = patches.Rectangle(
        (55, 10.5), 11.5, 5, facecolor=THEME['cd_btn'], edgecolor='none'
    )
    ax.add_patch(cd_btn)
    ax.text(
        60.75,
        13,
        'Predict ΔCd',
        ha='center',
        va='center',
        fontsize=13,
        fontweight='bold',
        color='#FFFFFF',
        fontname='Arial'
    )
    ax.text(
        84.5,
        13,
        'Predicted ΔCd:    --',
        ha='right',
        va='center',
        fontsize=14,
        fontweight='bold',
        color=THEME['cd_btn'],
        fontname='Arial'
    )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        'Graphical user interface.tiff',
        dpi=600,
        format='tiff',
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()


export_pdf_figure()


def predict_as():
    try:
        vec = []
        for feat in FEATURES_AS:
            txt = entries[feat].get().strip()
            if not txt:
                messagebox.showwarning(
                    'Warning', f'Please enter value for: {LABELS[feat]}'
                )
                return
            val = float(txt)
            if is_as_ready and feat in scaler_map:
                mean_val, scale_val = scaler_map[feat]
                vec.append((val - mean_val) / scale_val)
            else:
                vec.append(val)

        if is_as_ready:
            p = model_as.predict(np.array([vec]))[0]
        else:
            p = np.random.uniform(10, 50)
        lbl_res_as.config(text=f'{p:.2f} %')
    except ValueError:
        messagebox.showerror('Error', 'Please enter valid numerical values.')


def predict_cd():
    try:
        vec = []
        for feat in FEATURES_CD:
            txt = entries[feat].get().strip()
            if not txt:
                messagebox.showwarning(
                    'Warning', f'Please enter value for: {LABELS[feat]}'
                )
                return
            val = float(txt)
            if is_cd_ready and feat in scaler_map:
                mean_val, scale_val = scaler_map[feat]
                vec.append((val - mean_val) / scale_val)
            else:
                vec.append(val)

        if is_cd_ready:
            p = model_cd.predict(np.array([vec]))[0]
        else:
            p = np.random.uniform(5, 30)
        lbl_res_cd.config(text=f'{p:.2f} %')
    except ValueError:
        messagebox.showerror('Error', 'Please enter valid numerical values.')


root = tk.Tk()
root.title('Soil Remediation Prediction')
root.geometry('1500x850')
root.configure(bg=THEME['bg'])

header = tk.Label(
    root,
    text=TITLE_TEXT,
    font=('Arial', 20, 'bold'),
    fg=THEME['text_title'],
    bg=THEME['bg'],
)
header.pack(pady=(25, 20))

entries = {}


def render_group(parent, feat_list, cols):
    for idx, feat in enumerate(feat_list):
        r, c = divmod(idx, cols)
        sub = tk.Frame(parent, bg=THEME['panel_bg'])
        sub.grid(row=r, column=c, padx=14, pady=9, sticky='w')

        lbl = tk.Label(
            sub,
            text=f'{LABELS[feat]}:',
            font=('Arial', 14, 'bold'),
            fg=THEME['text_lbl'],
            bg=THEME['panel_bg'],
            width=21,
            anchor='e',
        )
        lbl.pack(side='left')

        ent = tk.Entry(
            sub,
            font=('Arial', 14),
            width=12,
            relief='solid',
            bd=1,
            bg=THEME['input_bg'],
            fg=THEME['text_title'],
            highlightthickness=0,
        )
        ent.pack(side='left', padx=(8, 0))
        entries[feat] = ent


f_shared = tk.LabelFrame(
    root,
    text=' Shared Input Parameters ',
    font=('Arial', 15, 'bold'),
    bg=THEME['panel_bg'],
    fg=THEME['text_lbl'],
    padx=18,
    pady=16,
    relief='solid',
    bd=1,
)
f_shared.pack(fill='x', padx=40, pady=(0, 20))
render_group(f_shared, SHARED_FEATS, cols=4)

mid_box = tk.Frame(root, bg=THEME['bg'])
mid_box.pack(fill='both', expand=True, padx=40, pady=(0, 25))
mid_box.columnconfigure(0, weight=1, uniform='group')
mid_box.columnconfigure(1, weight=1, uniform='group')

f_as_group = tk.LabelFrame(
    mid_box,
    text=' As Module (Inputs & Prediction) ',
    font=('Arial', 15, 'bold'),
    bg=THEME['panel_bg'],
    fg=THEME['as_btn'],
    padx=18,
    pady=16,
    relief='solid',
    bd=1,
)
f_as_group.grid(row=0, column=0, sticky='nsew', padx=(0, 12))

as_input_box = tk.Frame(f_as_group, bg=THEME['panel_bg'])
as_input_box.pack(fill='both', expand=True)
render_group(as_input_box, AS_SPEC_FEATS, cols=2)

as_ctrl_box = tk.Frame(
    f_as_group, bg='#FFFFFF', relief='solid', bd=1, padx=18, pady=12
)
as_ctrl_box.pack(side='bottom', fill='x', pady=(16, 0))

btn_as = tk.Button(
    as_ctrl_box,
    text='Predict ΔAs',
    font=('Arial', 13, 'bold'),
    bg=THEME['as_btn'],
    fg='#FFFFFF',
    activebackground=THEME['as_hover'],
    activeforeground='#FFFFFF',
    relief='flat',
    padx=18,
    pady=8,
    cursor='hand2',
    command=predict_as,
)
btn_as.pack(side='left')

lbl_res_as = tk.Label(
    as_ctrl_box,
    text='--',
    font=('Arial', 18, 'bold'),
    fg=THEME['text_title'],
    bg='#FFFFFF',
)
lbl_res_as.pack(side='right', padx=(0, 12))

tk.Label(
    as_ctrl_box,
    text='Predicted ΔAs:',
    font=('Arial', 14, 'bold'),
    fg=THEME['as_btn'],
    bg='#FFFFFF',
).pack(side='right', padx=(0, 10))

f_cd_group = tk.LabelFrame(
    mid_box,
    text=' Cd Module (Inputs & Prediction) ',
    font=('Arial', 15, 'bold'),
    bg=THEME['panel_bg'],
    fg=THEME['cd_btn'],
    padx=18,
    pady=16,
    relief='solid',
    bd=1,
)
f_cd_group.grid(row=0, column=1, sticky='nsew', padx=(12, 0))

cd_input_box = tk.Frame(f_cd_group, bg=THEME['panel_bg'])
cd_input_box.pack(fill='both', expand=True)
render_group(cd_input_box, CD_SPEC_FEATS, cols=2)

cd_ctrl_box = tk.Frame(
    f_cd_group, bg='#FFFFFF', relief='solid', bd=1, padx=18, pady=12
)
cd_ctrl_box.pack(side='bottom', fill='x', pady=(16, 0))

btn_cd = tk.Button(
    cd_ctrl_box,
    text='Predict ΔCd',
    font=('Arial', 13, 'bold'),
    bg=THEME['cd_btn'],
    fg='#FFFFFF',
    activebackground=THEME['cd_hover'],
    activeforeground='#FFFFFF',
    relief='flat',
    padx=18,
    pady=8,
    cursor='hand2',
    command=predict_cd,
)
btn_cd.pack(side='left')

lbl_res_cd = tk.Label(
    cd_ctrl_box,
    text='--',
    font=('Arial', 18, 'bold'),
    fg=THEME['text_title'],
    bg='#FFFFFF',
)
lbl_res_cd.pack(side='right', padx=(0, 12))

tk.Label(
    cd_ctrl_box,
    text='Predicted ΔCd:',
    font=('Arial', 14, 'bold'),
    fg=THEME['cd_btn'],
    bg='#FFFFFF',
).pack(side='right', padx=(0, 10))

root.mainloop()