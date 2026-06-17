import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

def preprocess(file_path):
    df = pd.read_csv(file_path, skiprows=[0, 2])
    
    train_set = int(re.search(r'T(\d+)', file_path.name).group(1))
    is_sp1900 = train_set <= 48
    
    def hex_to_dec(series):
        def convert(x):
            try: return int(str(x).strip() or '0', 16)
            except ValueError: return 0
        return series.apply(convert)

    velocity = (hex_to_dec(df['VMEAS+1']) * 4) + (hex_to_dec(df['VMEAS']) / 64)
    df['velocity'] = np.where(velocity > 150, 0, velocity)
    df['commanded'] = (hex_to_dec(df['VCOM+1']) * 4) + (hex_to_dec(df['VCOM']) / 64)
    df['thrust'] = (hex_to_dec(df['APX40']) / 255) * 100
    df['mode'] = np.where(hex_to_dec(df['APX44']) & 4, "Motoring", np.where(df['velocity'] > 0, np.where(hex_to_dec(df['APX44']) & 16, "Braking", "Coasting"), "Stopped"))
    df['grade'] = df['KTDX24'].astype(str).map({'1': 0.05575, '2': 0.0471, '3': 0.0385, '4': 0.02855, '5': 0.02135, '6': 0.0128, '7': 0.0043, '8': 0, '9': -0.0043, 'A': -0.0128, 'B': -0.02135, 'C': -0.02855, 'D': -0.0385, 'E': -0.0471, 'F': -0.05575}).fillna(0)

    return df, is_sp1900

def calculate_energy(df, is_sp1900, regen=0.2, eff=0.8):

    time = pd.to_datetime(df['Name'])
    dt = 1 / df.groupby('Name')['Name'].transform('count')
    df['time'] = time + pd.to_timedelta(df.groupby('Name').cumcount() * dt, unit='s')

    v = df['velocity']
    distance_dt = (v/3.6) * dt
    distance = distance_dt.sum()/1000
    thrust = df['thrust'] / 100

    is_motoring = df['mode'] == "Motoring"
    is_braking = df['mode'] == "Braking"

    if is_sp1900:
        mass_tons = 460
        fp = np.where(is_motoring, np.select([v<37, v<40, v<46, v<50, v<60, v<100, v>=100], [32, -5/9*v+470/9, -19/30*v+166/3, -11/20*v+103/2, -0.4*v+44, 150000/(v+3.43867)**2.15, 838.8884/(v+20)]) * 1000 * 16 * thrust, 0) 
        fb = np.where(is_braking, np.select([v<5, v<64, v>=64], [0, 25, 140000/(v+4.76177)**2.04]) * 1000 * 16 * thrust, 0) 
    else:
        mass_tons = 480
        fp = np.where(is_motoring, np.select([v<35,v<43,v<50,v<55,v>=55], [27, -0.625*v+48.875, -0.43*v+40.5, -0.3*v+33.99, 170000/(v+7.552235)**2.22]) * 1000 * 20 * thrust, 0)
        fb = np.where(is_braking, np.select([v<5,v<60,v>=60], [0, 25, 260000/(v+6.97678)**2.2]) * 1000 * 20 * thrust, 0)

    resistance = 4160+(6.4*(mass_tons))+(0.14*(mass_tons)*v)+(0.96*v**2)
    g_force = mass_tons * 1000 * 9.81 * np.sin(np.arctan(df['grade']))

    df['motor_J'] = np.where(is_motoring , (fp + resistance + g_force) / eff * distance_dt, 0)
    df['regen_J'] = np.where(is_braking , (-fb + resistance + g_force) * regen * distance_dt, 0)
    df['energy_J'] = np.maximum(0 , df['motor_J'])+ np.minimum(0 , df['regen_J'])

    total_kwh = df['energy_J'].sum() / 3.6e6

    kwh_carkm = total_kwh / (distance*8)

    return df, total_kwh, distance, kwh_carkm

def plot(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['time'], y=df['commanded'],mode='lines', name='Commanded',line=dict(color='blue', width=1, shape='hv'), hoverinfo='skip'))

    colors = {'Motoring': 'orange', 'Coasting': 'green', 'Braking': 'yellow', 'Stopped': 'lightgray'}
    
    for mode in colors.keys():
        df_mode = df[df['mode'] == mode]
        if not df_mode.empty:
            fig.add_trace(go.Bar(x=df_mode['time'], y=df_mode['velocity'],name=f'{mode}',marker=dict(color=colors[mode], line=dict(width=0)),width=360, 
                customdata=df_mode[['Name', 'thrust', 'energy_J', 'mode', 'commanded', 'grade']],
                hovertemplate= "<b>Time:</b> %{customdata[0]}<br>" + "<b>Mode:</b> %{customdata[3]}<br>" + "<b>Commanded Speed:</b> %{customdata[4]:.1f} km/h<br>" + "<b>Speed:</b> %{y:.1f} km/h<br>" 
                + "<b>Thrust:</b> %{customdata[1]:.1f}%<br>" + "<b>Grade:</b> %{customdata[5]:.5f} <br>" + "<b>Energy:</b> %{customdata[2]:.2f} J<br>" + "<extra></extra>"))

    fig.update_layout(title='Train Speed Profile ', xaxis_title='Time', yaxis_title='Speed (km/h)', hovermode='x unified', bargap=0,  template='plotly_white')
    return fig


def export_results(df, output_path="energy_results.csv"):
    df.to_csv(output_path, index=False)
    print(f"Done -> {output_path}")
