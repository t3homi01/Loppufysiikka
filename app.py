import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

st.set_page_config(layout="wide")
st.title("Kävely- ja juoksuanalyysi (Phyphox)")


# Data

def raw_url(filename):
    return f"https://raw.githubusercontent.com/t3homi01/Loppufysiikka/main/{filename}"

try:
    acc = pd.read_csv("Linear Accelerometer.csv")
    loc = pd.read_csv("Location.csv")
except FileNotFoundError:
    acc = pd.read_csv(raw_url("Linear Accelerometer.csv"))
    loc = pd.read_csv(raw_url("Location.csv"))

t_acc = acc.columns[0]
ax, ay, az = acc.columns[1:4]

t_loc = loc.columns[0]
lat, lon = loc.columns[1:3]
spd = loc.columns[3] if len(loc.columns) > 3 else None


# Kiihtyvyys

comp = st.sidebar.selectbox("Analysoitava kiihtyvyys", ["az", "ay", "ax"])
sig = acc[{"ax": ax, "ay": ay, "az": az}[comp]].to_numpy()
t = acc[t_acc].to_numpy()

dt = np.mean(np.diff(t))
fs = 1 / dt

b, a = butter(3, [0.7 / (fs / 2), 3.0 / (fs / 2)], btype="band")
sig_f = filtfilt(b, a, sig)

peaks, _ = find_peaks(sig_f, distance=0.3 * fs, prominence=0.5)
steps_time = len(peaks)

f, pxx = welch(sig, fs=fs)
mask = (f >= 0.7) & (f <= 3)
f_dom = f[mask][np.argmax(pxx[mask])]
duration = t[-1] - t[0]
steps_fft = int(f_dom * duration)


#GPS DATA

def hav(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

latv = loc[lat].to_numpy()
lonv = loc[lon].to_numpy()

dist = np.sum(hav(latv[:-1], lonv[:-1], latv[1:], lonv[1:]))

t_gps = loc[t_loc].to_numpy()
time_tot = t_gps[-1] - t_gps[0]
v_avg = dist / time_tot

step_len = dist / steps_time


#Testien tulokset
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Askelmäärä (suodatettu)", steps_time)
c2.metric("Askelmäärä (Fourier)", steps_fft)
c3.metric("Keskinopeus (m/s)", f"{v_avg:.2f}")
c4.metric("Matka (m)", f"{dist:.1f}")
c5.metric("Askelpituus (m)", f"{step_len:.2f}")


#Kuvaajat datasta
st.subheader("Suodatettu kiihtyvyys")
st.line_chart(pd.DataFrame({"t": t, "a": sig_f}), x="t", y="a")

st.subheader("Tehospektritiheys")
st.line_chart(pd.DataFrame({"f": f, "PSD": pxx}), x="f", y="PSD")

st.subheader("Reitti kartalla")
st.map(pd.DataFrame({"lat": latv, "lon": lonv}))
