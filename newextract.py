import os
import warnings
import numpy as np
import pandas as pd
import mne
import neurokit2 as nk

from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ======================================================
# SETTINGS
# ======================================================

ROOT_FOLDER = "."
N_JOBS = 2          # Increase to 4 if RAM >=16 GB
CHUNK_SECONDS = 600 # 10 minutes

# ======================================================
# SAFE FUNCTIONS
# ======================================================

def safe_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return np.median(x)


def add_interval(lst, value, low, high):
    if np.isfinite(value):
        if low <= value <= high:
            lst.append(value)


# ======================================================
# PROCESS ONE PATIENT
# ======================================================

def process_patient(patient_id):

    try:

        edf = os.path.join(
            ROOT_FOLDER,
            patient_id,
            patient_id,
            f"{patient_id}.edf"
        )

        if not os.path.exists(edf):
            print(f"Missing : {patient_id}")
            return None

        print(f"\nProcessing {patient_id}")

        raw = mne.io.read_raw_edf(
            edf,
            include=["ECG1"],
            preload=False,
            verbose=False
        )

        fs = int(raw.info["sfreq"])

        ecg = raw.get_data(
            picks="ECG1"
        )[0]

        # --------------------------------------
        # CLEAN ECG
        # --------------------------------------

        ecg = nk.ecg_clean(
            ecg,
            sampling_rate=fs
        )

        # --------------------------------------
        # R PEAKS
        # --------------------------------------

        _, info = nk.ecg_peaks(
            ecg,
            sampling_rate=fs
        )

        rpeaks = info["ECG_R_Peaks"]

        if len(rpeaks) < 20:
            print("Too few peaks")
            return None

        # --------------------------------------
        # RR
        # --------------------------------------

        rr = np.diff(rpeaks) / fs * 1000

        rr = rr[
            (rr >= 300) &
            (rr <= 2000)
        ]

        if len(rr) < 20:
            return None

        mean_rr = np.mean(rr)

        hrv = np.std(
            rr,
            ddof=1
        )

        heart_rate = np.mean(
            60000 / rr
        )

        # ====================================================
        # FEATURE LISTS
        # ====================================================

        p_list = []

        t_list = []

        pq_list = []

        pr_list = []

        qrs_list = []

        qt_list = []

        st_list = []

        rt_list = []

        pt_list = []

        tpte_list = []

        chunk_samples = CHUNK_SECONDS * fs

                # ====================================================
        # CHUNK PROCESSING
        # ====================================================

        for start in range(0, len(ecg), chunk_samples):

            end = min(start + chunk_samples, len(ecg))

            ecg_chunk = ecg[start:end]

            chunk_r = rpeaks[
                (rpeaks >= start) &
                (rpeaks < end)
            ] - start

            if len(chunk_r) < 20:
                continue

            try:

                _, waves = nk.ecg_delineate(
                    ecg_chunk,
                    rpeaks=chunk_r,
                    sampling_rate=fs,
                    method="dwt"
                )

                p_on = waves["ECG_P_Onsets"]
                p_pk = waves["ECG_P_Peaks"]
                p_off = waves["ECG_P_Offsets"]

                q_pk = waves["ECG_Q_Peaks"]

                r_on = waves["ECG_R_Onsets"]
                r_off = waves["ECG_R_Offsets"]

                s_pk = waves["ECG_S_Peaks"]

                t_on = waves["ECG_T_Onsets"]
                t_pk = waves["ECG_T_Peaks"]
                t_off = waves["ECG_T_Offsets"]

                n = min(
                    len(p_on),
                    len(p_pk),
                    len(p_off),
                    len(q_pk),
                    len(r_on),
                    len(r_off),
                    len(s_pk),
                    len(t_on),
                    len(t_pk),
                    len(t_off)
                )

                for i in range(n):

                    vals = [
                        p_on[i], p_pk[i], p_off[i],
                        q_pk[i],
                        r_on[i], r_off[i],
                        s_pk[i],
                        t_on[i], t_pk[i], t_off[i]
                    ]

                    if np.any(np.isnan(vals)):
                        continue

                    # -----------------------
                    # P duration
                    # -----------------------

                    p = (p_off[i]-p_on[i])/fs*1000
                    add_interval(p_list,p,20,250)

                    # -----------------------
                    # T duration
                    # -----------------------

                    t = (t_off[i]-t_on[i])/fs*1000
                    add_interval(t_list,t,40,400)

                    # -----------------------
                    # PQ
                    # -----------------------

                    pq = (q_pk[i]-p_off[i])/fs*1000
                    add_interval(pq_list,pq,0,200)

                    # -----------------------
                    # PR
                    # -----------------------

                    pr = (q_pk[i]-p_on[i])/fs*1000
                    add_interval(pr_list,pr,80,320)

                    # -----------------------
                    # QRS
                    # -----------------------

                    qrs = (s_pk[i]-q_pk[i])/fs*1000
                    add_interval(qrs_list,qrs,40,250)

                    # -----------------------
                    # QT
                    # -----------------------

                    qt = (t_off[i]-q_pk[i])/fs*1000
                    add_interval(qt_list,qt,200,700)

                    # -----------------------
                    # ST
                    # -----------------------

                    st = (t_on[i]-s_pk[i])/fs*1000
                    add_interval(st_list,st,20,500)

                    # -----------------------
                    # RT
                    # -----------------------

                    rt = (t_pk[i]-r_off[i])/fs*1000
                    add_interval(rt_list,rt,20,600)

                    # -----------------------
                    # PT
                    # -----------------------

                    pt = (t_pk[i]-p_pk[i])/fs*1000
                    add_interval(pt_list,pt,100,1000)

                    # -----------------------
                    # TPTE
                    # -----------------------

                    tpte = (t_off[i]-t_pk[i])/fs*1000
                    add_interval(tpte_list,tpte,20,250)

            except Exception:
                continue

                    # ====================================================
        # RETURN ONE ROW
        # ====================================================

        print(f"Finished {patient_id}")

        return {

            "Patient_ID": patient_id,

            "RR": round(mean_rr,3),

            "HRV": round(hrv,3),

            "HeartRate": round(heart_rate,3),

            "P": round(safe_median(p_list),3),

            "T": round(safe_median(t_list),3),

            "PQ": round(safe_median(pq_list),3),

            "PR": round(safe_median(pr_list),3),

            "QRS": round(safe_median(qrs_list),3),

            "QT": round(safe_median(qt_list),3),

            "ST": round(safe_median(st_list),3),

            "RT": round(safe_median(rt_list),3),

            "PT": round(safe_median(pt_list),3),

            "TPTE": round(safe_median(tpte_list),3),

            "Label": 0

        }

    except Exception as e:

        print(f"{patient_id} FAILED : {e}")

        return None


# ==========================================================
# FIND ALL HEALTHY PATIENTS
# ==========================================================

patients = sorted([

    folder

    for folder in os.listdir(ROOT_FOLDER)

    if folder.startswith("EPCTL")

    and os.path.isdir(folder)

])

print("\nHealthy Patients Found :", len(patients))

# ==========================================================
# RUN PARALLEL
# ==========================================================

results = Parallel(

    n_jobs=N_JOBS,

    backend="loky",

    verbose=10

)(

    delayed(process_patient)(patient)

    for patient in patients

)

# ==========================================================
# REMOVE FAILED PATIENTS
# ==========================================================

results = [

    x

    for x in results

    if x is not None

]

df = pd.DataFrame(results)

# ==========================================================
# SORT
# ==========================================================

df = df.sort_values(

    "Patient_ID"

).reset_index(

    drop=True

)

# ==========================================================
# SAVE
# ==========================================================

output = "healthy_features.csv"

df.to_csv(

    output,

    index=False

)

print("\n=====================================")

print("Extraction Completed")

print("Patients :", len(df))

print("Saved :", output)

print("=====================================\n")

print(df.head())

print("\nColumns\n")

print(df.columns)
