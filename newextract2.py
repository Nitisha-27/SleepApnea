import os
import warnings
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk

from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ============================================
# SETTINGS
# ============================================

ROOT_FOLDER = "train"

N_JOBS = 4

CHUNK_SECONDS = 600

# ============================================
# HELPER FUNCTIONS
# ============================================

def safe_median(x):

    x = np.asarray(x,dtype=float)

    x = x[np.isfinite(x)]

    if len(x)==0:

        return np.nan

    return np.median(x)


def add_interval(lst,val,low,high):

    if np.isfinite(val):

        if low<=val<=high:

            lst.append(val)


# ============================================
# PROCESS ONE PATIENT
# ============================================

def process_patient(patient):

    try:

        record_path=os.path.join(
            ROOT_FOLDER,
            patient
        )

        print(f"\nProcessing {patient}")

        record=wfdb.rdrecord(record_path)

        ecg=record.p_signal[:,0]

        fs=int(record.fs)

        ecg=nk.ecg_clean(
            ecg,
            sampling_rate=fs
        )

        _,info=nk.ecg_peaks(

            ecg,

            sampling_rate=fs

        )

        rpeaks=info["ECG_R_Peaks"]

        if len(rpeaks)<20:

            return None

        rr=np.diff(rpeaks)/fs*1000

        rr=rr[

            (rr>=300)

            &

            (rr<=2000)

        ]

        if len(rr)<20:

            return None

        mean_rr=np.mean(rr)

        hrv=np.std(

            rr,

            ddof=1

        )

        heart_rate=np.mean(

            60000/rr

        )

        p_list=[]

        t_list=[]

        pq_list=[]

        pr_list=[]

        qrs_list=[]

        qt_list=[]

        st_list=[]

        rt_list=[]

        pt_list=[]

        tpte_list=[]

        chunk_samples=CHUNK_SECONDS*fs

                # ============================================
        # CHUNK PROCESSING
        # ============================================

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

                p_on = np.asarray(waves["ECG_P_Onsets"])
                p_pk = np.asarray(waves["ECG_P_Peaks"])
                p_off = np.asarray(waves["ECG_P_Offsets"])

                q_pk = np.asarray(waves["ECG_Q_Peaks"])

                r_on = np.asarray(waves["ECG_R_Onsets"])
                r_off = np.asarray(waves["ECG_R_Offsets"])

                s_pk = np.asarray(waves["ECG_S_Peaks"])

                t_on = np.asarray(waves["ECG_T_Onsets"])
                t_pk = np.asarray(waves["ECG_T_Peaks"])
                t_off = np.asarray(waves["ECG_T_Offsets"])

                # Use the number of detected R-peaks in this chunk
                beats = len(chunk_r)

                for i in range(beats):

                    try:

                        if (
                            np.isnan(p_on[i]) or
                            np.isnan(p_pk[i]) or
                            np.isnan(p_off[i]) or
                            np.isnan(q_pk[i]) or
                            np.isnan(r_on[i]) or
                            np.isnan(r_off[i]) or
                            np.isnan(s_pk[i]) or
                            np.isnan(t_on[i]) or
                            np.isnan(t_pk[i]) or
                            np.isnan(t_off[i])
                        ):
                            continue

                        # --------------------
                        # P Duration
                        # --------------------
                        p = (p_off[i] - p_on[i]) / fs * 1000
                        add_interval(p_list, p, 20, 250)

                        # --------------------
                        # T Duration
                        # --------------------
                        t = (t_off[i] - t_on[i]) / fs * 1000
                        add_interval(t_list, t, 40, 400)

                        # --------------------
                        # PQ Interval
                        # --------------------
                        pq = (q_pk[i] - p_off[i]) / fs * 1000
                        add_interval(pq_list, pq, 0, 250)

                        # --------------------
                        # PR Interval
                        # --------------------
                        pr = (q_pk[i] - p_on[i]) / fs * 1000
                        add_interval(pr_list, pr, 80, 320)

                        # --------------------
                        # QRS Duration
                        # --------------------
                        qrs = (s_pk[i] - q_pk[i]) / fs * 1000
                        add_interval(qrs_list, qrs, 40, 250)

                        # --------------------
                        # QT Interval
                        # --------------------
                        qt = (t_off[i] - q_pk[i]) / fs * 1000
                        add_interval(qt_list, qt, 200, 700)

                        # --------------------
                        # ST Interval
                        # --------------------
                        st = (t_on[i] - s_pk[i]) / fs * 1000
                        add_interval(st_list, st, 20, 500)

                        # --------------------
                        # RT Interval
                        # --------------------
                        rt = (t_pk[i] - r_off[i]) / fs * 1000
                        add_interval(rt_list, rt, 20, 600)

                        # --------------------
                        # PT Interval
                        # --------------------
                        pt = (t_pk[i] - p_pk[i]) / fs * 1000
                        add_interval(pt_list, pt, 100, 1000)

                        # --------------------
                        # TPTE
                        # --------------------
                        tpte = (t_off[i] - t_pk[i]) / fs * 1000
                        add_interval(tpte_list, tpte, 20, 250)

                    except Exception:
                        continue

            except Exception:
                continue

                    # ============================================
        # COMPUTE FINAL FEATURES
        # ============================================

        result = {

            "Patient_ID": patient,

            "RR": round(np.mean(rr),3),

            "HRV": round(np.std(rr,ddof=1),3),

            "HeartRate": round(np.mean(60000/rr),3),

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

            "Label": 1

        }

        print(f"Finished {patient}")

        return result

    except Exception as e:

        print(f"{patient} FAILED : {e}")

        return None


# ============================================
# FIND ALL APNEA RECORDS
# ============================================

patients = sorted([

    os.path.splitext(f)[0]

    for f in os.listdir(ROOT_FOLDER)

    if f.endswith(".hea")

])

print(f"\nFound {len(patients)} apnea patients")

# ============================================
# RUN PARALLEL
# ============================================

results = Parallel(

    n_jobs=N_JOBS,

    backend="loky",

    verbose=10

)(

    delayed(process_patient)(p)

    for p in patients

)

# ============================================
# REMOVE FAILED RECORDS
# ============================================

results = [

    r

    for r in results

    if r is not None

]

df = pd.DataFrame(results)

df = df.sort_values(

    "Patient_ID"

).reset_index(

    drop=True

)

# ============================================
# SAVE CSV
# ============================================

output_file = "apnea_features.csv"

df.to_csv(

    output_file,

    index=False

)

print("\n===================================")

print("Extraction Completed")

print("Patients :",len(df))

print("Saved :",output_file)

print("===================================")

print(df.head())

print("\nColumns\n")

print(df.columns)