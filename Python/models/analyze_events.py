"""
analyze_events.py
Ported analyzeEvents function (PyTorch inference + statistics) as a reusable function.

Function:
    analyzeEvents(D, netArr, netBin, batch_size=1024, c_perc=0.95, device="cpu")
Returns:
    dict of statistics and diagnosis.
"""
import numpy as np
import torch
from scipy.special import erfinv

def analyzeEvents(D, netArr, netBin, batch_size=1024, c_perc=0.95, device="cpu"):
    """
    D: np.ndarray shaped [N, seqLen, feat]
    netArr, netBin: torch models (expecting input shaped [seqLen, feat, batch])
    """
    numSamples, seqLen, inputDim = D.shape

    # Z-score across entire dataset per feature
    mu = D.reshape(-1, inputDim).mean(axis=0)
    sigma = D.reshape(-1, inputDim).std(axis=0) + 1e-12
    Dn = (D - mu) / sigma

    classNamesArr = [
        "!", '"', "+", "/", "A", "E", "F", "J", "L", "N",
        "Q", "R", "V", "[", "]", "a", "e", "f", "j", "x", "|", "~"
    ]
    numClasses = len(classNamesArr)

    E_numEvents = np.zeros(numClasses, dtype=np.float64)
    scores_all_list = []

    netArr.eval()
    netBin.eval()
    device = torch.device(device)

    with torch.no_grad():
        for b in range(int(np.ceil(numSamples / batch_size))):
            start = b * batch_size
            end = min((b + 1) * batch_size, numSamples)
            batch = Dn[start:end]  # [B, seqLen, feat]
            # convert to tensor [seqLen, feat, batch]
            dataBatch = torch.tensor(batch, dtype=torch.float32, device=device).permute(1, 2, 0)

            out_bin = netBin(dataBatch).detach().cpu().numpy()  # shape [seqLen, num_bin, batch] or [seqLen*batch, classes] depending on model
            out_arr = netArr(dataBatch).detach().cpu().numpy()

            # We assume out_arr shape: [seqLen*batch, numArrClasses] or [numArrClasses, ...] depending on model impl.
            # To be compatible with earlier code, flatten to 2D rows.
            out_arr2 = out_arr.reshape(-1, out_arr.shape[-1]) if out_arr.ndim > 2 else out_arr
            out_bin2 = out_bin.reshape(-1, out_bin.shape[-1]) if out_bin.ndim > 2 else out_bin

            # Combine by inserting binary column at index 9 (the prior MATLAB logic)
            # Ensure shapes match on rows
            if out_arr2.shape[0] != out_bin2.shape[0]:
                # try broadcasting if netBin outputs single-column per batch
                min_rows = min(out_arr2.shape[0], out_bin2.shape[0])
                out_arr2 = out_arr2[:min_rows]
                out_bin2 = out_bin2[:min_rows]

            scoresBatch = np.concatenate([out_arr2[:, :9], out_bin2[:, [0]], out_arr2[:, 9:]], axis=1)
            scores_all_list.append(scoresBatch)
            E_numEvents += scoresBatch.sum(axis=0)
            print(f"[analyzeEvents] processed batch {b+1}/{int(np.ceil(numSamples/batch_size))}")

    scores_all = np.vstack(scores_all_list)
    N = scores_all.shape[0]
    Var_numEvents = ((scores_all - (E_numEvents / N)) ** 2).sum(axis=0) / max(1, (N - 1))
    Std_percentEvents = np.sqrt(Var_numEvents / N)
    E_percentEvents = E_numEvents / N

    # CI computation (vector-safe)
    c = -erfinv(-c_perc) / (Std_percentEvents + 1e-12)
    CI_lower = (E_numEvents - c) / N
    CI_upper = (E_numEvents + c) / N

    predIdx = np.argmax(scores_all, axis=1)

    diagnosis = {}
    if "V" in classNamesArr:
        V_idx = classNamesArr.index("V")
        PVC_percent = np.mean(predIdx == V_idx)
        diagnosis["FrequentPVCs"] = bool(PVC_percent > 0.10)
        diagnosis["PVC_percent"] = float(PVC_percent)

    if "L" in classNamesArr:
        L_idx = classNamesArr.index("L")
        diagnosis["LBBB"] = bool(np.any(predIdx == L_idx))
    if "R" in classNamesArr:
        R_idx = classNamesArr.index("R")
        diagnosis["RBBB"] = bool(np.any(predIdx == R_idx))

    if "/" in classNamesArr:
        P_idx = classNamesArr.index("/")
        Paced_percent = float(np.mean(predIdx == P_idx))
        diagnosis["PacedRhythm"] = bool(Paced_percent > 0.80)
        diagnosis["Paced_percent"] = Paced_percent

    if inputDim > 1:
        HR = Dn[:, 0, -1] if Dn.ndim == 3 else Dn[:, -1]
        diagnosis["SinusTachy"] = bool(np.mean(HR > 100) > 0.05)
        diagnosis["SinusBrady"] = bool(np.mean(HR < 50) > 0.05)

    if "[" in classNamesArr and "]" in classNamesArr:
        VF_start = classNamesArr.index("[")
        VF_end = classNamesArr.index("]")
        diagnosis["VF_detected"] = bool(np.any((predIdx == VF_start) | (predIdx == VF_end)))
    else:
        diagnosis["VF_detected"] = False

    results = {
        "E_percentEvents": E_percentEvents,
        "Std_percentEvents": Std_percentEvents,
        "CI_lower": CI_lower,
        "CI_upper": CI_upper,
        "scores_all": scores_all,
        "numClasses": numClasses,
        "classNames": classNamesArr,
        "diagnosis": diagnosis
    }
    return results
