"""
convert_to_tf.py
Take your existing arrhythmia_model and binary_model packages (assumed present as Python modules
with a create_model() and loadWeights() interface) and export them as Keras H5 and SavedModel.

Usage:
    from models.convert_to_tf import convert_models_to_tf
    convert_models_to_tf(arrhythmia_pkg_path, binary_pkg_path, out_dir='.')
"""
import os
import tensorflow as tf

def convert_models_to_tf(arrhythmia_pkg, binary_pkg, out_dir='.'):
    """
    arrhythmia_pkg and binary_pkg should be importable Python packages (already on sys.path)
    that have:
      - model.create_model()
      - loadWeights(model, filename=..., debug=False)
    """
    # ARRHYTHMIA
    import importlib
    arr_mod = importlib.import_module(arrhythmia_pkg)
    arr_model_mod = importlib.import_module(f"{arrhythmia_pkg}.model")
    load_arr_weights = getattr(arr_mod, "loadWeights", None) or getattr(arr_model_mod, "loadWeights", None)
    arr_net = arr_model_mod.create_model()
    if load_arr_weights is None:
        raise RuntimeError("arrhythmia package missing loadWeights")
    arr_weights_path = os.path.join(os.path.dirname(arr_mod.__file__), "weights.h5")
    load_arr_weights(arr_net, filename=arr_weights_path, debug=False)
    arr_h5 = os.path.join(out_dir, "arrhythmia_model.h5")
    arr_saved = os.path.join(out_dir, "arrhythmia_saved_model")
    arr_net.save(arr_h5)
    tf.saved_model.save(arr_net, arr_saved)
    print(f"[convert_to_tf] Saved arrhythmia model -> {arr_h5} and {arr_saved}")

    # BINARY
    bin_mod = importlib.import_module(binary_pkg)
    bin_model_mod = importlib.import_module(f"{binary_pkg}.model")
    load_bin_weights = getattr(bin_mod, "loadWeights", None) or getattr(bin_model_mod, "loadWeights", None)
    bin_net = bin_model_mod.create_model()
    if load_bin_weights is None:
        raise RuntimeError("binary package missing loadWeights")
    bin_weights_path = os.path.join(os.path.dirname(bin_mod.__file__), "weights.h5")
    load_bin_weights(bin_net, filename=bin_weights_path, debug=False)
    bin_h5 = os.path.join(out_dir, "binary_model.h5")
    bin_saved = os.path.join(out_dir, "binary_saved_model")
    bin_net.save(bin_h5)
    tf.saved_model.save(bin_net, bin_saved)
    print(f"[convert_to_tf] Saved binary model -> {bin_h5} and {bin_saved}")

    return {"arr_h5": arr_h5, "arr_saved": arr_saved, "bin_h5": bin_h5, "bin_saved": bin_saved}
