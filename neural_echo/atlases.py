"""fsaverage5 surface masks: anatomical (audio-relevant cortex) and network
(Yeo-7) parcellations, used by metric.py. Fetched once, cached to disk by
nilearn's own data dir so subsequent runs are instant and offline-safe.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

N_VERTICES_PER_HEMI = 10242
N_VERTICES = 2 * N_VERTICES_PER_HEMI

# Destrieux (aparc.a2009s) label substrings covering primary/secondary auditory
# cortex, superior temporal sulcus, inferior frontal gyrus (language), and
# insula (salience) — the brief's "Heschl's, planum temporale, STS, IFG, insula"
# list. Matched case-sensitively against nilearn's fetch_atlas_surf_destrieux
# label names (e.g. "G_temp_sup-G_T_transv", "S_temporal_sup", ...).
AUDIO_RELEVANT_DESTRIEUX_SUBSTRINGS = [
    "G_temp_sup",       # superior temporal gyrus (incl. Heschl's, planum temporale)
    "S_temporal_sup",   # superior temporal sulcus
    "Lat_Fis-post",     # posterior lateral fissure (near planum temporale)
    "G_front_inf",       # inferior frontal gyrus (Broca's area / IFG)
    "G_Ins_lg",          # long insular gyrus
    "G_insular_short",   # short insular gyri
    "S_circular_insula",  # circular sulcus of the insula
]


def fetch_anatomical_mask() -> np.ndarray:
    """Boolean mask over N_VERTICES (both hemispheres) selecting Destrieux
    ROIs relevant to audio/language/salience processing.
    """
    from nilearn import datasets

    destrieux = datasets.fetch_atlas_surf_destrieux()
    labels = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in destrieux.labels]
    keep_label_ids = {
        i for i, lbl in enumerate(labels)
        if any(sub in lbl for sub in AUDIO_RELEVANT_DESTRIEUX_SUBSTRINGS)
    }
    if not keep_label_ids:
        raise RuntimeError("No Destrieux labels matched the audio-relevant substrings")

    map_left = np.asarray(destrieux.map_left)
    map_right = np.asarray(destrieux.map_right)
    mask_left = np.isin(map_left, list(keep_label_ids))
    mask_right = np.isin(map_right, list(keep_label_ids))
    mask = np.concatenate([mask_left, mask_right])
    if mask.shape[0] != N_VERTICES:
        raise RuntimeError(
            f"Destrieux mask has {mask.shape[0]} vertices, expected {N_VERTICES} "
            "(fsaverage5 mismatch)"
        )
    logger.info("Anatomical mask: %d / %d vertices", mask.sum(), N_VERTICES)
    return mask


_YEO_NETWORK_NAMES = [
    "Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
    "Limbic", "Frontoparietal", "Default",
]


def fetch_yeo7_network_labels() -> np.ndarray | None:
    """Per-vertex Yeo-7 network id (1..7, 0=unlabeled) over N_VERTICES, or
    None if the atlas can't be fetched (e.g. no network access) — callers
    should fall back to the anatomical mask alone for network geometry.
    """
    try:
        from nilearn import datasets, surface

        yeo = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
        fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
        left = surface.vol_to_surf(
            yeo.maps if hasattr(yeo, "maps") else yeo["thick_7"],
            fsaverage.pial_left,
            interpolation="nearest_most_frequent",
        )
        right = surface.vol_to_surf(
            yeo.maps if hasattr(yeo, "maps") else yeo["thick_7"],
            fsaverage.pial_right,
            interpolation="nearest_most_frequent",
        )
        labels = np.concatenate([np.ravel(left), np.ravel(right)]).round().astype(int)
        if labels.shape[0] != N_VERTICES:
            raise RuntimeError(f"Yeo projection has {labels.shape[0]} vertices, expected {N_VERTICES}")
        return labels
    except Exception:
        logger.warning("Could not fetch/project Yeo-7 atlas; network geometry will use anatomical sub-ROIs only", exc_info=True)
        return None


def network_name(network_id: int) -> str:
    if 1 <= network_id <= len(_YEO_NETWORK_NAMES):
        return _YEO_NETWORK_NAMES[network_id - 1]
    return "Unlabeled"
