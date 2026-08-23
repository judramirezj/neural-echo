"""fsaverage5 surface parcellation used by metric.py: groups the Destrieux
(aparc.a2009s) atlas's fine-grained labels into ~25 coarser anatomical
lobule groups, per hemisphere, so metric.py can compare candidate vs.
reference brain responses region-by-region rather than vertex-by-vertex.
Fetched once, cached to disk by nilearn's own data dir so subsequent runs
are instant and offline-safe.
"""
import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

N_VERTICES_PER_HEMI = 10242
N_VERTICES = 2 * N_VERTICES_PER_HEMI

# Groups Destrieux (aparc.a2009s) labels into coarser anatomical lobules by
# substring match on the label name. Order matters: the first matching group
# wins, so more specific groups should precede more general ones where their
# substrings could otherwise overlap.
LOBULE_RULES = [
    ("auditory_primary", ["temp_sup-g_t_transv", "s_temporal_transverse"]),
    ("auditory_assoc", ["temp_sup-plan_tempo", "temp_sup-plan_polar", "temp_sup-lateral", "s_temporal_sup"]),
    ("temporal_mid", ["temporal_middle", "s_temporal_inf"]),
    ("temporal_inferior", ["temporal_inf", "oc-temp_lat-fusifor", "s_oc-temp_lat"]),
    ("temporal_pole", ["pole_temporal", "temporal_transverse_pole"]),
    ("parahippocampal", ["oc-temp_med-parahip", "oc-temp_med-lingual", "s_oc-temp_med", "s_collat_transv"]),
    ("frontal_inferior", ["front_inf", "s_front_inf", "triangul", "opercular", "orbital"]),
    ("frontal_mid", ["front_middle", "s_front_middle"]),
    ("frontal_superior", ["front_sup", "s_front_sup"]),
    ("motor_premotor", ["precentral", "s_precentral"]),
    ("orbitofrontal", ["rectus", "orbital-h_shaped", "orbital_lateral", "orbital_medial", "subcallosal"]),
    ("somatosensory", ["postcentral", "s_postcentral"]),
    ("parietal_superior", ["parietal_sup", "s_parieto_occipital"]),
    ("parietal_inferior", ["pariet_inf", "angular", "supramar", "s_intrapariet"]),
    ("precuneus", ["precuneus", "subparietal"]),
    ("visual_primary", ["calcarine", "cuneus"]),
    ("visual_assoc", ["occip", "lingual", "s_oc_middle", "s_oc_sup", "s_calcarine"]),
    ("insula_anterior", ["ins_lg_and_s_cent", "s_circular_insula_ant", "s_circular_insula_sup"]),
    ("insula_posterior", ["insular", "s_circular_insula_inf"]),
    ("cingulate_anterior", ["cingul-ant", "s_cingul-marginalis", "pericallosal"]),
    ("cingulate_mid", ["cingul-mid-ant", "cingul-mid-post"]),
    ("cingulate_posterior", ["cingul-post"]),
    ("sylvian", ["lat_fis"]),
    ("central", ["s_central", "s_interm_prim"]),
    ("motor_medial", ["paracentral", "subcentral"]),
    ("prefrontal_polar", ["frontomargin", "transv_frontopol"]),
]


def classify_label(name: str) -> str:
    n = name.lower()
    if "medial_wall" in n or "unknown" in n:
        return "exclude"
    for group, substrings in LOBULE_RULES:
        if any(s in n for s in substrings):
            return group
    return "other"


@lru_cache(maxsize=1)
def build_lobule_regions() -> dict[str, np.ndarray]:
    """Destrieux labels -> {"{group}_left"/"{group}_right": vertex_indices}
    over all N_VERTICES (both hemispheres, right offset by N_VERTICES_PER_HEMI).
    """
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [lbl.decode() if isinstance(lbl, bytes) else str(lbl) for lbl in atlas["labels"]]

    regions: dict[str, list[np.ndarray]] = {}
    for hemi, map_key, offset in [("left", "map_left", 0), ("right", "map_right", N_VERTICES_PER_HEMI)]:
        hemi_map = np.asarray(atlas[map_key])
        for label_id, name in enumerate(labels):
            group = classify_label(name)
            if group == "exclude":
                continue
            idx = np.where(hemi_map == label_id)[0]
            if idx.size == 0:
                continue
            regions.setdefault(f"{group}_{hemi}", []).append(idx + offset)

    return {k: np.concatenate(v) for k, v in regions.items()}
