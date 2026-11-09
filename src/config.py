from dataclasses import dataclass

@dataclass
class SolarFilNetV4Config:
    data_root: str = "/content/MAGFiLO_1.0_Kaggle_2026"
    work_dir: str = "./work_solarfilnet_v4"
    cache_dir: str = "/content/cache_v4"
    ann_filename: str = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    # Arsitektur Utama SolarFilNet-V4 (Clean Single-Model)
    backbone: str = "pvt_v2_b1"
    in_channels: int = 3
    orig_size: int = 2048
    image_size: int = 1024
    use_fullres_branch: bool = True
    use_dffn: bool = True

    # Konfigurasi Pelatihan
    mode: str = "val"
    n_folds: int = 5
    val_fold: int = 0
    epochs: int = 20
    val_every: int = 2
    batch_size: int = 4
    grad_accum_steps: int = 2
    lr: float = 1.5e-4
    backbone_lr_mult: float = 0.5
    min_lr: float = 1.0e-6
    weight_decay: float = 1.0e-4
    grad_clip: float = 2.0
    ema_decay: float = 0.998
    mixed_precision: bool = True
    num_workers: int = 4
    seed: int = 2026
    auto_resume: bool = True
    skip_tune: bool = False
    init_ckpt: str = ""

    # Komposisi Compound Loss Optimal V4 (Focal-Tversky + PCL + Topology)
    w_focal: float = 1.00
    w_dice: float = 0.85
    w_tversky: float = 0.50
    w_pcl: float = 0.25
    w_cldice: float = 0.25
    w_boundary: float = 0.40
    w_spine: float = 0.40
    w_edge: float = 0.20
    w_lovasz: float = 0.20

    # Parameter Post-Processing & Astro-Kinematic Anchor
    spine_boost_alpha: float = 0.15
    edge_suppress_beta: float = 0.12
    pp_thr: float = 0.46
    pp_join_px: int = 5
    pp_min_area: int = 320
    pp_min_score: float = 0.80
    pp_fill_holes: int = 1
    tta_views: int = 4
    tune_rounds: int = 2
    eval_kinematic_anchor: bool = True
    anchor_max_gap_hours: float = 72.0
