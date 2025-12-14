"""
RTIOD/LTDv2 Thermal Detection Training Script
==============================================
- All parameters read from config.yaml (Hydra)


Usage:
  python train.py                           # Use config defaults
"""

from ultralytics import YOLO
import hydra
from omegaconf import DictConfig
import torch
import os


@hydra.main(config_path='config', config_name='config', version_base="1.3")
def main(cfg: DictConfig):
    """Train YOLO11m on LTDv2 thermal dataset with optimized settings."""
    
    # =========================================================================
    # DEVICE SETUP
    # =========================================================================
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("YOLO11M THERMAL DETECTION - OPTIMIZED TRAINING")
    print(f"{'='*70}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} | VRAM: {gpu_memory:.1f} GB")
    else:
        print("WARNING: Running on CPU (very slow)")
    
    print(f"{'='*70}\n")
    
    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    # Check if resuming from checkpoint or starting fresh
    model_path = cfg.get('modelCheckpoint', 'yolo11m.pt')
    
    if os.path.exists(model_path) and 'epoch' in model_path:
        print(f"Resuming from checkpoint: {model_path}")
        model = YOLO(model_path)
    else:
        print(f"Starting fresh with: yolo11m.pt")
        model = YOLO('yolo11m.pt')
    
    # =========================================================================
    # PRINT CONFIGURATION
    # =========================================================================
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Epochs:        {cfg.epochs}")
    print(f"Image Size:    {cfg.imgsz}px")
    print(f"Batch Size:    {cfg.batch}")
    print(f"Workers:       {cfg.workers}")
    print(f"Optimizer:     {cfg.optimizer}")
    print(f"Learning Rate: {cfg.lr0} → {cfg.lr0 * cfg.lrf:.5f} (lrf={cfg.lrf})")
    print(f"Warmup:        {cfg.warmup_epochs} epochs")
    print(f"Patience:      {cfg.patience}")
    print(f"Fraction:      {cfg.fraction}")
    print(f"{'='*70}")
    print(f"Mosaic:        {cfg.mosaic} (close at {cfg.close_mosaic})")
    print(f"Mixup:         {cfg.mixup}")
    print(f"Copy-Paste:    {cfg.copy_paste}")
    print(f"HSV-V:         {cfg.hsv_v} (thermal brightness)")
    print(f"Scale:         {cfg.scale}")
    print(f"{'='*70}\n")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    results = model.train(
        # Data
        data="data/data.yaml",
        
        # Training schedule
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=device,
        workers=cfg.workers,
        cache=cfg.cache,
        
        # Learning rate schedule (CRITICAL FIX)
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        cos_lr=cfg.cos_lr,
        
        # Warmup (CRITICAL FIX - prevents epoch 6 crash)
        warmup_epochs=cfg.warmup_epochs,
        warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr,
        
        # Optimizer
        optimizer=cfg.optimizer,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        
        # Loss weights
        box=cfg.box,
        cls=cfg.cls,
        dfl=cfg.dfl,
        
        # Thermal-specific augmentation
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        
        # Geometric augmentation
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        
        # Multi-image augmentation
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        auto_augment=cfg.auto_augment,
        erasing=cfg.erasing,
        close_mosaic=cfg.close_mosaic,
        
        # Early stopping (CRITICAL FIX)
        patience=cfg.patience,
        
        # Training settings
        amp=cfg.amp,
        pretrained=cfg.pretrained,
        save_period=cfg.save_period,
        plots=cfg.plots,
        verbose=cfg.verbose,
        seed=cfg.seed,
        fraction=cfg.fraction,
        
        # Validation inference settings
        conf=cfg.conf,
        iou=cfg.iou,
        max_det=cfg.max_det,
        
        # Output
        project='runs/train',
        name='yolo11m_thermal_optimized',
        exist_ok=True
    )
    
    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================
    print(f"\n{'='*70}")
    print("FINAL VALIDATION")
    print(f"{'='*70}\n")
    
    val_results = model.val(
        data="data/data.yaml",
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=device,
        workers=cfg.workers,
        save_json=True,
        plots=True
    )
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"mAP@50:       {val_results.box.map50:.4f}  (Target: 0.55+)")
    print(f"mAP@50-95:    {val_results.box.map:.4f}")
    print(f"Precision:    {val_results.box.mp:.4f}")
    print(f"Recall:       {val_results.box.mr:.4f}")
    print(f"{'='*70}")
    
    # Performance assessment
    if val_results.box.map50 >= 0.60:
        print("🎉 EXCELLENT! Well above target!")
    elif val_results.box.map50 >= 0.55:
        print("✅ SUCCESS! Target achieved!")
    elif val_results.box.map50 >= 0.50:
        print("📈 Good progress, close to target")
    else:
        print("⚠️  Below target - check training curves")
    
    print(f"\n📁 Best weights: runs/train/yolo11m_thermal_optimized/weights/best.pt")
    print(f"📁 Last weights: runs/train/yolo11m_thermal_optimized/weights/last.pt\n")
    
    return results


if __name__ == "__main__":
    main()
