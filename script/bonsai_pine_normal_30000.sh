python launch.py --config configs/edit-ctn.yaml --train --gpu 0 trainer.max_steps=1500 system.cache_dir="bonsai_pine_normal_30000" system.seg_prompt="bonsai" system.prompt_processor.prompt="turn bonsai into a pine tree" system.guidance.control_type="normal" system.max_densify_percent=0.01 system.anchor_weight_init_g0=1 system.anchor_weight_init=0.5 system.anchor_weight_multiplier=1.5 system.loss.lambda_anchor_color=0 system.loss.lambda_anchor_geo=5 system.loss.lambda_anchor_scale=5 system.loss.lambda_anchor_opacity=0 system.densify_from_iter=100 system.densify_until_iter=5000 system.densification_interval=300 data.source=/home/lui/cv2/GaussianEditor/dataset/bonsai system.gs_source=/home/lui/cv2/GaussianEditor/dataset/bonsai/point_cloud/iteration_30000/point_cloud.ply