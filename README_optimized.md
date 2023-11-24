Passi per installare le dipendenze necessarie per l'esecuzione del progetto.
```bash
conda create --name mmpose python=3.8 -y
conda activate mmpose
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
```

A questo punto clonare il repository se non lo si e installare MMPose da sorgente:
```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Testare il codice con demo live usando la webcam e il modello VideoPose3D più piccolo.

* Scaricare config file e pesi del modello VideoPose3D più piccolo:
```bash
mim download mmpose --config video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m --dest ./configs/body_3d_keypoint/image_pose_lift/h36m
```

* Eseguire il codice di demo live:
```bash
python demo/inferencer_demo.py webcam --pose3d configs/body_3d_keypoint/image_pose_lift/h36m/video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m.py --pose3d-weights configs/body_3d_keypoint/image_pose_lift/h36m/videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth
```

