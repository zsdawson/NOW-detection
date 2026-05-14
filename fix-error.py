Dataset folder: /project/ai-for-trap-processing-now/output-new-train
Image folder: /project/ai-for-trap-processing-now/output-new-train/images/train
Label folder: /project/ai-for-trap-processing-now/output-new-train/labels/train
Deleting old seed_good folder...

Seed dataset created.
Copied usable images: 48
Kept original labels: 4
Fixed 270 labels: 44
Skipped bad: 2
Missing: 0

Data YAML:
path: /project/ai-for-trap-processing-now/seed_good
train: images/train
val: images/val

names:
  0: NOW

Train images: 38
Val images: 10

Starting mini-model training...
Ultralytics 8.3.78 🚀 Python-3.9.21 torch-2.6.0+cu124 CUDA:0 (Tesla V100-SXM2-32GB, 32494MiB)
engine/trainer: task=detect, mode=train, model=yolov8n.pt, data=/project/ai-for-trap-processing-now/seed_good/data.yaml, epochs=50, time=None, patience=100, batch=4, imgsz=960, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=seed_orientation_checker2, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=/project/ai-for-trap-processing-now/test_venv/runs/detect/seed_orientation_checker2
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]           
Model summary: 129 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs

Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /project/ai-for-trap-processing-now/seed_good/labels/train... 38 images, 0 backgrounds, 0 corrupt: 100%|██████████| 38/38 [00:00<00:00, 69.24it/s]
train: New cache created: /project/ai-for-trap-processing-now/seed_good/labels/train.cache

/project/ai-for-trap-processing-now/test_venv/lib64/python3.9/site-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 1, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
val: Scanning /project/ai-for-trap-processing-now/seed_good/labels/val... 10 images, 0 backgrounds, 0 corrupt: 100%|██████████| 10/10 [00:00<00:00, 312.29it/s]
val: New cache created: /project/ai-for-trap-processing-now/seed_good/labels/val.cache

/project/ai-for-trap-processing-now/test_venv/lib64/python3.9/site-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 16 worker processes in total. Our suggested max number of worker in current system is 1, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
No module named 'pandas'
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Image sizes 960 train, 960 val
Using 8 dataloader workers
Logging results to /project/ai-for-trap-processing-now/test_venv/runs/detect/seed_orientation_checker2
Starting training for 50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50       1.8G      2.018      3.614      1.257        119        960: 100%|██████████| 10/10 [00:27<00:00,  2.70s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 2/2 [00:02<00:00,  1.39s/it]
                   all         10        155     0.0103        0.2     0.0141    0.00559

---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[5], line 289
    285 print("\nStarting mini-model training...")
    287 model = YOLO(MODEL_NAME)
--> 289 model.train(
    290     data=str(data_yaml),
    291     epochs=EPOCHS,
    292     imgsz=IMG_SIZE,
    293     batch=BATCH,
    294     name="seed_orientation_checker",
    295 )
    297 print("\nDone.")
    298 print("Mini model should be in:")

File /project/ai-for-trap-processing-now/test_venv/lib64/python3.9/site-packages/ultralytics/engine/model.py:810, in Model.train(self, trainer, **kwargs)
    807     self.model = self.trainer.model
    809 self.trainer.hub_session = self.session  # attach optional HUB session
--> 810 self.trainer.train()
    811 # Update model and cfg after training
    812 if RANK in {-1, 0}:

File /project/ai-for-trap-processing-now/test_venv/lib64/python3.9/site-packages/ultralytics/engine/trainer.py:208, in BaseTrainer.train(self)
    205         ddp_cleanup(self, str(file))
    207 else:
--> 208     self._do_train(world_size)

File /project/ai-for-trap-processing-now/test_venv/lib64/python3.9/site-packages/ultralytics/engine/trainer.py:441, in BaseTrainer._do_train(self, world_size)
    439     # Save model
    440     if self.args.save or final_epoch:
--> 441         self.save_model()
    442         self.run_callbacks("on_model_save")
    444 # Scheduler

File /project/ai-for-trap-processing-now/test_venv/lib64/python3.9/site-packages/ultralytics/engine/trainer.py:530, in BaseTrainer.save_model(self)
    518 # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
    519 buffer = io.BytesIO()
    520 torch.save(
    521     {
    522         "epoch": self.epoch,
    523         "best_fitness": self.best_fitness,
    524         "model": None,  # resume and final checkpoints derive from EMA
    525         "ema": deepcopy(self.ema.ema).half(),
    526         "updates": self.ema.updates,
    527         "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
    528         "train_args": vars(self.args),  # save as dict
    529         "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
--> 530         "train_results": self.read_results_csv(),
    531         "date": datetime.now().isoformat(),
    532         "version": __version__,
    533         "license": "AGPL-3.0 (https://ultralytics.com/license)",
    534         "docs": "https://docs.ultralytics.com",
    535     },
    536     buffer,
    537 )
    538 serialized_ckpt = buffer.getvalue()  # get the serialized content to save
    540 # Save checkpoints

File /project/ai-for-trap-processing-now/test_venv/lib64/python3.9/site-packages/ultralytics/engine/trainer.py:510, in BaseTrainer.read_results_csv(self)
    508 def read_results_csv(self):
    509     """Read results.csv into a dict using pandas."""
--> 510     import pandas as pd  # scope for faster 'import ultralytics'
    512     return pd.read_csv(self.csv).to_dict(orient="list")

ModuleNotFoundError: No module named 'pandas'

Click to add a cell.

