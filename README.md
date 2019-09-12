# Self-Supervised Depth Estimation in Pytorch


![](clip_1.gif) ![](clip_2.gif)

*Depth prediction from ours (2nd column), SfmLearner by Zhou et al. (3rd column), and GeoNet by Yin et al. (4th column). The first row displays results from a testing patient and camera which are not seen during training. The second row is from a training patient and camera. Depth maps are re-scaled with sparse depth maps generated from SfM results for visualization.*


This codebase implements the system described in the paper:

**Self-supervised Learning for Dense Depth Estimation in Monocular Endoscopy**

Xingtong Liu, Ayushi Sinha, Mathias Unberath, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, Austin Reiter

In International Workshop on Computer-Assisted and Robotic Endoscopy 2018 (**Best paper award & Best presentation award**)

The journal version of this paper has been submitted to IEEE Transactions on Medical Imaging under revision.

Please contact [**Xingtong Liu**](http://www.cs.jhu.edu/~xingtongl/) (xliu89@jh.edu) if you have any questions.

Please cite [CARE Workshop](https://link.springer.com/chapter/10.1007/978-3-030-01201-4_15) or [TMI Submission Arxiv Preprint](https://arxiv.org/abs/1902.07766) if you use the code here.
```
@incollection{liu2018self,
  title={Self-supervised Learning for Dense Depth Estimation in Monocular Endoscopy},
  author={Liu, Xingtong and Sinha, Ayushi and Unberath, Mathias and Ishii, Masaru and Hager, Gregory D and Taylor, Russell H and Reiter, Austin},
  booktitle={OR 2.0 Context-Aware Operating Theaters, Computer Assisted Robotic Endoscopy, Clinical Image-Based Procedures, and Skin Image Analysis},
  pages={128--138},
  year={2018},
  publisher={Springer}
}
```

```
@ARTICLE{2019arXiv190207766L,
       author = {{Liu}, Xingtong and {Sinha}, Ayushi and {Ishii}, Masaru and
         {Hager}, Gregory D. and {Reiter}, Austin and {Taylor}, Russell H. and
         {Unberath}, Mathias},
        title = "{Self-supervised Learning for Dense Depth Estimation in Monocular Endoscopy}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Statistics - Machine Learning},
         year = "2019",
        month = "Feb",
          eid = {arXiv:1902.07766},
        pages = {arXiv:1902.07766},
archivePrefix = {arXiv},
       eprint = {1902.07766},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/\#abs/2019arXiv190207766L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
## Instructions

1. Install all necessary python packages: Pytorch, OpenCV, numpy, tqdm, pathlib, torchsummary, tensorboardX, albumentations, argparse, pickle, plyfile, yaml, datetime, shutil, matplotlib

2. Generate training data from training videos using Structure from Motion (SfM) or Simultaneous Localization and Mapping (SLAM). In terms of the format, please refer to one training data example in this repository.

3. Run ```student_training.py``` with proper arguments for self-supervised learning. One example is:
```
/path/to/python /path/to/student_training.py --id_range 1 11 --input_downsampling 4.0 --network_downsampling 64 --adjacent_range 5 30 --torchsummary_input_size 256 320 --batch_size 8 --num_workers 8 --validation_interval 1 --display_interval 50 --dcl_weight 0.1 --sfl_weight 20.0 --max_lr 1.0e-3 --min_lr 1.0e-4 --inlier_percentage 0.99 --testing_patient_id 4 --load_intermediate_data --use_hsv_colorspace --number_epoch 100 --num_iter 1000 --architecture_summary --training_result_root "/path/to/training/directory" --training_data_root "/path/to/training/data"
```

4. Run ```evaluate.py``` to generate evaluation results. Apply any rigid registration algorithm to register the predicted point clouds to the corresponding CT model to calculate residual errors (this step may require manual point cloud initialization).


## Disclaimer

This codebase is only experimental and not ready for clinical applications.

Authors are not responsible for any accidents related to this repository.

This codebase is only allowed for non-commercial usage.


