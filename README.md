# OBMO: One Bounding Box Multiple Objects for Monocular 3D Object Detection

This project is based on our paper: **OBMO: One Bounding Box Multiple Objects for Monocular 3D Object Detection** and monocular 3D detectors: **Rethinking Pseudo-LiDAR Representation(PatchNet)** and **Pseudo-lidar from visual depth estimation: Bridging the gap in 3d object detection for autonomous driving(Pseudo-LiDAR)**.

## Usage

The Usage is similar to PatchNet. And you should first install the environment and prepare the data, refer to `README_ori.md`.

```sh
cd #root/tools/data_prepare
python OBMO_data_prepare.py --gen_train --gen_val --gen_val_rgb_detection --car_only
mv *.pickle ../../data/KITTI/pickle_files
```

After that, move to the workspace and train the model:

```sh
cd #root
cd experiments/patchnet_OBMO
python ../../tools/train_val.py --obmo --config config_OBMO.yaml
```

Finally, generate the results using the trained model and evaluate the generated results:

```sh
python ../../tools/train_val.py --obmo --config config_OBMO.yaml --e
../../tools/kitti_eval/evaluate_object_3d_offline_ap40 ../../data/KITTI/object/training/label_2 ./output
```

## Offline version

We offer [an offline version](tools/offline_OBMO.py) to quickly test whether OBMO module benefits your model. And you can use it to choose super-parameters.

``` sh
python tools/offline_OBMO.py [pred] [gt]
```

We tested the offline version on models such as PatchNet, and the performance were greatly improved. 

Note that the offline results do not necessarily represent the results after training with the OBMO module.

## Acknowlegment

This code benefits from the excellent works: [PatchNet](https://github.com/xinzhuma/patchnet), [FPointNet](https://github.com/charlesq34/frustum-pointnets), [DORN](https://github.com/hufu6371/DORN) and [pseudo-LIDAR](https://github.com/mileyan/pseudo_lidar).
