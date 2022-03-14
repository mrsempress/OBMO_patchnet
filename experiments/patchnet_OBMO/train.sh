python ../../tools/train_val.py --obmo --config config_OBMO.yaml
python ../../tools/train_val.py --obmo --config config_OBMO.yaml --evaluation
../../tools/kitti_eval/evaluate_object_3d_offline_ap11 ../../data/KITTI/object/training/label_2 ./output
../../tools/kitti_eval/evaluate_object_3d_offline_ap40 ../../data/KITTI/object/training/label_2 ./output