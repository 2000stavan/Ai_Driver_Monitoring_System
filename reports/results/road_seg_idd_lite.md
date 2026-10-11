# Road segmentation on Indian roads - IDD Lite val

Model: `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, input 416x288, 1403 train / 204 val images, device mps.

IoU per IDD level-1 class (zero-shot = Cityscapes classes merged into IDD level-1):

|                                      |   mIoU |   pixel_acc |   drivable |   non_drivable |   living_things |   vehicles |   roadside_objects |   far_objects |   sky |
|:-------------------------------------|-------:|------------:|-----------:|---------------:|----------------:|-----------:|-------------------:|--------------:|------:|
| Cityscapes only (zero-shot, Western) |  0.565 |       0.813 |      0.852 |          0.227 |           0.38  |      0.634 |              0.25  |         0.681 | 0.928 |
| IDD Lite fine-tuned (Indian)         |  0.711 |       0.897 |      0.941 |          0.466 |           0.535 |      0.782 |              0.522 |         0.782 | 0.953 |

Fine-tuning: 30 epochs, best val mIoU 0.711 at epoch 27. Note: val is used for checkpoint selection (IDD Lite has no public test labels), so the fine-tuned number is slightly optimistic.

|   epoch |   train_loss |   val_miou |
|--------:|-------------:|-----------:|
|       1 |       0.4665 |     0.623  |
|       2 |       0.3443 |     0.6417 |
|       3 |       0.3222 |     0.6608 |
|       4 |       0.3097 |     0.6666 |
|       5 |       0.3028 |     0.6637 |
|       6 |       0.2914 |     0.6779 |
|       7 |       0.2809 |     0.6672 |
|       8 |       0.2771 |     0.6821 |
|       9 |       0.2779 |     0.6849 |
|      10 |       0.2626 |     0.6864 |
|      11 |       0.2559 |     0.687  |
|      12 |       0.2574 |     0.6899 |
|      13 |       0.2517 |     0.6954 |
|      14 |       0.2447 |     0.6941 |
|      15 |       0.2431 |     0.6939 |
|      16 |       0.2374 |     0.7037 |
|      17 |       0.2369 |     0.7005 |
|      18 |       0.233  |     0.6962 |
|      19 |       0.2282 |     0.7054 |
|      20 |       0.2266 |     0.7054 |
|      21 |       0.2225 |     0.7108 |
|      22 |       0.2233 |     0.7075 |
|      23 |       0.2203 |     0.7065 |
|      24 |       0.2178 |     0.7075 |
|      25 |       0.217  |     0.7077 |
|      26 |       0.2131 |     0.7083 |
|      27 |       0.2135 |     0.7114 |
|      28 |       0.2115 |     0.7102 |
|      29 |       0.2088 |     0.7098 |
|      30 |       0.2088 |     0.711  |