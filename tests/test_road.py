import numpy as np
import torch

from dms.road.segmentation import CITYSCAPES_TO_L1, IGNORE, L1_CLASSES, ConfusionMatrix, cityscapes_to_l1


def test_every_cityscapes_class_maps_to_an_idd_level1_class():
    assert sorted(CITYSCAPES_TO_L1) == list(range(19))
    assert set(CITYSCAPES_TO_L1.values()) == set(range(len(L1_CLASSES)))
    pred = cityscapes_to_l1(torch.tensor([[0, 10, 13, 11]]))
    assert pred.tolist() == [[0, 6, 3, 2]]  # road, sky, car, person


def test_confusion_matrix_iou_ignores_unlabelled_pixels():
    cm = ConfusionMatrix(2)
    cm.update(torch.tensor([0, 0, 1, 1, 1]), torch.tensor([0, 1, 1, 1, IGNORE]))
    iou = cm.iou()
    assert np.isclose(iou[0], 1 / 2) and np.isclose(iou[1], 2 / 3)
    assert np.isclose(cm.pixel_acc(), 3 / 4)
