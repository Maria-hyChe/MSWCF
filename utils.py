import numpy as np

IMAGE_MEANS =np.array([303.21, 510.85, 384.11, 3715.82, 3006.94, 1796.22])
IMAGE_STDS = np.array([103.54, 112.02, 147.53, 775.25, 726.39, 425.84])

LABEL_CLASSES = [0, 101,104,115,127]
LABEL_CLASS_COLORMAP = { # Color map for Chesapeake dataset
    0:  (0, 0, 0),
    101: (70, 107, 159),
    104: (209, 222, 248),
    115: (222, 197, 197),
    127: (217, 146, 130)
}

LABEL_IDX_COLORMAP = {
    idx: LABEL_CLASS_COLORMAP[c]
    for idx, c in enumerate(LABEL_CLASSES)
}

def get_label_class_to_idx_map():
    label_to_idx_map = []
    idx = 0
    for i in range(LABEL_CLASSES[-1]+1):
        if i in LABEL_CLASSES:
            label_to_idx_map.append(idx)
            idx += 1
        else:
            label_to_idx_map.append(0)
    label_to_idx_map = np.array(label_to_idx_map).astype(np.int64)
    return label_to_idx_map

LABEL_CLASS_TO_IDX_MAP = get_label_class_to_idx_map()

if __name__ == '__main__':
    print(LABEL_CLASS_TO_IDX_MAP)