from collections import namedtuple
from itertools import groupby

Class = namedtuple('Class', 'class_name class_index group_name')

gtsrb_classes: list[Class] = [
    Class('speed limit 20km/h', 0, 'speed limit signs'),
    Class('speed limit 30km/h', 1, 'speed limit signs'),
    Class('speed limit 50km/h', 2, 'speed limit signs'),
    Class('speed limit 60km/h', 3, 'speed limit signs'),
    Class('speed limit 70km/h', 4, 'speed limit signs'),
    Class('speed limit 80km/h', 5, 'speed limit signs'),
    Class('speed limit 100km/h', 7, 'speed limit signs'),
    Class('speed limit 120km/h', 8, 'speed limit signs'),

    Class('no passing', 9, 'other prohibitory signs'),
    Class('no passing for trucks', 10, 'other prohibitory signs'),
    Class('no vehicles', 15, 'other prohibitory signs'),
    Class('no trucks', 16, 'other prohibitory signs'),

    Class('priority road', 12, 'unique signs'),
    Class('yield', 13, 'unique signs'),
    Class('stop', 14, 'unique signs'),
    Class('no entry', 17, 'unique signs'),

    Class('right of way at next intersection', 11, 'danger signs'),
    Class('general caution', 18, 'danger signs'),
    Class('dangerous curve left', 19, 'danger signs'),
    Class('dangerous curve right', 20, 'danger signs'),
    Class('winding road', 21, 'danger signs'),
    Class('bumpy road', 22, 'danger signs'),
    Class('slippery road', 23, 'danger signs'),
    Class('road narrows on the right', 24, 'danger signs'),
    Class('road work', 25, 'danger signs'),
    Class('traffic lights', 26, 'danger signs'),
    Class('pedestrians', 27, 'danger signs'),
    Class('children crossing', 28, 'danger signs'),
    Class('bicycles crossing', 29, 'danger signs'),
    Class('beware of ice/snow', 30, 'danger signs'),
    Class('wild animals crossing', 31, 'danger signs'),

    Class('turn right', 33, 'mandatory signs'),
    Class('turn left', 34, 'mandatory signs'),
    Class('only straight', 35, 'mandatory signs'),
    Class('only straight or right', 36, 'mandatory signs'),
    Class('only straight or left', 37, 'mandatory signs'),
    Class('keep right', 38, 'mandatory signs'),
    Class('keep left', 39, 'mandatory signs'),
    Class('roundabout mandatory', 40, 'mandatory signs'),

    Class('speed limit 80km/h end', 6, 'derestriction signs'),
    Class('end of all speed and passing limits', 32, 'derestriction signs'),
    Class('end of no overtaking limit', 41, 'derestriction signs'),
    Class('end of no overtaking limit for trucks', 42, 'derestriction signs'),
]

groups: list[str] = [
    ['airplane', 'ship', 'dog'],
    ['automobile', 'truck', 'cat'],
    ['bird', 'airplane', 'dog'],
    ['cat', 'dog', 'frog'],
    ['deer', 'horse', 'truck'],
    ['dog', 'cat', 'bird'],
    ['frog', 'ship', 'truck'],
    ['horse', 'deer', 'airplane'],
    ['ship', 'airplane', 'deer'],
    ['truck', 'automobile', 'airplane']
]

cifar10_labels: dict[str, int] = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

def to_list_of_lists(classes: list[Class]) -> list[list[Class]]:
    return [list(group) for _, group in groupby(sorted(classes, key=lambda x: x.group_name), key=lambda x: x.group_name)]

gtsrb_groups: list[list[Class]] = to_list_of_lists(gtsrb_classes)
cifar10_groups: list[list[int]] = [[cifar10_labels[e] for e in g] for g in groups]