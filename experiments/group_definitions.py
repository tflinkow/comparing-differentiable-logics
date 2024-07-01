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

cifar100_classes: list[Class] = [
    Class('beaver', 4, 'aquatic mammals'),
    Class('dolphin', 31, 'aquatic mammals'),
    Class('otter', 55, 'aquatic mammals'),
    Class('seal', 72, 'aquatic mammals'),
    Class('whale', 95, 'aquatic mammals'),

    Class('aquarium fish', 1, 'fish'),
    Class('flatfish', 33, 'fish'),
    Class('ray', 67, 'fish'),
    Class('shark', 73, 'fish'),
    Class('trout', 91, 'fish'),

    Class('orchids', 54, 'flowers'),
    Class('poppies', 62, 'flowers'),
    Class('roses', 70, 'flowers'),
    Class('sunflowers', 82, 'flowers'),
    Class('tulips', 92, 'flowers'),

    Class('bottles', 9, 'food containers'),
    Class('bowls', 10, 'food containers'),
    Class('cans', 16, 'food containers'),
    Class('cups', 29, 'food containers'),
    Class('plates', 61, 'food containers'),

    Class('apples', 0, 'fruit and vegetables'),
    Class('mushrooms', 51, 'fruit and vegetables'),
    Class('oranges', 53, 'fruit and vegetables'),
    Class('pears', 57, 'fruit and vegetables'),
    Class('sweet peppers', 83, 'fruit and vegetables'),

    Class('clock', 22, 'household electrical devices'),
    Class('computer keyboard', 25, 'household electrical devices'),
    Class('lamp', 40, 'household electrical devices'),
    Class('telephone', 86, 'household electrical devices'),
    Class('television', 87, 'household electrical devices'),

    Class('bed', 5, 'household furniture'),
    Class('chair', 20, 'household furniture'),
    Class('couch', 26, 'household furniture'),
    Class('table', 84, 'household furniture'),
    Class('wardrobe', 94, 'household furniture'),

    Class('bee', 6, 'insects'),
    Class('beetle', 7, 'insects'),
    Class('butterfly', 14, 'insects'),
    Class('caterpillar', 18, 'insects'),
    Class('cockroach', 24, 'insects'),

    Class('bear', 3, 'large carnivores'),
    Class('leopard', 42, 'large carnivores'),
    Class('lion', 43, 'large carnivores'),
    Class('tiger', 88, 'large carnivores'),
    Class('wolf', 97, 'large carnivores'),

    Class('bridge', 12, 'large man-made outdoor things'),
    Class('castle', 17, 'large man-made outdoor things'),
    Class('house', 38, 'large man-made outdoor things'),
    Class('road', 68, 'large man-made outdoor things'),
    Class('skyscraper', 76, 'large man-made outdoor things'),

    Class('cloud', 23, 'large natural outdoor scenes'),
    Class('forest', 34, 'large natural outdoor scenes'),
    Class('mountain', 49, 'large natural outdoor scenes'),
    Class('plain', 60, 'large natural outdoor scenes'),
    Class('sea', 71, 'large natural outdoor scenes'),

    Class('camel', 15, 'large omnivores and herbivores'),
    Class('cattle', 19, 'large omnivores and herbivores'),
    Class('chimpanzee', 21, 'large omnivores and herbivores'),
    Class('elephant', 32, 'large omnivores and herbivores'),
    Class('kangaroo', 39, 'large omnivores and herbivores'),

    Class('fox', 35, 'medium-sized mammals'),
    Class('porcupine', 63, 'medium-sized mammals'),
    Class('possum', 64, 'medium-sized mammals'),
    Class('raccoon', 66, 'medium-sized mammals'),
    Class('skunk', 75, 'medium-sized mammals'),

    Class('crab', 27, 'non-insect invertebrates'),
    Class('lobster', 45, 'non-insect invertebrates'),
    Class('snail', 77, 'non-insect invertebrates'),
    Class('spider', 79, 'non-insect invertebrates'),
    Class('worm', 99, 'non-insect invertebrates'),

    Class('baby', 2, 'people'),
    Class('boy', 11, 'people'),
    Class('girl', 36, 'people'),
    Class('man', 46, 'people'),
    Class('woman', 98, 'people'),

    Class('crocodile', 28, 'reptiles'),
    Class('dinosaur', 30, 'reptiles'),
    Class('lizard', 44, 'reptiles'),
    Class('snake', 78, 'reptiles'),
    Class('turtle', 93, 'reptiles'),

    Class('hamster', 37, 'small mammals'),
    Class('mouse', 50, 'small mammals'),
    Class('rabbit', 65, 'small mammals'),
    Class('shrew', 74, 'small mammals'),
    Class('squirrel', 80, 'small mammals'),

    Class('maple', 47, 'trees'),
    Class('oak', 52, 'trees'),
    Class('palm', 56, 'trees'),
    Class('pine', 59, 'trees'),
    Class('willow', 96, 'trees'),

    Class('bicycle', 8, 'vehicles 1'),
    Class('bus', 13, 'vehicles 1'),
    Class('motorcycle', 48, 'vehicles 1'),
    Class('pickup truck', 58, 'vehicles 1'),
    Class('train', 90, 'vehicles 1'),

    Class('lawn-mower', 41, 'vehicles 2'),
    Class('rocket', 69, 'vehicles 2'),
    Class('streetcar', 81, 'vehicles 2'),
    Class('tank', 85, 'vehicles 2'),
    Class('tractor', 89, 'vehicles 2'),
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
cifar100_groups: list[list[Class]] = to_list_of_lists(cifar100_classes)

cifar10_groups: list[list[int]] = [[cifar10_labels[e] for e in g] for g in groups]