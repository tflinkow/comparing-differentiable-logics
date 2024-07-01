# Robustness on MNIST (compares only LEQ)
python main.py --data-set=mnist --batch-size=256 --lr=0.0001 --epochs=50 --pgd-steps=15 --constraint="Robustness(eps=0.8, delta=0.01)"
python main.py --data-set=mnist --batch-size=256 --lr=0.0001 --epochs=50 --pgd-steps=15 --constraint="Robustness(eps=0.8, delta=0.01)" --logic=DL2
python main.py --data-set=mnist --batch-size=256 --lr=0.0001 --epochs=50 --pgd-steps=15 --constraint="Robustness(eps=0.8, delta=0.01)" --logic=GD

# Robustness on GTSRB (compares only LEQ)
python main.py --data-set=gtsrb --batch-size=256 --lr=0.0001 --epochs=50 --pgd-steps=40 --constraint="Robustness(eps=0.4, delta=0.01)"
python main.py --data-set=gtsrb --batch-size=256 --lr=0.0001 --epochs=50 --pgd-steps=40 --constraint="Robustness(eps=0.4, delta=0.01)" --logic=DL2
python main.py --data-set=gtsrb --batch-size=256 --lr=0.0001 --epochs=50 --pgd-steps=40 --constraint="Robustness(eps=0.4, delta=0.01)" --logic=GD

# Group on GTSRB (compares AND and OR)
python main.py --data-set=gtsrb --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=30 --constraint="Groups(eps=0.6, delta=0.02)"
python main.py --data-set=gtsrb --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=30 --constraint="Groups(eps=0.6, delta=0.02)" --logic=DL2
python main.py --data-set=gtsrb --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=30 --constraint="Groups(eps=0.6, delta=0.02)" --logic=GD
python main.py --data-set=gtsrb --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=30 --constraint="Groups(eps=0.6, delta=0.02)" --logic=LK
python main.py --data-set=gtsrb --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=30 --constraint="Groups(eps=0.6, delta=0.02)" --logic=RC
python main.py --data-set=gtsrb --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=30 --constraint="Groups(eps=0.6, delta=0.02)" --logic=YG

# Group on CIFAR100 (compares AND and OR)
python main.py --data-set=cifar100 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="Groups(eps=0.2, delta=0.09)"
python main.py --data-set=cifar100 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="Groups(eps=0.2, delta=0.09)" --logic=DL2
python main.py --data-set=cifar100 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="Groups(eps=0.2, delta=0.09)" --logic=GD
python main.py --data-set=cifar100 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="Groups(eps=0.2, delta=0.09)" --logic=LK
python main.py --data-set=cifar100 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="Groups(eps=0.2, delta=0.09)" --logic=RC
python main.py --data-set=cifar100 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="Groups(eps=0.2, delta=0.09)" --logic=YG

# Class-Similarity on CIFAR10 (compares IMPL and AND)
python main.py --data-set=cifar10 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="ClassSimilarity(eps=0.6)"
python main.py --data-set=cifar10 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="ClassSimilarity(eps=0.6)" --logic=DL2
python main.py --data-set=cifar10 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="ClassSimilarity(eps=0.6)" --logic=GD
python main.py --data-set=cifar10 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="ClassSimilarity(eps=0.6)" --logic=KD
python main.py --data-set=cifar10 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="ClassSimilarity(eps=0.6)" --logic=LK
python main.py --data-set=cifar10 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="ClassSimilarity(eps=0.6)" --logic=GG
python main.py --data-set=cifar10 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="ClassSimilarity(eps=0.6)" --logic=RC
python main.py --data-set=cifar10 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="ClassSimilarity(eps=0.6)" --logic=RCS
python main.py --data-set=cifar10 --batch-size=512 --lr=0.00001 --epochs=50 --pgd-steps=50 --constraint="ClassSimilarity(eps=0.6)" --logic=YG