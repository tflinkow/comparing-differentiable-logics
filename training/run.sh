# # StrongClassificationRobustness on MNIST - to run verification, copy resulting .csv and .onnx from the results folder into the verification folder after training
python main.py --data-set=mnist --batch-size=2048 --lr=1e-3 --epochs=50 --save-onnx --constraint="StrongClassificationRobustness(eps=0.4, delta=3.0)" --oracle-steps=40 --oracle-restarts=70
python main.py --data-set=mnist --batch-size=2048 --lr=1e-3 --epochs=50 --save-onnx --constraint="StrongClassificationRobustness(eps=0.4, delta=3.0)" --oracle-steps=40 --oracle-restarts=70 --logic=DL2
python main.py --data-set=mnist --batch-size=2048 --lr=1e-3 --epochs=50 --save-onnx --constraint="StrongClassificationRobustness(eps=0.4, delta=3.0)" --oracle-steps=40 --oracle-restarts=70 --logic=GD

# # StandardRobustness on MNIST (compares LEQ)
python main.py --data-set=mnist --batch-size=2048 --lr=1e-3 --epochs=50 --constraint="StandardRobustness(eps=0.4, delta=0.1)" --oracle-steps=20 --oracle-restarts=30
python main.py --data-set=mnist --batch-size=2048 --lr=1e-3 --epochs=50 --constraint="StandardRobustness(eps=0.4, delta=0.1)" --oracle-steps=20 --oracle-restarts=30 --logic=DL2 --initial-dl-weight=1.5
python main.py --data-set=mnist --batch-size=2048 --lr=1e-3 --epochs=50 --constraint="StandardRobustness(eps=0.4, delta=0.1)" --oracle-steps=20 --oracle-restarts=30 --logic=GD --initial-dl-weight=1.5

# # StandardRobustness on Fashion-MNIST (compares LEQ)
python main.py --data-set=fmnist --batch-size=4096 --lr=1e-3 --epochs=50 --constraint="StandardRobustness(eps=0.2, delta=0.1)" --oracle-steps=30 --oracle-restarts=10
python main.py --data-set=fmnist --batch-size=4096 --lr=1e-3 --epochs=50 --constraint="StandardRobustness(eps=0.2, delta=0.1)" --oracle-steps=30 --oracle-restarts=10 --logic=DL2 --initial-dl-weight=1.3
python main.py --data-set=fmnist --batch-size=4096 --lr=1e-3 --epochs=50 --constraint="StandardRobustness(eps=0.2, delta=0.1)" --oracle-steps=30 --oracle-restarts=10 --logic=GD --initial-dl-weight=1.3

# # Groups on GTSRB (compares AND and OR)
python main.py --data-set=gtsrb --batch-size=4096 --lr=1e-3 --epochs=50 --constraint="Groups(eps=16/255, delta=0.02)" --oracle-steps=30 --oracle-restarts=10
python main.py --data-set=gtsrb --batch-size=4096 --lr=1e-3 --epochs=50 --constraint="Groups(eps=16/255, delta=0.02)" --oracle-steps=30 --oracle-restarts=10 --logic=DL2
python main.py --data-set=gtsrb --batch-size=4096 --lr=1e-3 --epochs=50 --constraint="Groups(eps=16/255, delta=0.02)" --oracle-steps=30 --oracle-restarts=10 --logic=GD
python main.py --data-set=gtsrb --batch-size=4096 --lr=1e-3 --epochs=50 --constraint="Groups(eps=16/255, delta=0.02)" --oracle-steps=30 --oracle-restarts=10 --logic=LK
python main.py --data-set=gtsrb --batch-size=4096 --lr=1e-3 --epochs=50 --constraint="Groups(eps=16/255, delta=0.02)" --oracle-steps=30 --oracle-restarts=10 --logic=RC
python main.py --data-set=gtsrb --batch-size=4096 --lr=1e-3 --epochs=50 --constraint="Groups(eps=16/255, delta=0.02)" --oracle-steps=30 --oracle-restarts=10 --logic=YG

# EvenOdd on MNIST (compares AND and IMPL)
python main.py --data-set=mnist --batch-size=4096 --lr=1e-5 --epochs=50 --constraint="EvenOdd(eps=0.4, delta=0.6, gamma=3.0)" --oracle-steps=20 --oracle-restarts=10
python main.py --data-set=mnist --batch-size=4096 --lr=1e-5 --epochs=50 --constraint="EvenOdd(eps=0.4, delta=0.6, gamma=3.0)" --oracle-steps=20 --oracle-restarts=10 --delay=10 --logic=DL2
python main.py --data-set=mnist --batch-size=4096 --lr=1e-5 --epochs=50 --constraint="EvenOdd(eps=0.4, delta=0.6, gamma=3.0)" --oracle-steps=20 --oracle-restarts=10 --delay=10 --logic=GD
python main.py --data-set=mnist --batch-size=4096 --lr=1e-5 --epochs=50 --constraint="EvenOdd(eps=0.4, delta=0.6, gamma=3.0)" --oracle-steps=20 --oracle-restarts=10 --delay=10 --logic=KD
python main.py --data-set=mnist --batch-size=4096 --lr=1e-5 --epochs=50 --constraint="EvenOdd(eps=0.4, delta=0.6, gamma=3.0)" --oracle-steps=20 --oracle-restarts=10 --delay=10 --logic=LK
python main.py --data-set=mnist --batch-size=4096 --lr=1e-5 --epochs=50 --constraint="EvenOdd(eps=0.4, delta=0.6, gamma=3.0)" --oracle-steps=20 --oracle-restarts=10 --delay=10 --logic=GG
python main.py --data-set=mnist --batch-size=4096 --lr=1e-5 --epochs=50 --constraint="EvenOdd(eps=0.4, delta=0.6, gamma=3.0)" --oracle-steps=20 --oracle-restarts=10 --delay=10 --logic=RC
python main.py --data-set=mnist --batch-size=4096 --lr=1e-5 --epochs=50 --constraint="EvenOdd(eps=0.4, delta=0.6, gamma=3.0)" --oracle-steps=20 --oracle-restarts=10 --delay=10 --logic=RCS
python main.py --data-set=mnist --batch-size=4096 --lr=1e-5 --epochs=50 --constraint="EvenOdd(eps=0.4, delta=0.6, gamma=3.0)" --oracle-steps=20 --oracle-restarts=10 --delay=10 --logic=YG

# ClassSimilarity on CIFAR10 (compares AND and IMPL)
python main.py --data-set=cifar10 --batch-size=1024 --lr=1e-4 --epochs=50 --constraint="ClassSimilarity(eps=24/255, delta=0.1)" --oracle-steps=30 --oracle-restarts=3
python main.py --data-set=cifar10 --batch-size=1024 --lr=1e-4 --epochs=50 --constraint="ClassSimilarity(eps=24/255, delta=0.1)" --oracle-steps=30 --oracle-restarts=3 --delay=10 --logic=DL2
python main.py --data-set=cifar10 --batch-size=1024 --lr=1e-4 --epochs=50 --constraint="ClassSimilarity(eps=24/255, delta=0.1)" --oracle-steps=30 --oracle-restarts=3 --delay=10 --logic=GD
python main.py --data-set=cifar10 --batch-size=1024 --lr=1e-4 --epochs=50 --constraint="ClassSimilarity(eps=24/255, delta=0.1)" --oracle-steps=30 --oracle-restarts=3 --delay=10 --logic=KD
python main.py --data-set=cifar10 --batch-size=1024 --lr=1e-4 --epochs=50 --constraint="ClassSimilarity(eps=24/255, delta=0.1)" --oracle-steps=30 --oracle-restarts=3 --delay=10 --logic=LK
python main.py --data-set=cifar10 --batch-size=1024 --lr=1e-4 --epochs=50 --constraint="ClassSimilarity(eps=24/255, delta=0.1)" --oracle-steps=30 --oracle-restarts=3 --delay=10 --logic=GG
python main.py --data-set=cifar10 --batch-size=1024 --lr=1e-4 --epochs=50 --constraint="ClassSimilarity(eps=24/255, delta=0.1)" --oracle-steps=30 --oracle-restarts=3 --delay=10 --logic=RC
python main.py --data-set=cifar10 --batch-size=1024 --lr=1e-4 --epochs=50 --constraint="ClassSimilarity(eps=24/255, delta=0.1)" --oracle-steps=30 --oracle-restarts=3 --delay=10 --logic=RCS
python main.py --data-set=cifar10 --batch-size=1024 --lr=1e-4 --epochs=50 --constraint="ClassSimilarity(eps=24/255, delta=0.1)" --oracle-steps=30 --oracle-restarts=3 --delay=10 --logic=YG