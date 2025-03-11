import argparse
import idx2numpy
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int)
    args = parser.parse_args()

    np.random.seed(42)

    images = idx2numpy.convert_from_file(f'original_dataset/t10k-images.idx3-ubyte')
    images = images / 255.

    labels = idx2numpy.convert_from_file(f'original_dataset/t10k-labels.idx1-ubyte')

    assert len(images) == len(labels)
    idx = np.random.choice(len(images), size=args.size, replace=False)

    print(idx)

    idx2numpy.convert_to_file(f't{args.size}-images.idx', images[idx])
    idx2numpy.convert_to_file(f't{args.size}-labels.idx', labels[idx])

if __name__ == '__main__':
    main()