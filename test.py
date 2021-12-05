from main import parse_args, generate_batch, test_eval
from utl import Cell_Net
from glob import glob


def load_dataset_test(dataset_path, n_folds, rand_state):
    return glob(dataset_path + '/*/img*')


if __name__ == '__main__':
    input_dim = (27, 27, 3)
    args = parse_args()

    model = Cell_Net.cell_net(input_dim, args, use_mul_gpu=False)
    model.load_weights("Saved_model/_Batch_size_1epoch_best.hd5")
    print(model.summary())

    test_bags = load_dataset_test(dataset_path="/home/wf/code/data/Patches", n_folds=5, rand_state=1)

    # convert bag to batch
    test_set = generate_batch(test_bags)

    loss, acc = test_eval(model, test_set)
    print("loss=%.3f, acc=%.3f" % (loss, acc))
