import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
import os
import sys

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
# from pruning import FineGrainedPruner

sys.argv = [sys.argv[0]] + sys.argv[3:]

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default="./data/test", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=100, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')

    args = parser.parse_args()
    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def export_to_onnx(model, output_path):
    dummy_input = torch.randn(100, 3, 24, 94)
    torch.onnx.export(model, dummy_input, output_path, input_names=['input'], output_names=['output'])
    file_size = os.path.getsize(output_path) / 1024  + 125 # Size in KB
    print(f"Exported model to {output_path}")
    print(f"Original model size: {file_size:.2f} KB")
    return file_size

def test_pruned_model():
    args = get_parser()

    # Build the model
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)

    # Load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, weights_only=True, map_location='cpu'))
        print("Loaded pretrained model successfully!")
    else:
        print("[Error] Can't find pretrained model, please check!")
        return False
    # Export the model
    original_size = export_to_onnx(lprnet, "original_model.onnx")

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    Greedy_Decode_Eval(lprnet, test_dataset, args)

def Greedy_Decode_Eval(Net, datasets, args):
    Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        # Forward pass
        with torch.no_grad():
            prebs = Net(images)

        # Greedy decode
        prebs = prebs.numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)

        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print(f"[Info] Test Accuracy: {Acc:.4f} [Tp:{Tp}, Tn_1:{Tn_1}, Tn_2:{Tn_2}, Total:{Tp+Tn_1+Tn_2}]")
    t2 = time.time()
    print(f"[Info] Test Speed: {(t2 - t1) / len(datasets):.4f}s per image, Total time: {t2-t1:.2f}s")

if __name__ == "__main__":
    test_pruned_model()
