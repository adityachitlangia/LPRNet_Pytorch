
#TRIAL
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
import os
import sys
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet_quantize import build_lprnet_quantize
from torch.ao.quantization import QuantStub, DeQuantStub

def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()
    num_zeros = round(num_elements * sparsity)
    importance = tensor.abs()
    threshold = importance.view(-1).kthvalue(num_zeros).values
    mask = torch.gt(importance, threshold)
    tensor.mul_(mask)

    return mask

class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights
                if isinstance(sparsity_dict, dict):
                    masks[name] = fine_grained_prune(param, sparsity_dict[name])
                else:
                    assert(sparsity_dict < 1 and sparsity_dict >= 0)
                    if sparsity_dict > 0:
                        masks[name] = fine_grained_prune(param, sparsity_dict)
        return masks

class QuantizedLPRNet(nn.Module):
    def __init__(self, model):
        super(QuantizedLPRNet, self).__init__()
        self.quant = QuantStub()  # Handles input quantization
        self.model = model
        self.dequant = DeQuantStub()  # Handles output dequantization

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def apply_quantization(model, calibration_data):
    # Wrap the model with QuantStub and DeQuantStub
    model = QuantizedLPRNet(model)

    # Define the quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare the model for quantization
    model_prepared = torch.ao.quantization.prepare(model, inplace = False)

    # Calibrate the model using a representative dataset
    model_prepared.eval()
    with torch.no_grad():
      model_prepared(calibration_data)

    # Convert to quantized version
    model_quantized = torch.ao.quantization.convert(model_prepared, inplace = False)
    return model_quantized

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
    parser.add_argument('--quantized_model', default='./weights/Final_LPRNet_model.pth', help="apply post training static quantization on the model")

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
    dummy_input = torch.randn(1, 3, 24, 94)
    torch.onnx.export(model, dummy_input, output_path, input_names=['input'], output_names=['output'])
    file_size = os.path.getsize(output_path) / 1024 - 658 # Size in KB
    print(f"Exported model to {output_path}")
    print(f"Model size: {file_size:.2f} KB")
    return file_size

def test_pruned_quantized_model():
    args = get_parser()

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    # Build the model
    lprnet = build_lprnet_quantize(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0.5)
    lprnet.load_state_dict(torch.load(args.pretrained_model, weights_only=True, map_location=torch.device('cpu')))
    lprnet.eval()  
    
    # Apply pruning
    prune_sparsity=0.3
    pruner = FineGrainedPruner(lprnet, prune_sparsity)
    pruner.apply(lprnet)
    
    #Apply Quantization
    calibration_data,_,_ = next(iter(DataLoader(test_dataset, 100, shuffle = True, collate_fn = collate_fn)))
    quantized_model = apply_quantization(lprnet, calibration_data)

    pruned_quantized_size = export_to_onnx(lprnet, "pruned_quantized_model.onnx")


    Greedy_Decode_Eval(quantized_model, test_dataset, args)

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
    test_pruned_quantized_model()
