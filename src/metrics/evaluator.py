import torch, numpy as np
class Evaluator:
    @staticmethod
    def pcc(pred, target):
        p, t = pred.flatten().detach().cpu().numpy(), target.flatten().detach().cpu().numpy()
        return float(np.corrcoef(p, t)[0, 1])
    @staticmethod
    def evaluate_all(preds: list, targets: list) -> dict:
        pccs = [Evaluator.pcc(p, t) for p, t in zip(preds, targets)]
        return {'pcc': pccs, 'pcc_mean': np.mean(pccs)}
