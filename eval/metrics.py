
def compute_accuracy(pred, ground_truth):
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    
    correct = sum(p == gt for p, gt in zip(pred, ground_truth))
    total = len(ground_truth)
    
    return correct / total if total > 0 else 0.0
    
    
def compute_spcc(pred, ground_truth):
    from scipy.stats import spearmanr
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    coefficient, _ = spearmanr(pred, ground_truth)
    return coefficient


def compute_plcc(pred, ground_truth):
    from scipy.stats import pearsonr
    assert len(pred) == len(ground_truth), "len(pred) should be the same as len(ground_truth)"
    coefficient, _ = pearsonr(pred, ground_truth)
    return coefficient
