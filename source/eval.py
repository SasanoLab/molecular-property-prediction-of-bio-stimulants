from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def calc_result(golds, preds, probs):
        pres,recs, fs, rocs, prs = [], [], [], [], []

        precision, recall, thresholds = precision_recall_curve(golds, probs)
        prs.append(auc(recall, precision))

        rocs.append(roc_auc_score(golds, probs))

        pre,rec,f,_ = precision_recall_fscore_support(
                golds,
                preds,
                average="binary",
                zero_division=0,
                labels=[0,1],
        )
        pres.append(pre)
        recs.append(rec)
        fs.append(f)

        return round(sum(pres)/len(pres),4), round(sum(recs)/len(recs),4), round(sum(fs)/len(fs),4), round(sum(rocs)/len(rocs),4), round(sum(prs)/len(prs),4)
