import matplotlib.pyplot as plt
import sklearn.metrics


def evaluate(y_true, y_pred, title=None,
             classification_report=True, avg_scores=True, average='macro'):

    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

    if title is not None:
        plt.title(title)

    plt.show()

    if classification_report:
        print(sklearn.metrics.classification_report(y_true, y_pred))

    if avg_scores:
        prec = sklearn.metrics.precision_score(y_true, y_pred, average=average)
        rec = sklearn.metrics.recall_score(y_true, y_pred, average=average)
        f1 = sklearn.metrics.f1_score(y_true, y_pred, average=average)
        print(f'AVG PRECISION: {round(prec, 3)} ({average})')
        print(f'AVG RECALL: {round(rec, 3)} ({average})')
        print(f'AVG F1: {round(f1, 3)} ({average})')
