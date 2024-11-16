import sys
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from itertools import product

warnings.filterwarnings(action="ignore")
np.random.seed(100)

def load_data(X, y):
    """손글씨 데이터를 X, y로 읽어온 후 학습 데이터, 테스트 데이터로 나눕니다."""
    X_train = X[:1600]
    Y_train = y[:1600]

    X_test = X[1600:]
    Y_test = y[1600:]

    return X_train, Y_train, X_test, Y_test

def train_MLP_classifier(X, y):
    """MLPClassifier를 정의하고 학습을 시킵니다."""
    clf = MLPClassifier(hidden_layer_sizes=(128,128),solver='adam', beta_1=0.99999)
    clf.fit(X, y)
    return clf

def report_clf_stats(clf, X, y):
    """정확도를 출력하는 함수입니다."""
    score = clf.score(X, y)
    return score

def main():
    """main 함수를 완성합니다."""
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, Y_train, X_test, Y_test = load_data(X, y)

    # Hidden layer의 뉴런 수 조합을 정의합니다.
    # layer_sizes = [128, 256, 512, 1024]
    # for hidden_layer_sizes in product(layer_sizes, repeat=2):
    #     clf = train_MLP_classifier(X_train, Y_train, hidden_layer_sizes)
    #     score = report_clf_stats(clf, X_test, Y_test)

    #     print(f"Testing hidden_layer_sizes={hidden_layer_sizes}: Accuracy={score:.2f}")

    #     if score > 0.95:
    #         print(f"Found hidden_layer_sizes={hidden_layer_sizes} with Accuracy={score:.2f}")
    #         break

    clf = train_MLP_classifier(X_train, Y_train)
    score = report_clf_stats(clf, X_test, Y_test)
    print(score)
    return 0

if __name__ == "__main__":
    sys.exit(main())


# score - 0.9543147208121827
# MLPClassifer 의 adam solver , beta_1:  adam optimizer 로 각 파리머터 마다 다른 학습률 적용, 과거 그래디언트에 대해서 빠른 수렴과 안정성을 확보하고자 해서 쓰는 건데, 과거 그래프언트의 지수이동평균과 제곱 둘다 이용
# 1차 모멘텀과 2차 모멘텀으로 각 파라미터에 대한 학습률과 학습 방향에 대한 효과적인 학습 가능
# beta_1은 수렴 안정성 같은거, 얼마나 내가 이전 그래디언트를 반영할지 인데 현재 데이터가 많이 없으므로 학습에 불안정성이 심해서 0.99999로 설정.