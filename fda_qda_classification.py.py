import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.datasets
from sklearn.datasets import load_wine
from sklearn.metrics import roc_auc_score

class FDA:
    def __init__(self):
        pass

    def fit(self, X1, X2):
        d = X1.shape[0]
        m1 = X1.mean(axis=0)
        m2 = X2.mean(axis=0)
        X = np.concatenate([X1, X2])
        Si = np.linalg.inv(np.cov(X.T))
        self.w = Si.dot(m2-m1)
        return self

    def predict(self, X):
        return np.dot(X, self.w)

X,T = sklearn.datasets.load_wine(return_X_y=True)

X = X[:,4:8] # extract compositional features of interest
X = X - X.mean(axis=0)
X = X / X.std(axis=0)
T = 1.0*(T==0) # indicator function of the cultivar of interest
XN = X[T==0] # negative examples
XP = X[T==1]

Xpca = sklearn.decomposition.PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(6,4))
plt.scatter(*Xpca[T==0].T,alpha=0.25,label='neg',s=50)
plt.scatter(*Xpca[T==1].T,alpha=0.25,label='pos',s=50)
plt.legend()
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

# Calculate total variance
def total_variance(X):
    return np.sum(np.var(X, axis=0))

print("Total variance for negative class:", total_variance(XN))
print("Total variance for positive class:", total_variance(XP))

# Class QDA for Quadratic Discriminant Analysis
class QDA:
    def __init__(self):
        pass

    def fit(self, X1, X2):
        self.m1 = np.mean(X1, axis=0)
        self.m2 = np.mean(X2, axis=0)
        self.S1 = np.cov(X1.T)
        self.S2 = np.cov(X2.T)
        self.p1 = len(X1) / (len(X1) + len(X2))
        self.p2 = len(X2) / (len(X1) + len(X2))
        return self

    def predict(self, X):
        def discriminant(x, m, S, p):
            inv_S = np.linalg.inv(S)
            return -0.5 * (x - m).T.dot(inv_S).dot(x - m) - 0.5 * np.log(np.linalg.det(S)) + np.log(p)

        scores1 = [discriminant(x, self.m2, self.S2, self.p1) for x in X]
        scores2 = [discriminant(x, self.m1, self.S1, self.p2) for x in X]

        return   np.array(scores1) - np.array(scores2)

# Apply FDA and QDA
fda = FDA().fit(XN, XP)
qda = QDA().fit(XN, XP)

# Histogram visualization

def hist(ZN,ZP,name):
    plt.figure(figsize=(6, 2))
    plt.hist([ZN,ZP],color=['C0','C1'],bins=25,alpha=0.25,histtype='stepfilled')
    plt.hist([ZN,ZP],color=['C0','C1'],bins=25,alpha=0.75,histtype='step',lw=2.5)
    plt.xlabel('Discriminant score')
    plt.ylabel('Frequency')
    plt.show()
    plt.xlabel(name)

hist(fda.predict(XN),fda.predict(XP),'Fisher')
hist(qda.predict(XN),qda.predict(XP),'QDA') 


# AUROC Calculation
fda_scores = np.hstack([fda.predict(XN), fda.predict(XP)])
qda_scores = np.hstack([qda.predict(XN), qda.predict(XP)])
labels = np.hstack([np.zeros(len(XN)), np.ones(len(XP))])

print("AUROC-Fisher:", roc_auc_score(labels, fda_scores))
print("AUROC-QDA:", roc_auc_score(labels, qda_scores))
