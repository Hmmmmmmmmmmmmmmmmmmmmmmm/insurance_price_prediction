import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class StandardScalerSelf:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self,X,y=None):
        X = np.array(X)
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        self.std[self.std==0]=1
        return self
    def transform(self, X):
        X = np.array(X)
        return (X-self.mean)/self.std
    def fit_transform(self, X,y=None):
        self.fit(X)
        return self.transform(X)
class Linear_Regression_Self:
    def __init__(self,
        learning_rate=0.01,
        epochs=10000,
        batch_size=None,
        early_stopping=True,
        patience=10
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.w_=None
        self.b_=0

        self.losses = []

        # self.mean = None
        # self.std = None
        # self.scaler = StandardScaler()

    def _mse(self,y,y_pred):
        return np.mean((y-y_pred)**2)

    def r2_score(self,y,y_pred):
        ss_res = np.sum((y-y_pred)**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        return 1-(ss_res/ss_tot)

    def fit(self, X, y):
        return self._fit_internal(X, y)

    def _fit_internal(self, X, y, auto_plot=False, verbose=True):
        X = np.array(X)
        y=np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        # Store original X for visualization before scaling
        self.X_raw = X if X.ndim > 1 else X.reshape(-1, 1)
        self.y_raw = y
        # X = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        if len(y) != n_samples:
            raise ValueError("X and y must have the same number of samples")
        self.w_=np.zeros(n_features)
        self.b_=0

        batch_size = self.batch_size or n_samples
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                y_pred = (X_batch @ self.w_)+ self.b_
                dw = (2/len(y_batch)) * (X_batch.T @ (y_pred-y_batch))
                db = (2/len(y_batch)) * (np.sum(y_pred-y_batch))

                self.w_ -= self.learning_rate*dw
                self.b_ -= self.learning_rate*db
            full_pred = X@self.w_ + self.b_
            loss = self._mse(y,full_pred)
            self.losses.append(loss)
            if self.early_stopping:
                if loss<best_loss:
                    best_loss = loss
                    patience_counter=0
                else:
                    patience_counter+=1
                if patience_counter >= self.patience:
                    print(f'Early Stopping at epoch {epoch}')
                    break
        if verbose: print("Training Completed")
        # NEW: Automatically run the visualizations at the end!
        if auto_plot:
            self.plot_all_diagnostics(self.X_raw, self.y_raw)


    def fit_normal(self,X,y):
        X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        # X = self.scaler.fit_transform(X)
        X_bias = np.c_[np.ones((X.shape[0],1)), X] # Adding a column of 1s to X to handle the bias (intercept) automatically
        # The Normal Equation: theta = (X^T * X)^-1 * X^T * y
        # Using pinv (pseudo-inverse) is safer than inv for singular matrices
        theta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

        self.b_ = theta[0]
        self.w_ = theta[1:]

    def predict(self,X):
        if self.w_ is None:
            raise ValueError("Model has not been trained yet")
        X=np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, self.w_.shape[0])
        # X = self.scaler.transform(X)
        return X@self.w_+self.b_


    def plot_all_diagnostics(self, X, y):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Plot 1: Learning Curve
        axes[0].plot(self.losses, color='navy')
        axes[0].set_title('Learning Curve (Loss)')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('MSE')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Actual vs Predicted
        y_pred = self.predict(X)
        axes[1].scatter(y, y_pred, alpha=0.5, color='teal')
        min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1].set_title(f'Actual vs Pred (R2: {self.r2_score(y, y_pred):.2f})')
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')

        # Plot 3: Residuals
        residuals = y - y_pred
        axes[2].scatter(y_pred, residuals, alpha=0.5, color='purple')
        axes[2].axhline(0, color='red', linestyle='--')
        axes[2].set_title('Residuals (Errors)')
        axes[2].set_xlabel('Predicted')

        plt.tight_layout()
        plt.show()
