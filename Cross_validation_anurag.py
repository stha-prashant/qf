import numpy as np

x = np.array([0.23, 0.88, 0.21, 0.92, 0.49, 0.62, 0.77, 0.52, 0.3, 0.19])
y = np.array([0.19, 0.96, 0.33, 0.8, 0.46, 0.45, 0.67, 0.32, 0.38, 0.37])

def kfold_cv(x, y, p, K):

    sum_error = 0

    step_size = int(x.size / K)
    # Do k-fold CV
    for k in range(0, x.size, step_size):
    # Calculate the Z matrix for the polynomial order p .
        Z = []
        Z_test = []
        for i in range(1, p + 1): # Loop for each degree of x i.e. x, x^2, x^3,...., x^p
            row = []
            row_test = []
            for j in range(0, x.size):
                if j < k or j >= k + step_size:
                    row.append(x[j]**i)
                else:
                    row_test.append(x[j]**i)
            Z.append(row)
            Z_test.append(row_test)

        Z_np = np.array(Z, dtype = np.float32)
        Z_np_test = np.array(Z_test, dtype = np.float32)
        # print(Z)
        Z_np_T = Z_np.T # Find the transpose of Z
        Z_mul_Z_T = Z_np @ Z_np_T

        """y_test = []
        # The error is here. You don't take all of y. Only the ones used for training.
        for j in range(0, y.size):
            if j >= k and j < k + step_size:
                y_test.append(y[j])
        
        print(len(y_test))"""

        y_train = [] # The output data for points used for training
        y_test = [] # The output data for points used for testing
        for j in range(0, x.size):
                if j < k or j >= k + step_size:
                    y_train.append(y[j])
                else:
                    y_test.append(y[j])

        """print(Z_np)
        print(np.array(y_train, dtype = np.float32).shape)"""
        Z_mul_y = Z_np @ np.array(y_train, dtype = np.float32)  

        # Solve Z @ Z_T @ w = Z @ y  to find w.
        w = np.linalg.solve(Z_mul_Z_T, Z_mul_y)

        LS_values = np.expand_dims(w, axis=0) @ Z_np_test # least square values for test data

        #print(LS_values, y_test)

        residuals = LS_values - y_test # residuals for test data
        residuals_squares = np.square(residuals)
        sum_residuals = np.sum(residuals_squares)

        sum_error = sum_error + sum_residuals
    
    return sum_error / K 

print(kfold_cv(x, y, 2, 5))
print(kfold_cv(x, y, 2, x.size)) # having errors with this....