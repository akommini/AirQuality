# Import python modules
import numpy as np
import time
import kaggle
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Read in train and test data
def read_data_air_foil():
	print('Reading air foil dataset ...')
	train_data = np.load('../../Data/AirFoil/train.npy')
	train_x = train_data[:,0:train_data.shape[1]-1]
	train_y = train_data[:,train_data.shape[1]-1]
	test_data = np.load('../../Data/AirFoil/test_distribute.npy')
	test_x = test_data

	return (train_x, train_y, test_x)

def read_data_air_quality():
	print('Reading air quality dataset ...')
	train_data = np.load('../../Data/AirQuality/train.npy')
	train_x = train_data[:,0:train_data.shape[1]-2]
	train_y = train_data[:,train_data.shape[1]-2:train_data.shape[1]]
	test_data = np.load('../../Data/AirQuality/test_distribute.npy')
	test_x = test_data
	
	return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

def DecisionTree_reg(train_x, train_y, test_x,tree_depth):
    # Question 1 decision tree
    # Creating kfold crossvalidation indices using KFold
    # 5 different nearest neighbors regressors using the following number of {3, 5, 10, 20, 25}
    #tree_depth=[3, 6, 9, 12, 15, 20, 25, 30, 35, 40]
    tim_arr=[]
    # mean error to model select
    mean_err=np.ones((len(tree_depth),1))
    for i in range(len(tree_depth)):
        dec_Rig =  DecisionTreeRegressor(max_depth=tree_depth[i])    
        # intializing error array to calculate the average
        error=[];
        t = time.process_time();
        for train_index, test_index in kf.split(train_x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            ytest_pre = dec_Rig.fit(X_train, y_train).predict(X_test)
            error.append(compute_error(ytest_pre, y_test));
        end = time.process_time()-t;
        mean_err[i]=np.abs(error).mean()
        tim_arr.append(end)
    index_min = np.argmin(mean_err)
    
    print('Decision tree depth with minimum error for 5-fold cross validation is:',tree_depth[index_min])
    # Training with the total data set with optimal no of neighbors
    dec_Rig =  DecisionTreeRegressor(max_depth=tree_depth[i])
    ytrain_full = dec_Rig.fit(train_x, train_y).predict(train_x)
    err_trainfull_knn=compute_error(ytrain_full, train_y)
    y_predict=dec_Rig.predict(test_x)
    return (err_trainfull_knn, y_predict,mean_err,index_min,tim_arr)


def knn_regression(train_x, train_y, test_x):
    # Question 3 Nearest Neighbors analysis
    # 5 different nearest neighbors regressors using the following number of {3, 5, 10, 20, 25}
    no_neighbors=[3, 5, 10, 20, 25]
    # mean error to model select
    mean_err=np.ones((len(no_neighbors),1))
    for i in range(len(no_neighbors)):
        # knn defined using scikit learn
        knn = neighbors.KNeighborsRegressor(no_neighbors[i], weights='uniform')    
        # intializing error array to calculate the average
        error=[];
        # cross validation to train data
        for train_index, test_index in kf.split(train_x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            # prediction for the test data
            ytest_pre = knn.fit(X_train, y_train).predict(X_test)
            error.append(compute_error(ytest_pre, y_test));
        #mean eror for all the cross validation sets
        mean_err[i]=np.abs(error).mean()
    # minimum mean error to select the best model
    index_min = np.argmin(mean_err)
    print('Number of neighbors with minimum error for 5-fold cross validation is:',no_neighbors[index_min])
    # Training with the total data set with optimal no of neighbors
    knn = neighbors.KNeighborsRegressor(no_neighbors[index_min], weights='uniform')
    ytrain_full = knn.fit(train_x, train_y).predict(train_x)
    err_trainfull_knn=compute_error(ytrain_full, train_y)
    # Prediction for the final test data
    y_predict=knn.predict(test_x)
    return (err_trainfull_knn, y_predict,mean_err,index_min)

def lasso_regression(train_x, train_y, test_x, alpha_arr):
    # Question 4 Lasso regression
    #alpha_arr=[10**-6,10**-4,10**-2,1,10]
    # mean error to model select
    mean_err=np.ones((len(alpha_arr),1))
    for i in range(len(alpha_arr)):
         # Lasso regression defined using scikit learn
        clf = linear_model.Lasso(alpha=alpha_arr[i])   
        # intializing error array to calculate the average
        error=[];
        # cross validation to train data
        for train_index, test_index in kf.split(train_x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            # prediction for the test data
            ytest_pre = clf.fit(X_train, y_train).predict(X_test)
            error.append(compute_error(ytest_pre, y_test));
         #mean eror for all the cross validation sets
        mean_err[i]=np.abs(error).mean()
    # minimum mean error to select the best model   
    index_min = np.argmin(mean_err)
    print('Alpha for Lasso with minimum error for 5-fold cross validation is:',alpha_arr[index_min])
    # Training with the total data set with optimal alpha
    clf = linear_model.Lasso(alpha=alpha_arr[i])
    ytrain_full = clf.fit(train_x, train_y).predict(train_x)
    err_trainfull_knn=compute_error(ytrain_full, train_y)
    # Prediction for the final test data
    y_predict=clf.predict(test_x)
    return (err_trainfull_knn, y_predict,mean_err,index_min)


def ridge_regression(train_x, train_y, test_x, alpha_arr):
    # Question 4 Ridge regression
    #alpha_arr=[10**-6,10**-4,10**-2,1,10]
    # mean error to model select
    mean_err=np.ones((len(alpha_arr),1))
    for i in range(len(alpha_arr)):
        # Lasso regression defined using scikit learn
        clf = Ridge(alpha=alpha_arr[i])   
        # intializing error array to calculate the average
        error=[];
        # cross validation to train data
        for train_index, test_index in kf.split(train_x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            ytest_pre = clf.fit(X_train, y_train).predict(X_test)
            # prediction for the test data
            error.append(compute_error(ytest_pre, y_test));
            #mean eror for all the cross validation sets
        mean_err[i]=np.abs(error).mean()
     # minimum mean error to select the best model     
    index_min = np.argmin(mean_err)
    print('Alpha for Ridge with minimum error for 5-fold cross validation is:',alpha_arr[index_min])
    # Training with the total data set with optimal alpha
    clf = Ridge(alpha=alpha_arr[i])
    ytrain_full = clf.fit(train_x, train_y).predict(train_x)
    err_trainfull_knn=compute_error(ytrain_full, train_y)
    # Prediction for the final test data
    y_predict=clf.predict(test_x)
    return (err_trainfull_knn, y_predict,mean_err,index_min)
###############################################################################
###############################################################################
# Creating kfold crossvalidation indices using KFold
kf = KFold(n_splits=5);

# Reading airfoil data
train_x, train_y, test_x = read_data_air_foil()
print('Train=', train_x.shape)
print('Test=', test_x.shape)
# Maximum tree depths used for decision tree
tree_depth=[3, 6, 9, 12, 15]
# Regularization constant used for linear models
alpha_arr=[10**-6,10**-4,10**-2,1,10]
print('Training depths used for airfoil data=', tree_depth)
# Decision tree for airfoil
err_trainfull_decR_airfoil, y_predict_decR_airfoil,mean_err_decR_airfoil,index_min_decR_airfoil,tim_arr_airfoil=DecisionTree_reg(train_x, train_y, test_x, tree_depth);
# knn regression 
err_trainfull_knn_airfoil, y_predict_knn_airfoil,mean_err_airfoil,index_min_airfoil=knn_regression(train_x, train_y, test_x);
# Lasso and Ridge regression
print('Regularization constants used for airfoil data=', alpha_arr)
err_trainfull_lasso_airfoil, y_predict_lasso_airfoil,mean_err_lasso_airfoil,index_min_lasso_airfoil=lasso_regression(train_x, train_y, test_x,alpha_arr);
err_trainfull_ridge_airfoil, y_predict_ridge_airfoil,mean_err_ridge_airfoil,index_min_ridge_airfoil=ridge_regression(train_x, train_y, test_x,alpha_arr);

# Reading airquality data
train_x, train_y, test_x = read_data_air_quality()
print('Train=', train_x.shape)
print('Test=', test_x.shape)
# Maximum tree depths used for decision tree
tree_depth=[20, 25, 30, 35, 40]
# Regularization constant used for linear models
alpha_arr=[10**-4,10**-2,1,10]
print('Training depths used for airquality data=', tree_depth)
# Decision tree for airquality
err_trainfull_decR_airquality, y_predict_decR_airquality,mean_err_decR_airquality,index_min_decR_airquality,tim_arr_airquality=DecisionTree_reg(train_x, train_y, test_x,tree_depth);
# knn regression 
err_trainfull_knn_airquality, y_predict_knn_airquality,mean_err_airquality,index_min_airquality=knn_regression(train_x, train_y, test_x);
# Lasso and Ridge regression
print('Regularization constants used for airquality data=', alpha_arr)
err_trainfull_lasso_airquality, y_predict_lasso_airquality,mean_err_lasso_airquality,index_min_lasso_airquality=lasso_regression(train_x, train_y, test_x, alpha_arr);
err_trainfull_ridge_airquality, y_predict_ridge_airquality,mean_err_ridge_airquality,index_min_ridge_airquality=ridge_regression(train_x, train_y, test_x, alpha_arr);

##############################################################################
# Model selction
# k-fold used for best model selection
kf = KFold(n_splits=10);
#tree depth used for decison trees
# Reading airfoil data
train_x, train_y, test_x = read_data_air_foil()
tree_depth_airfoil=[25]
# Decision tree for airfoil for best model parameters
err_trainfull_best_airfoil, y_predict_best_airfoil,mean_err_best_airfoil,index_min_best_airfoil,tim_arr_airfoil_best=DecisionTree_reg(train_x, train_y, test_x, tree_depth_airfoil);
# Reading airquality data
train_x, train_y, test_x = read_data_air_quality()
tree_depth_airquality=[20]
# Decision tree for airquality for best model parameters
err_trainfull_best_airquality, y_predict_best_airquality,mean_err_best_airquality,index_min_best_airquality,tim_arr_airquality_best=DecisionTree_reg(train_x, train_y, test_x,tree_depth_airquality);
# Create dummy test output values
#predicted_y = np.ones(test_x.shape[0]) * -1
# Output file location
file_name_Foil = '../Predictions/AirFoil/best.csv'
file_name_Quality = '../Predictions/AirQuality/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name_Foil)
kaggle.kaggleize(y_predict_best_airfoil, file_name_Foil)
print('Writing output to ', file_name_Quality)
kaggle.kaggleize(y_predict_best_airquality, file_name_Quality)

#tree_depth=[3, 6, 9, 12, 15, 20, 25, 30, 35, 40]
#prod_airfoil = [i * 1000 for i in tim_arr_airfoil[0:4]]
#prod_airquality = [i * 1000 for i in tim_arr_airquality[5:9]]
#plt.plot(tree_depth[0:4],prod_airfoil)
#plt.title('AirFoil Data')
#plt.ylabel('Time in millisecs')
#plt.xlabel('maximum decision tree depth')
#plt.show()
#plt.title('AirQuality Data')
#plt.plot(tree_depth[5:9],prod_airquality)G
#plt.ylabel('Time in millisecs')
#plt.xlabel('maximum decision tree depth')
#plt.show()