import streamlit as st
from PIL import Image
import io
import pandas as pd
import numpy as np #python -m pip install numpy==1.23.1 --Version: 1.24.4 (01.06.2024)
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
import plotly.express as px

# Data Cleaning and Preparation
from category_encoders import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# Model Performance and Spliting
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay,RocCurveDisplay,precision_recall_curve,PrecisionRecallDisplay)
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel, RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# from stqdm import stqdm

# Modeling Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
import shap

import warnings
warnings.filterwarnings('ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)






def main():
    st.title("Mobile Price Prediction")
    st.sidebar.title("Operations")
    image = Image.open(r"mobile.jpg")
    st.image(image)

    # Cached Data  Functons
    @st.cache_data()
    def load_data():
        data = pd.read_csv(r"train.csv")
        return data

    @st.cache_data()
    def scaling_data(df):
        scaled_df = df.copy()
        transformer = MinMaxScaler()

        def scaling(columns):
            return transformer.fit_transform(scaled_df[columns].values.reshape(-1, 1))

        columns_to_be_scaled = set(numerical_variables) - set(['price_range'])
        for i in columns_to_be_scaled:
            scaled_df[i] = scaling(i)

        return scaled_df

    @st.cache_data()
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            predictions = model.predict(X_test)
            cm = confusion_matrix(y_test, predictions, labels=model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot()
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            predictions = model.predict(X_test)
            precision, recall, _ = precision_recall_curve(y_test, predictions)
            disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            disp.plot()
            st.pyplot()

    df = load_data()
    st.sidebar.subheader("Data Analysis")
    
    if st.sidebar.checkbox("Data Info"):
        st.write("**There are ", df.shape[0], "rows and", df.shape[1], "columns in the dataset.**")
        st.subheader("Here is the information of the dataset!")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Descriptive Information of the Dataset")
        describe = df.describe().T
        st.write(describe)

        st.write("Is there any mising values in the dataset? **Give me the answer!**", df.isnull().values.any())
        st.write("Is there duplicated values in the dataset? **Give me the answer!**", df.duplicated().any())

    categorical_variables = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
    numerical_variables = list(set(df.columns.tolist()) - set(categorical_variables))

    if st.sidebar.checkbox("Categorical Feature Analysis"):
        st.subheader("**Data Visualization for Categorical Features**")
        st.write("Categorical Figures are:\n", categorical_variables)

        for i in range(0, len(categorical_variables), 2):
            df[categorical_variables[i]] = df[categorical_variables[i]].astype('object')
            if i + 1 == len(categorical_variables):
                plt.figure(figsize=(8, 5))
                sns.countplot(x=df[categorical_variables[i]], palette='tab20')
                st.write(figure)
            else:
                df[categorical_variables[i + 1]] = df[categorical_variables[i + 1]].astype('object')
                figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
                sns.countplot(ax=axis[0], x=df[categorical_variables[i]], palette='tab20')
                sns.countplot(ax=axis[1], x=df[categorical_variables[i + 1]], palette='tab20')
                st.write(figure)

    if st.sidebar.checkbox("Numerical Feature Analysis"):
        st.subheader("**Data Visualization for Numerical Features**")
        st.write("Numerical Columns:", numerical_variables)
        attributes_to_drop = categorical_variables
        attributes_to_drop.append('price_range')
        if st.checkbox("Correlation Matrix"):
            st.subheader("Correlation Matrix of Numeric Features")
            numeric_df = df.drop(attributes_to_drop, axis=1)
            correlation_matrix = numeric_df.corr(method='spearman')
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(data=correlation_matrix, annot=True, fmt='.2g', ax=ax)
            st.write(fig)
        if st.checkbox("Distribution Plot"):
            for i in numerical_variables:
                dist_fig = plt.figure(figsize=(6, 2))
                sns.distplot(df[i])
                st.write(dist_fig)

    if st.sidebar.checkbox("Analysis of Price_Range"):
        fig_size = (9, 5)
        figure_countplot, axis = plt.subplots(figsize=fig_size)
        sns.countplot(data=df, x='price_range', palette="rocket", ax=axis)
        st.write(figure_countplot)

        label = df['price_range'].value_counts()
        transuction = label.index
        quantity = label.values
        figure = px.pie(df.price_range, values=quantity, names=transuction, hole=.5)
        st.write(figure)

        figure_pie, axis = plt.subplots(figsize=fig_size)
        df['price_range'].value_counts().plot.pie(ax=axis, autopct='%1.1f%%',
                                                  title="Pie Chart for count of price_range",
                                                  legend=False, colormap='Wistia', startangle=0,
                                                  ylabel='')
        st.write(figure_pie)

    scaled_df = scaling_data(df)
    y = scaled_df['price_range'].astype('int')
    X = scaled_df.drop(['price_range'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    class_names = [0, 1, 2, 3]

    st.sidebar.subheader("Modeling Section (Default Parameters)")
    selected_algorithm = st.sidebar.selectbox("Machine Learning Algorithms",
                                              ("LogisticRegression", "RandomForestClassifier", "KNeighborsClassifier"),
                                              index=None, placeholder="Choose Algorithm", key="default_params")

    if selected_algorithm == 'LogisticRegression':
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        y_pred_log_reg = log_reg.predict(X_test)
        st.write('<p style="font-size:24px;"><b>LOGISTIC REGRESSION MODEL RESULTS</b></p>', unsafe_allow_html=True)
        st.write('Accuracy score of testing set:', round(accuracy_score(y_test, y_pred_log_reg), 2))
        st.write('Precision score of testing set:',
                 round(precision_score(y_test, y_pred_log_reg, average="weighted"), 2))
        st.write('Recall score of testing set:', round(recall_score(y_test, y_pred_log_reg, average="weighted"), 2))
        st.write('F1 score of testing set:', round(f1_score(y_test, y_pred_log_reg, average="weighted"), 2))

        cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm_log_reg, annot=True, square=True, fmt='1', cmap='BuPu', ax=ax)
        st.write(fig)

        cm_log_reg = confusion_matrix(y_test, y_pred_log_reg, labels=log_reg.classes_)
        disp_cm_log_reg = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg, display_labels=log_reg.classes_)
        fig, ax = plt.subplots(figsize=(10, 6))
        disp_cm_log_reg.plot(ax=ax)
        st.pyplot()

        feature_importance_abs = abs(log_reg.coef_[0])
        feature_importance = 100.0 * (feature_importance_abs / feature_importance_abs.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        featfig = plt.figure(figsize=(14, 9))
        featax = featfig.add_subplot(1, 1, 1)
        featax.barh(pos, feature_importance[sorted_idx], align='center')
        plt.title('Most 10 Relative Feature Importance for Logistic Regression Model', fontsize=13, fontweight='bold')
        featax.set_yticks(pos)
        featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=12)

        plt.tight_layout()
        st.pyplot()

    if selected_algorithm == 'RandomForestClassifier':
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        st.write("**RANDOM FOREST CLASSIFIER MODEL RESULTS**")
        st.write('Accuracy score of testing set:', round(accuracy_score(y_test, y_pred_rf), 2))
        st.write('Precision score of testing set:', round(precision_score(y_test, y_pred_rf, average="weighted"), 2))
        st.write('Recall score of testing set:', round(recall_score(y_test, y_pred_rf, average="weighted"), 2))
        st.write('F1 score of testing set:', round(f1_score(y_test, y_pred_rf, average="weighted"), 2))

        cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
        disp_cm_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf.classes_)
        fig, ax = plt.subplots(figsize=(10, 6))
        disp_cm_rf.plot(ax=ax)
        st.pyplot()

        importance_df = pd.DataFrame({"Feature_Name": X.columns, "Importance": rf.feature_importances_})
        sorted_importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=sorted_importance_df, x='Importance', y='Feature_Name')
        plt.title('10 Most Important Features - Random Forest Classifier', fontweight='bold', fontsize=15)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Feature Name', fontsize=12)
        st.pyplot()

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test[0:100])
        shap.summary_plot(shap_values, X_test[0:100], plot_type='bar', max_display=10, class_names=rf.classes_)
        st.pyplot()

    if selected_algorithm == 'KNeighborsClassifier':
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        st.write("**KNN CLASSIFIER MODEL RESULTS**")
        st.write('Accuracy score of testing set', round(accuracy_score(y_test, y_pred_knn), 2))
        st.write('Precision score of testing set', round(precision_score(y_test, y_pred_knn, average="weighted"), 2))
        st.write('Recall score of testing set', round(recall_score(y_test, y_pred_knn, average="weighted"), 2))
        st.write('F1 score of testing set', round(f1_score(y_test, y_pred_knn, average="weighted"), 2))

        cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
        disp_cm_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn.classes_)
        fig, ax = plt.subplots(figsize=(10, 6))
        disp_cm_knn.plot(ax=ax)
        st.pyplot()

    if st.sidebar.button("Compare All Default Models ", False):
        models = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier()]

        st.header("Model Comparisons - F1 Score ")
        result_f1 = []
        results_f1 = pd.DataFrame(columns=["Models", "F1"])

        for model in models:
            names = model.__class__.__name__
            y_pred = model.fit(X_train, y_train).predict(X_test)
            f1score = cross_val_score(model, X_test, y_test, cv=3, scoring="f1_weighted", n_jobs=-1).mean()
            result_f1 = pd.DataFrame([[names, f1score * 100]], columns=["Models", "F1"])
            results_f1 = results_f1._append(result_f1)

        sns.barplot(x='F1', y='Models', data=results_f1, palette="coolwarm")
        plt.xlabel('F1 %')
        plt.ylabel("Sharat's Model Names")
        plt.title("Sharat's Models", loc="center")
        st.pyplot()
        st.write(results_f1.sort_values(by="F1", ascending=False))

        st.header("Model Comparisons - Accuracy Score")
        result_accuracy = []
        results_accuracy = pd.DataFrame(columns=["Models", "Accuracy"])

        for model in models:
            names = model.__class__.__name__
            y_pred = model.fit(X_train, y_train).predict(X_test)
            accuracy = cross_val_score(model, X_test, y_test, cv=3, scoring="f1_weighted", n_jobs=-1).mean()
            result_accuracy = pd.DataFrame([[names, accuracy * 100]], columns=["Models", "Accuracy"])
            results_accuracy = results_accuracy._append(result_accuracy)

        sns.barplot(x='Accuracy', y='Models', data=results_accuracy, palette="coolwarm")
        plt.xlabel('Accuracy %')
        plt.ylabel("Sharat's Model Names")
        plt.title("Sharat's Models", loc="center")
        st.pyplot()
        st.write(results_accuracy.sort_values(by="Accuracy", ascending=False))

    st.sidebar.subheader("Hyperparameter Optimization")
    optimize_algorithm = st.sidebar.selectbox("Machine Learning Algorithms",
                                              ("LogisticRegression", "RandomForestClassifier", "KNeighborsClassifier"),
                                              index=None, placeholder="Choose Algorithm", key="hyperparameters")

    if optimize_algorithm == 'KNeighborsClassifier':
        knn_parameters = {"algorithm": ["auto", "ball_tree"], "n_neighbors": [3, 5]}
        knn_model_cv = KNeighborsClassifier()
        knn_grid_cv = GridSearchCV(knn_model_cv, knn_parameters, cv=2, verbose=False, n_jobs=-1).fit(X_train, y_train)
        st.write("Best parameters of KNN Classifier:", str(knn_grid_cv.best_params_))

        knn_model_tuned = knn_grid_cv.best_estimator_
        knn_model_tuned.fit(X_train, y_train)
        y_pred_knn_tuned = knn_model_tuned.predict(X_test)

        st.write("**TUNED KNN CLASSIFIER TRAINING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of training set',
                 round(cross_val_score(knn_model_tuned, X_train, y_train, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of training set',
                 round(cross_val_score(knn_model_tuned, X_train, y_train, cv=3, scoring='precision_weighted').mean(),
                       2))
        st.write('Recall score of training set',
                 round(cross_val_score(knn_model_tuned, X_train, y_train, cv=3, scoring='recall_weighted').mean(), 2))
        st.write('F1 score of training set',
                 round(cross_val_score(knn_model_tuned, X_train, y_train, cv=3, scoring='f1_weighted').mean(), 2))

        st.write("**TUNED KNN CLASSIFIER TESTING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of testing set',
                 round(cross_val_score(knn_model_tuned, X_test, y_test, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of testing set',
                 round(cross_val_score(knn_model_tuned, X_test, y_test, cv=3, scoring='precision_weighted').mean(), 2))
        st.write('Recall score of testing set',
                 round(cross_val_score(knn_model_tuned, X_test, y_test, cv=3, scoring='recall_weighted').mean(), 2))
        st.write('F1 score of testing set',
                 round(cross_val_score(knn_model_tuned, X_test, y_test, cv=3, scoring='f1_weighted').mean(), 2))

        neighbors = np.arange(3, 10, 2)
        train_accuracy = np.empty(len(neighbors))
        test_accuracy = np.empty(len(neighbors))

        for i, k in enumerate(neighbors):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            train_accuracy[i] = knn.score(X_train, y_train)
            test_accuracy[i] = knn.score(X_test, y_test)

        plt.title('k-NN: Varying Number of Neighbors')
        plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
        plt.plot(neighbors, train_accuracy, label='Training Accuracy')
        plt.legend()
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy')
        st.pyplot()

        k_list = list(range(3, 6, 1))
        cv_scores = []

        for k in k_list:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='accuracy')
            cv_scores.append(scores.mean())

        MSE = [1 - x for x in cv_scores]

        plt.figure(figsize=(8, 3))
        plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
        plt.xlabel('Number of Neighbors K', fontsize=15)
        plt.ylabel('Misclassification Error', fontsize=15)
        plt.plot(k_list, MSE);
        st.pyplot()

    if optimize_algorithm == 'RandomForestClassifier':
        rf_parameters = {"n_estimators": [500, 1000], "criterion": ['gini', 'entropy']}
        rf_model_cv = RandomForestClassifier()
        rf_grid_cv = GridSearchCV(rf_model_cv, rf_parameters, cv=2, verbose=False, n_jobs=-1).fit(X_train, y_train)
        st.write("Best parameters of RF Classifier:", str(rf_grid_cv.best_params_))

        rf_model_tuned = rf_grid_cv.best_estimator_
        rf_model_tuned.fit(X_train, y_train)
        y_pred_rf_tuned = rf_model_tuned.predict(X_test)

        st.write("**TUNED RF CLASSIFIER TRAINING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of training set',
                 round(cross_val_score(rf_model_tuned, X_train, y_train, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of training set',
                 round(cross_val_score(rf_model_tuned, X_train, y_train, cv=3, scoring='precision_weighted').mean(), 2))
        st.write('Recall score of training set',
                 round(cross_val_score(rf_model_tuned, X_train, y_train, cv=3, scoring='recall_weighted').mean(), 2))
        st.write('F1 score of training set',
                 round(cross_val_score(rf_model_tuned, X_train, y_train, cv=3, scoring='f1_weighted').mean(), 2))

        st.write("**TUNED RF CLASSIFIER TESTING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of testing set',
                 round(cross_val_score(rf_model_tuned, X_test, y_test, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of testing set',
                 round(cross_val_score(rf_model_tuned, X_test, y_test, cv=3, scoring='precision_weighted').mean(), 2))
        st.write('Recall score of testing set',
                 round(cross_val_score(rf_model_tuned, X_test, y_test, cv=3, scoring='recall_weighted').mean(), 2))
        st.write('F1 score of testing set',
                 round(cross_val_score(rf_model_tuned, X_test, y_test, cv=3, scoring='f1_weighted').mean(), 2))

    if optimize_algorithm == 'LogisticRegression':
        st.write("Optimizat")
        lr_parameters = {"C": [1, 2], "penalty": ['l1', 'l2']}
        lr_model_cv = LogisticRegression()
        lr_grid_cv = GridSearchCV(lr_model_cv, lr_parameters, cv=2, verbose=False, n_jobs=-1).fit(X_train, y_train)
        st.write("Best parameters of RF Classifier:", str(lr_grid_cv.best_params_))

        lr_model_tuned = lr_grid_cv.best_estimator_
        lr_model_tuned.fit(X_train, y_train)
        y_pred_lr_tuned = lr_model_tuned.predict(X_test)

        st.write("**TUNED LR CLASSIFIER TRAINING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of training set',
                 round(cross_val_score(lr_model_tuned, X_train, y_train, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of training set',
                 round(cross_val_score(lr_model_tuned, X_train, y_train, cv=3, scoring='precision_weighted').mean(), 2))
        st.write('Recall score of training set',
                 round(cross_val_score(lr_model_tuned, X_train, y_train, cv=3, scoring='recall_weighted').mean(), 2))
        st.write('F1 score of training set',
                 round(cross_val_score(lr_model_tuned, X_train, y_train, cv=3, scoring='f1_weighted').mean(), 2))

        st.write("**TUNED LR CLASSIFIER TESTING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of testing set',
                 round(cross_val_score(lr_model_tuned, X_test, y_test, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of testing set',
                 round(cross_val_score(lr_model_tuned, X_test, y_test, cv=3, scoring='precision_weighted').mean(), 2))
        st.write('Recall score of testing set',
                 round(cross_val_score(lr_model_tuned, X_test, y_test, cv=3, scoring='recall_weighted').mean(), 2))
        st.write('F1 score of testing set',
                 round(cross_val_score(lr_model_tuned, X_test, y_test, cv=3, scoring='f1_weighted').mean(), 2))

    st.sidebar.subheader("Building Your Own ML Models")
    classifier = st.sidebar.selectbox("Classifier",
                                      ("RandomForestClassifier", "KNeighborsClassifier", "LogisticRegression"),
                                      index=None,
                                      placeholder="Choose Algorithm", key="builtonyourown")

    if classifier == 'RandomForestClassifier':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators = st.sidebar.slider("The number of trees in the forest", 100, 1000, step=100, key="n_estimators")
        max_depth = st.sidebar.slider("The maximum depth of the tree", 5, 50, step=5, key="max_depth")
        min_samples_split = st.sidebar.slider("The minimum samples split of the tree", 5, 20, step=5,
                                              key="min_samples_split")
        criterion_rf = st.sidebar.radio("Criterion", ('gini', 'entropy', 'log_loss'), key='criterion_rf')

        metrics = st.sidebar.multiselect("Which metrics do you want to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader('Random Forest Classifier Results')
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion_rf,
                                           min_samples_split=min_samples_split)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy score for testing set: ", accuracy.round(2))
            st.write('Precision score for testing set: ', precision_score(y_test, y_pred, labels=class_names, average='weighted').round(2))
            st.write('Recall score for testing set: ', recall_score(y_test, y_pred, labels=class_names, average='weighted').round(2))
            st.write('F1 score for testing set: ', f1_score(y_test, y_pred, labels=class_names, average='weighted').round(2))
            plot_metrics(metrics)

    if classifier == 'KNeighborsClassifier':
        st.sidebar.subheader('Model Hyperparameters')
        k = st.sidebar.slider("k (Number of Neighbors Parameter)", 3, 10, step=2, key='k')
        p = st.sidebar.slider("p (Power Parameter)", 1, 2, step=1, key='p')
        algorithm = st.sidebar.radio("Algorithm", ("auto", "ball_tree", "kd_tree", "brute"), key="algorithm")
        weights = st.sidebar.radio("Weight", ('uniform', 'distance'), key='weights')

        metrics = st.sidebar.multiselect("Which metrics do you want to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader('K-Nearest Classifier Results')
            model = KNeighborsClassifier(n_neighbors=k, p=p, algorithm=algorithm, weights=weights)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy score for testing set: ", accuracy.round(2))
            st.write('Precision score for testing set: ', precision_score(y_test, y_pred, labels=class_names, average='weighted').round(2))
            st.write('Recall score for testing set: ', recall_score(y_test, y_pred, labels=class_names, average='weighted').round(2))
            st.write('F1 score for testing set: ', f1_score(y_test, y_pred, labels=class_names, average='weighted').round(2))
            plot_metrics(metrics)

    if classifier == 'LogisticRegression':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input("C (Regularition Parameter)", 0.1, 10.0, step=0.5, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 1000, key='max_iter_LR')
        penalty = st.sidebar.radio("Norm of penalty", ('l1', 'l2', 'elasticnet', None), key='penalty')
        solver = st.sidebar.radio("Algorithm for optimization",
                                  ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'))

        metrics = st.sidebar.multiselect("Which metrics do you want to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader('Logistic Regression Classifier Results')
            model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy score for testing set: ", accuracy.round(2))
            st.write('Precision score for testing set: ', precision_score(y_test, y_pred, labels=class_names, average='weighted').round(2))
            st.write('Recall score for testing set: ', recall_score(y_test, y_pred, labels=class_names, average='weighted').round(2))
            st.write('F1 score for testing set: ', f1_score(y_test, y_pred, labels=class_names, average='weighted').round(2))
            plot_metrics(metrics)
            st.success('Succesfully modeled!', icon="✅")

    st.sidebar.subheader("Advanced ML - Feature Selection")
    if st.sidebar.checkbox('SelectKBest - f_classif'):
        st.header('SelectKBest - f_classif')
        selectkbest_fclf = SelectKBest(score_func=f_classif, k=3)
        fit_func = selectkbest_fclf.fit(X_train, y_train)
        selectkbest_fclf_selected = X_train.columns[fit_func.get_support(indices=True)].tolist()
        selectkbest_fclf_selected.append('price_range')
        df_one_hot_fclf_subsets = df_one_hot_sample[selectkbest_fclf_selected]
        y_train_flassif = df_one_hot_fclf_subsets[
            'price_range']
        X_train_flassif = df_one_hot_fclf_subsets.drop(['price_range'], axis=1)
        X_train_flassif, X_test_flassif, y_train_flassif, y_test_flassif = train_test_split(X_train_flassif,
                                                                                            y_train_flassif,
                                                                                            test_size=0.3)
        knn_model_f_classif = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        knn_model_f_classif.fit(X_train_flassif, y_train_flassif)
        y_pred_knn_f_classif = knn_model_f_classif.predict(X_test_flassif)
        st.write("**f_classif APPLIED KNN CLASSIFIER TRAINING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of training set', round(
            cross_val_score(knn_model_f_classif, X_train_flassif, y_train_flassif, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of training set', round(
            cross_val_score(knn_model_f_classif, X_train_flassif, y_train_flassif, cv=3,
                            scoring='precision_weighted').mean(), 2))
        st.write('Recall score of training set', round(
            cross_val_score(knn_model_f_classif, X_train_flassif, y_train_flassif, cv=3,
                            scoring='recall_weighted').mean(),
            2))
        st.write('F1 score of training set', round(
            cross_val_score(knn_model_f_classif, X_train_flassif, y_train_flassif, cv=3, scoring='f1_weighted').mean(),
            2))

    if st.sidebar.checkbox('SelectKBest - chi2'):
        st.header('SelectKBest - chi2')
        selectkbest_chi2 = SelectKBest(score_func=chi2, k=3)
        fit_func_chi2 = selectkbest_chi2.fit(X_train, y_train)
        selectkbest_chi2_selected = X_train.columns[fit_func_chi2.get_support(indices=True)].tolist()
        selectkbest_chi2_selected.append('price_range')
        df_one_hot_chi2_subsets = df_one_hot_sample[selectkbest_chi2_selected]
        y_train_chi2 = df_one_hot_chi2_subsets[
            'price_range']
        X_train_chi2 = df_one_hot_chi2_subsets.drop(['price_range'], axis=1)
        X_train_chi2, X_test_chi2, y_train_chi2, y_test_chi2 = train_test_split(X_train_chi2, y_train_chi2,
                                                                                test_size=0.3)
        knn_model_chi2 = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        knn_model_chi2.fit(X_train_chi2, y_train_chi2)
        y_pred_knn_chi2 = knn_model_chi2.predict(X_test_chi2)
        st.write("**chi2 APPLIED KNN CLASSIFIER TRAINING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of training set',
                 round(cross_val_score(knn_model_chi2, X_train_chi2, y_train_chi2, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of training set', round(
            cross_val_score(knn_model_chi2, X_train_chi2, y_train_chi2, cv=3, scoring='precision_weighted').mean(), 2))
        st.write('Recall score of training set',
                 round(cross_val_score(knn_model_chi2, X_train_chi2, y_train_chi2, cv=3,
                                       scoring='recall_weighted').mean(),
                       2))
        st.write('F1 score of training set',
                 round(cross_val_score(knn_model_chi2, X_train_chi2, y_train_chi2, cv=3, scoring='f1_weighted').mean(),
                       2))

    if st.sidebar.checkbox('SelectKBest - mutual_info_classif'):
        st.header('SelectKBest - mutual_info_classif')
        selectkbest_mutual_info = SelectKBest(score_func=mutual_info_classif, k=3)
        fit_func_mutual = selectkbest_mutual_info.fit(X_train, y_train)
        selectkbest_mutual_selected = X_train.columns[fit_func_mutual.get_support(indices=True)].tolist()
        selectkbest_mutual_selected.append('price_range')
        df_one_hot_mutual_subsets = df_one_hot_sample[selectkbest_mutual_selected]
        y_train_mutual = df_one_hot_mutual_subsets[
            'price_range']
        X_train_mutual = df_one_hot_mutual_subsets.drop(['price_range'], axis=1)
        X_train_mutual, X_test_mutual, y_train_mutual, y_test_mutual = train_test_split(X_train_mutual, y_train_mutual,
                                                                                        test_size=0.3)
        knn_model_mutual = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        knn_model_mutual.fit(X_train_mutual, y_train_mutual)
        y_pred_knn_mutual = knn_model_mutual.predict(X_test_mutual)
        st.write("**mutual_info_classif APPLIED KNN CLASSIFIER TRAINING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of training set',
                 round(
                     cross_val_score(knn_model_mutual, X_train_mutual, y_train_mutual, cv=3, scoring='accuracy').mean(),
                     2))
        st.write('Precision score of training set', round(
            cross_val_score(knn_model_mutual, X_train_mutual, y_train_mutual, cv=3,
                            scoring='precision_weighted').mean(),
            2))
        st.write('Recall score of training set', round(
            cross_val_score(knn_model_mutual, X_train_mutual, y_train_mutual, cv=3, scoring='recall_weighted').mean(),
            2))
        st.write('F1 score of training set', round(
            cross_val_score(knn_model_mutual, X_train_mutual, y_train_mutual, cv=3, scoring='f1_weighted').mean(), 2))

    if st.sidebar.checkbox('SelectFromModel'):
        st.header('SelectFromModel')
        sfm = SelectFromModel(estimator=RandomForestClassifier(), max_features=3, threshold='median')
        sfm.fit(X_train, y_train)
        feature_ids = sfm.get_support()
        feature_names = X_train.columns[feature_ids].tolist()
        feature_names.append('price_range')
        df_one_hot_sfm_subsets = df_one_hot_sample[feature_names]
        y_train_sfm = df_one_hot_sfm_subsets[
            'price_range']
        X_train_sfm = df_one_hot_sfm_subsets.drop(['price_range'], axis=1)
        X_train_sfm, X_test_sfm, y_train_sfm, y_test_sfm = train_test_split(X_train_sfm, y_train_sfm, test_size=0.3)
        knn_model_sfm = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        knn_model_sfm.fit(X_train_sfm, y_train_sfm)
        y_pred_knn_sfm = knn_model_sfm.predict(X_test_sfm)

        sample_x_test_df = X_test_sfm.sample(n=10, random_state=123)
        sample_y_test = y_test_sfm.sample(n=10, random_state=123)
        sample_y_test_df = pd.DataFrame(sample_y_test, columns=['price_range'])
        sample_y_test_df.rename({'price_range': 'true_label'}, inplace=True, axis=1)

        model_pred_labels_df = pd.DataFrame(knn_model_sfm.predict(sample_x_test_df),
                                            columns=["model_label_prediction"]).reset_index(drop=True)
        model_pred_probs_df = pd.DataFrame(knn_model_sfm.predict_proba(sample_x_test_df),
                                           columns=["probability_price_range_No",
                                                    "probability_price_range_Yes"]).reset_index(
            drop=True)
        sample_y_test_df.reset_index(drop=True, inplace=True)
        concat_final_df = pd.concat([model_pred_probs_df, model_pred_labels_df, sample_y_test_df], axis=1)
        st.write(concat_final_df)

        st.write("**SFM APPLIED KNN CLASSIFIER TRAINING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of training set',
                 round(cross_val_score(knn_model_sfm, X_train_sfm, y_train_sfm, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of training set',
                 round(cross_val_score(knn_model_sfm, X_train_sfm, y_train_sfm, cv=3,
                                       scoring='precision_weighted').mean(),
                       2))
        st.write('Recall score of training set',
                 round(cross_val_score(knn_model_sfm, X_train_sfm, y_train_sfm, cv=3, scoring='recall_weighted').mean(),
                       2))
        st.write('F1 score of training set',
                 round(cross_val_score(knn_model_sfm, X_train_sfm, y_train_sfm, cv=3, scoring='f1_weighted').mean(), 2))

    if st.sidebar.checkbox('RFE (Recursive Feature Elimination)'):
        st.header('RFE (Recursive Feature Elimination)')
        rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=3, step=1, verbose=False)
        rfe.fit(X_train, y_train)
        feature_ids = rfe.support_
        feature_names = X_train.columns[feature_ids].tolist()
        feature_names.append('price_range')
        df_one_hot_rfe_subsets = df_one_hot_sample[feature_names]
        y_train_rfe = df_one_hot_rfe_subsets[
            'price_range']
        X_train_rfe = df_one_hot_rfe_subsets.drop(['price_range'], axis=1)

        X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = train_test_split(X_train_rfe, y_train_rfe, test_size=0.3)
        knn_model_rfe = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        knn_model_rfe.fit(X_train_rfe, y_train_rfe)
        y_pred_knn_rfe = knn_model_rfe.predict(X_test_rfe)
        st.write("**RFE APPLIED KNN CLASSIFIER TRAINING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of training set',
                 round(cross_val_score(knn_model_rfe, X_train_rfe, y_train_rfe, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of training set',
                 round(cross_val_score(knn_model_rfe, X_train_rfe, y_train_rfe, cv=3,
                                       scoring='precision_weighted').mean(),
                       2))
        st.write('Recall score of training set',
                 round(cross_val_score(knn_model_rfe, X_train_rfe, y_train_rfe, cv=3, scoring='recall_weighted').mean(),
                       2))
        st.write('F1 score of training set',
                 round(cross_val_score(knn_model_rfe, X_train_rfe, y_train_rfe, cv=3, scoring='f1_weighted').mean(), 2))

    if st.sidebar.checkbox('SFS (Sequential Forward Selection)'):
        st.header('SFS (Sequential Forward Selection)')
        y_train = df_one_hot_sample['price_range']
        X_train = df_one_hot_sample.drop(['price_range'], axis=1)
        X_train_sample, y_train_sample, X_test_sample, y_test_sample = train_test_split(X_train, y_train, test_size=0.3)
        knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        sfs1 = SFS(estimator=knn, k_features=3, forward=True, floating=False, verbose=False, scoring='f1', cv=3)
        feature_names = ('City_Dallas', 'City_New York City', 'City_Los Angeles', 'City_Mountain View', 'City_Boston',
                         'City_Washington D.C.', 'City_San Diego', 'City_Austin', 'Gender_Male', 'Age', 'Income')
        sfs1 = sfs1.fit(X_train, y_train)
        fig = plot_sfs(sfs1.get_metric_dict(), kind='std_err')
        plt.title('Sequential Forward Selection (w. StdErr)')
        plt.grid()
        st.pyplot()

        feature_names = list(sfs1.k_feature_names_)
        feature_names.append('price_range')

        df_one_hot_SFS_subsets = df_one_hot_sample[feature_names]
        y_train_SFS = df_one_hot_SFS_subsets[
            'price_range']
        X_train_SFS = df_one_hot_SFS_subsets.drop(['price_range'], axis=1)
        X_train_sample_SFS, X_test_sample_SFS, y_train_sample_SFS, y_test_sample_SFS = train_test_split(X_train_SFS,
                                                                                                        y_train_SFS,
                                                                                                        test_size=0.3)

        knn_model_SFS = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        knn_model_SFS.fit(X_train_sample_SFS, y_train_sample_SFS)
        y_pred_knn_SFS = knn_model_SFS.predict(X_test_sample_SFS)

        st.write("**SFS APPLIED KNN CLASSIFIER TRAINING DATASET MODEL RESULTS (CROSS-VALIDATED)**")
        st.write('Accuracy score of training set', round(
            cross_val_score(knn_model_SFS, X_train_sample_SFS, y_train_sample_SFS, cv=3, scoring='accuracy').mean(), 2))
        st.write('Precision score of training set', round(
            cross_val_score(knn_model_SFS, X_train_sample_SFS, y_train_sample_SFS, cv=3,
                            scoring='precision_weighted').mean(), 2))
        st.write('Recall score of training set', round(
            cross_val_score(knn_model_SFS, X_train_sample_SFS, y_train_sample_SFS, cv=3,
                            scoring='recall_weighted').mean(),
            2))
        st.write('F1 score of training set', round(
            cross_val_score(knn_model_SFS, X_train_sample_SFS, y_train_sample_SFS, cv=3, scoring='f1_weighted').mean(),
            2))


if __name__ == '__main__':
    main()