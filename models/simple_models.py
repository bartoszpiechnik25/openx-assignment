import pandas as pd, joblib
from typing import Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def simple_heuristic(elevation: float) -> int:
    """
    This function implements a simple heuristic for the covertype dataset.

    Args:
        elevation (float): The elevation data to predict on.

    Returns:
        int: The predicted target.
    """
    if elevation < 2500:
        return 1
    elif elevation < 2750:
        return 2
    elif elevation < 3000:
        return 3
    elif elevation < 3250:
        return 4
    elif elevation < 3500:
        return 5
    elif elevation < 3750:
        return 6
    else:
        return 7


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[DecisionTreeClassifier, float]:
    """
    This function trains a decision tree model and test the model accuracy.

    Args:
        data (pd.DataFrame): The data to train the model on.

    Returns:
        Tuple[DecisionTreeClassifier, float]: The trained model and the accuracy of the model on unseen data.
    """
    assert len(X_train) == len(y_train), "X_train and y_train must be of the same length"
    assert len(X_test) == len(y_test), "X_test and y_test must be of the same length"

    #Define decision tree model
    model = DecisionTreeClassifier(random_state=234, max_depth=3)

    #Train model
    model.fit(X_train, y_train)

    #Validate model
    test_acc = model.score(X_test, y_test)

    #Save model
    joblib.dump(model, "./models/checkpoints/decision_tree.joblib")

    return model, test_acc
    

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[LogisticRegression, MinMaxScaler, float]:
    """
    This function trains a KNN model and test the model accuracy.

    Args:
        data (pd.DataFrame): The data to train the model on.

    Returns:
        Tuple[LogisticRegression, float]: The trained model and the accuracy of the model.
    """
    assert len(X_train) == len(y_train), "X_train and y_train must be of the same length"
    assert len(X_test) == len(y_test), "X_test and y_test must be of the same length"

    #Define KNN model
    model = LogisticRegression(multi_class='multinomial',
                                n_jobs=-1,
                                solver='lbfgs',
                                max_iter=100)                                

    #Scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Train model
    model.fit(X_train, y_train)

    #Validate model
    test_acc = model.score(X_test, y_test)

    #Save model
    joblib.dump(model, "./models/checkpoints/logistic_regression.joblib")
    joblib.dump(scaler, "./models/checkpoints/regression_scaler.joblib")

    return model, scaler, test_acc

def evaluate_model(model: Union[LogisticRegression, DecisionTreeClassifier, str],
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   title: str='',
                   path: str='') -> None:
    """
    This function evaluates a model.

    Args:
        model (Union[LogisticRegression, DecisionTreeClassifier, str]): The model to evaluate.
        X_test (pd.DataFrame): Test input data.
        y_test (pd.DataFrame): Test target data.
        title (str, optional): Name of the .png file with confusion matrix. Defaults to ''.
    """
    if isinstance(model, str):
        elevation = X_test.iloc[:, 0].to_list()
        y_pred = [simple_heuristic(e) for e in elevation]
    else:
        y_pred = model.predict(X_test)
    
    classes = [i + 1 for i in range(7)]
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=classes)
    
    print(f"Accuracy: {accuracy:.4f}")
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.title(f"{title} confusion matrix\nTest accuracy: {accuracy:.4f}")
    plt.savefig(f'{path}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def train_and_eval(data: pd.DataFrame) -> None:
    """
    This function trains and evaluates heuristic, decision tree and Logistic Regression models.

    Args:
        data (pd.DataFrame): The data to train and evaluate the models on.
    """
    #Split data into train and test. Here we are splitting 90% of the data into training and 10% into testing
    # because in the nn model 80% is train 10% is validation and 10% is test, so we want to make sure that
    # metrics are comparable
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=234)

    #Train and evaluate decision tree model
    tree, _t = train_decision_tree(X_train, y_train, X_test, y_test)
    print("\nDecision tree accuracy:")
    evaluate_model(tree, X_test, y_test, title='Decision tree', path='./models/evaluation/decision_tree')

    #Train and evaluate heuristic model
    print("Heuristic accuracy:")
    evaluate_model('heuristic', X_test, y_test, title='Heuristic', path='./models/evaluation/heuristic')

    #Train and evaluate logistic regression model
    lr, scaler, _ = train_logistic_regression(X_train, y_train, X_test, y_test)
    print("\nLogistic Regression accuracy:")
    evaluate_model(lr, scaler.transform(X_test), y_test, title='Logistic Regression', path='./models/evaluation/logistic_regression')
    
