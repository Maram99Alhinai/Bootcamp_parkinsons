import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from pathlib import Path
from utils.process_images import *
from utils.process_data import *
import matplotlib.pyplot as plt
import joblib

# Define a function for data processing
def process_data(data_dir, activity):
    df = pd.DataFrame({'path': list(data_dir.glob(f'{activity}//*//*//*.png'))})
    df['img_id'] = df['path'].map(lambda x: x.stem)
    df['disease'] = df['path'].map(lambda x: x.parent.stem)
    df['validation'] = df['path'].map(lambda x: x.parent.parent.stem)
    df['activity'] = df['path'].map(lambda x: x.parent.parent.parent.stem)

    df['thresh_img'] = df['path'].map(lambda x: read_and_thresh(x, resize=False))
    df['clean_img'] = df['thresh_img'].map(lambda x: closing(label_sort(x) > 0, disk(1)))
    df['thickness'] = df['clean_img'].map(lambda x: stroke_thickness(x))
    df['mean_thickness'] = df['thickness'].apply(np.mean)
    df['std_thickness'] = df['thickness'].apply(np.std)
    df['num_pixels'] = df['clean_img'].map(lambda x: sum_pixels(skeleton_drawing(x)))
    df['num_ep'] = df['clean_img'].map(lambda x: number_of_end_points(x, k_nn))
    df['num_inters'] = df['clean_img'].map(lambda x: number_of_intersection_points(x, k_nn))

    feature_columns = ['mean_thickness', 'std_thickness', 'num_pixels', 'num_ep', 'num_inters']
    target_column = ['disease']

    train_df = df.loc[df['validation'] == 'training']
    train_df = shuffle(train_df, random_state=42)
    print(f"[INFO] training samples for {activity}: {len(train_df.index)}")
    test_df = df.loc[df['validation'] == 'testing']
    test_df = shuffle(test_df, random_state=42)
    print(f"[INFO] testing samples for {activity}: {len(test_df.index)}...")

    X_train, y_train = train_df[feature_columns], train_df[target_column].to_numpy().ravel()
    X_test, y_test = test_df[feature_columns], test_df[target_column].to_numpy().ravel()

    X_train = create_interactions(X_train)
    X_test = create_interactions(X_test)

    X_train, X_test = standardize(X_train, X_test, verbose=False)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train).ravel()
    y_test = lb.transform(y_test).ravel()

    return X_train, X_test, y_train, y_test, lb

# Define a function for training and evaluating the random forest model
def train_and_evaluate_rf(X_train, X_test, y_train, y_test, lb):
    print("[INFO] fitting RF...")
    rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cls.fit(X_train, y_train)
    
    print("[INFO] evaluating...")
    rf_preds = rf_cls.predict(X_test)

    print(classification_report(y_test, rf_preds, target_names=lb.classes_))
    print(f'[INFO] RF accuracy: {rf_cls.score(X_test, y_test)}')

    # Save the trained model to a file
    model_filename = f'rf_model_{activity}.joblib'
    joblib.dump(rf_cls, model_filename)
    print(f"[INFO] Saved the random forest model to {model_filename}")


data_dir = Path(r'drawings')
activity = 'wave'

X_train, X_test, y_train, y_test, lb = process_data(data_dir, activity)
train_and_evaluate_rf(X_train, X_test, y_train, y_test, lb)
