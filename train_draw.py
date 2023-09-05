import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from pathlib import Path
from utils.process_images import *
from utils.process_data import *
import matplotlib.pyplot as plt
import pickle




data_dir = Path(r'C:\Users\Taqana\Desktop\Harbour.Space University\Project Bootcamp\last try\Bootcamp_parkinsons\drawings')

print('[INFO] loading data...')
df = pd.DataFrame({'path': list(data_dir.glob('*/*/*/*.png'))})
df['img_id'] = df['path'].map(lambda x: x.stem)
df['disease'] = df['path'].map(lambda x: x.parent.stem)
df['validation'] = df['path'].map(lambda x: x.parent.parent.stem)
df['activity'] = df['path'].map(lambda x: x.parent.parent.parent.stem)
print(f'[INFO] {df.shape[0]} images loaded.')

print('[INFO] processing images to create features...')
df['thresh_img'] = df['path'].map(lambda x: read_and_thresh(x, resize=False))
df['clean_img'] = df['thresh_img'].map(lambda x: closing(label_sort(x)>0, disk(1)))
df['thickness'] = df['clean_img'].map(lambda x: stroke_thickness(x))
df['mean_thickness'] = df['thickness'].apply(np.mean)
df['std_thickness'] = df['thickness'].apply(np.std)
df['num_pixels'] = df['clean_img'].map(lambda x: sum_pixels(skeleton_drawing(x)))
df['num_ep'] = df['clean_img'].map(lambda x: number_of_end_points(x, k_nn))
df['num_inters'] = df['clean_img'].map(lambda x: number_of_intersection_points(x, k_nn))
# draw_df['nn_img'] = draw_df['clean_img'].map(lambda x: get_cleaned_nn_and_label(x, k_nn)[0])
# draw_df['label_img'] = draw_df['clean_img'].map(lambda x: get_cleaned_nn_and_label(x, k_nn)[1])
print('done.')
#spiral and wave separately.
activities = ['wave']
for activity in activities:
    print(f"[INFO] creating dataset for {activity}...")
    draw_df = df.loc[df['activity'] == activity]

    feature_columns = ['mean_thickness',
                       'std_thickness',
                       'num_pixels',
                       'num_ep',
                       'num_inters']
    target_column = ['disease']

    train_df = draw_df.loc[draw_df['validation'] == 'training']
    train_df = shuffle(train_df, random_state=42)
    print(f"[INFO] training samples for {activity}: {len(train_df.index)}")
    test_df = draw_df.loc[draw_df['validation'] == 'testing']
    test_df = shuffle(test_df, random_state=42)
    print(f"[INFO] testing samples for {activity}: {len(test_df.index)}...")

    X_train, y_train = train_df[feature_columns], train_df[target_column].to_numpy().ravel()
    X_test, y_test = test_df[feature_columns], test_df[target_column].to_numpy().ravel()

    #add interaction terms for all i != j columns: xi*xj
    X_train = create_interactions(X_train)
    X_test = create_interactions(X_test)

    X_train, X_test = standardize(X_train, X_test, verbose=False)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train).ravel()
    #print(lb.classes_)
    y_test = lb.transform(y_test).ravel()

    print(pdtabulate(X_train.sample(random_state = 42)))

    # print("[INFO] tuning hyperparams for LR...")
    # params_lr = {"C": [0.001, 0.01, 1.0, 10.0]}
    # lr_cls = GridSearchCV(LogisticRegression(solver="lbfgs",
    #                                         multi_class="auto",
    #                                         random_state=42),
    #                      params_lr,
    #                      cv=5,
    #                      n_jobs=-1)
    print("[INFO] fitting LR...")
    lr_cls = LogisticRegression(solver="lbfgs",
                                multi_class="auto",
                                C=1.0,
                                random_state=42)
    lr_cls.fit(X_train, y_train)
    # print("[INFO] best hyperparams for LR: {}".format(lr_cls.best_params_))

    print("[INFO] fitting RF...")
    rf_cls = RandomForestClassifier(n_estimators=100,
                                    random_state=42)
    rf_cls.fit(X_train, y_train)
    important_features_list = feature_importance(rf_cls)
    print(f'[INFO] Feature impact (in order of importance): {X_train.columns[important_features_list].values}')

    print("[INFO] evaluating...")
    rf_preds = rf_cls.predict(X_test)
    lr_preds = lr_cls.predict(X_test)

    rf_accuracy = (rf_preds == y_test).mean()
    lr_accuracy = (lr_preds == y_test).mean()

    print(f'[INFO] RF accuracy: {rf_accuracy}')
    print(f'[INFO] LR accuracy: {lr_accuracy}')



# Save imge 
import pickle

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Save your RF model
save_model(rf_cls, 'rf_model.pickle')
