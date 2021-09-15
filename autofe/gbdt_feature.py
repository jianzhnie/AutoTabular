from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin


class gbdt_lr(BaseEstimator, ClassifierMixin):

    def __init__(self, gbdt=None, lr=None,gbdt_params=None,lr_params=None,cv=CountVectorizer(analyzer='word',preprocessor=None,ngram_range=(1,1),stop_words=None,min_df=0,)):
        self.gbdt=gbdt(**gbdt_params)
        self.lr=lr(**lr_params)
        self.cv=cv
    def fit(self, X, y):
        self.gbdt.fit(X,y)
        leaf = (self.gbdt.predict(X, pred_leaf=True)).astype(str).tolist()

        leaf=[' '.join(item) for item in leaf]
        self.result=self.cv.fit_transform(leaf)
        X=
        self.lr.fit(self.result,y)
        return self
    
    def predict_proba(self, X):
        leaf=self.gbdt.predict(X, pred_leaf=True)
        leaf = (self.gbdt.predict(X, pred_leaf=True)).astype(str).tolist()
        leaf=[' '.join(item) for item in leaf]
        result=self.cv.transform(leaf)
        
        return self.lr.predict_proba(result)
     


def gbdt_lr_predict(data, category_feature, continuous_feature): # 0.43616
    # 离散特征one-hot编码
    print('开始one-hot...')
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')

    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis = 1, inplace = True)

    # 划分数据集
    print('划分数据集...')
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 2018)

    print('开始训练gbdt..')
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=100,
                            max_depth = 12,
                            learning_rate=0.05,
                            n_estimators=10,
                            )

    gbm.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric = 'binary_logloss',
            # early_stopping_rounds = 100,
            )
    model = gbm.booster_
    print('训练得到叶子数')
    gbdt_feats_train = model.predict(train, pred_leaf = True)
    gbdt_feats_test = model.predict(test, pred_leaf = True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)

    print('构造新的数据集...')
    train = pd.concat([train, df_train_gbdt_feats], axis = 1)
    test = pd.concat([test, df_test_gbdt_feats], axis = 1)
    train_len = train.shape[0]
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()




from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

gbm1 = GradientBoostingClassifier(n_estimators=50, random_state=10, subsample=0.6, max_depth=7,
                                  min_samples_split=900)
gbm1.fit(X_train, Y_train)
train_new_feature = gbm1.apply(X_train)
train_new_feature = train_new_feature.reshape(-1, 50)

enc = OneHotEncoder()

enc.fit(train_new_feature)

# # 每一个属性的最大取值数目
# print('每一个特征的最大取值数目:', enc.n_values_)
# print('所有特征的取值数目总和:', enc.n_values_.sum())

train_new_feature2 = np.array(enc.transform(train_new_feature).toarray())




params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}


# number of leaves,will be used in feature transformation
num_leaf = 64

print('Start training...')
# train
gbm = lgb.train(params=params,
                train_set=lgb_train,
                valid_sets=lgb_train, )


print('Start predicting...')
# y_pred分别落在100棵树上的哪个节点上
y_pred = gbm.predict(x_train, pred_leaf=True)
y_pred_prob = gbm.predict(x_train)


result = []
threshold = 0.5
for pred in y_pred_prob:
    result.append(1 if pred > threshold else 0)
print('result:', result)


print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[1]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred)):
    # temp表示在每棵树上预测的值所在节点的序号（0,64,128,...,6436 为100棵树的序号，中间的值为对应树的节点序号）
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    # 构造one-hot 训练数据集
    transformed_training_matrix[i][temp] += 1

y_pred = gbm.predict(x_test, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[1]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    # 构造one-hot 测试数据集
    transformed_testing_matrix[i][temp] += 1