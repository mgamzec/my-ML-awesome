
################################################################
# YAPAY ÖĞRENME İLE YETENEK AVCILIĞI SINIFLANDIRMA
################################################################

################################################################
# İş Problemi:
################################################################
# Scoutlar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.


################################################################
# Veriseti Hikayesi:
################################################################

# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.
# attributes: Oyuncuları değerlendiren kullanıcıların bir maçta izleyip değerlendirdikleri her oyuncunun özelliklerine verdikleri puanları içeriyor. (bağımsız değişkenler)
#potential_labels: Oyuncuları değerlendiren kullanıcıların her bir maçta oyuncularla ilgili nihai görüşlerini içeren potansiyel etiketlerini içeriyor. (hedef değişken)
# 9 Değişken, 10730 Gözlem, 0.65 mb


################################################################
# Değişkenler:
################################################################

# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi.

# match_id: İlgili maçın id'si.

# evaluator_id: Değerlendiricinin(scout'un) id'si.

# player_id: İlgili oyuncunun id'si.

# position_id: İlgili oyuncunun o maçta oynadığı pozisyonun id'si.

# 1- Kaleci
# 2- Stoper
# 3- Sağ bek
# 4- Sol bek
# 5- Defansif orta saha
# 6- Merkez orta saha
# 7- Sağ kanat
# 8- Sol kanat
# 9- Ofansif orta saha
# 10- Forvet


# analysis_id: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme.

# attribute_id: Oyuncuların değerlendirildiği her bir özelliğin id'si.

# attribute_value: Bir scoutun bir oyuncunun bir özelliğine verilen değer(puan).

# potential_label: Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)




import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import *
from sklearn.model_selection import cross_val_predict
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

################################################################
# Görev 1: Veri Setinin Hazırlanması
################################################################
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)
################################################################
# Adım 1: 2 csv dosyasını okutunuz.
################################################################

df = pd.read_csv(r"C:\Users\Sade\Desktop\VBO\python\Modül_6_Makine_Öğrenmesi\Scoutium\scoutium_attributes.csv", sep=";")
df.head()
df2 = pd.read_csv(r"C:\Users\Sade\Desktop\VBO\python\Modül_6_Makine_Öğrenmesi\Scoutium\scoutium_potential_labels.csv", sep=";")
df2.head()
################################################################
# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz. ("task_response_id", 'match_id', 'evaluator_id' "player_id"  4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
################################################################

dff = pd.merge(df, df2, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])

df.shape
df2.shape
dff.shape
################################################################
# Adım 3: position_id içerisindeki Kaleci (1) sınıfını verisetinden kaldırınız.
################################################################
#neden olduğunu konuşabiliriz

dff = dff[dff["position_id"] != 1]


################################################################
# Adım 4: potential_label içerisindeki below_average sınıfını verisetinden kaldırınız.
# ( below_average sınıfı  tüm verisetinin %1'ini oluşturur)
################################################################


dff = dff[dff["potential_label"] != "below_average"]


################################################################
#Adım 5: Oluşturduğunuz verisetinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz.
# Bu pivot table'da her satır bir oyuncu olacak şekilde manipülasyon yapınız.
################################################################

################################################################
#Her sütunda oyuncunun “position_id”, “potential_label” ve her oyuncunun
# sırayla bütün “attribute_idleri” içerecek şekilde işlem yapınız.
################################################################

pt = pd.pivot_table(dff, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])
pt.head()

################################################################
#“reset_index” fonksiyonunu kullanarak index hatasından kurtulunuz ve
# “attribute_id” sütunlarının isimlerini stringe çeviriniz. (df.columns.map(str))
################################################################

pt = pt.reset_index(drop=False)
pt.columns = pt.columns.map(str)



################################################################
# Görev 3: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye kaydediniz.
################################################################

num_cols = pt.columns[3:]

#num_cols = [col for col in df_final.columns if col not in ["player_id","position_id","potential_label"]]

##################################
# GÖREV 4: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# Adım 1: GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(pt)


##################################
# Adım 2:  Numerik ve kategorik değişkenleri inceleyiniz.
##################################

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in ["position_id","potential_label"]:
    cat_summary(pt, col)



##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(pt, col, plot=False)


##################################
# Adım 3:  Numerik değişkenler ile hedef değişken incelemesini yapınız.
##################################

##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(pt, "potential_label", col)


##################################
# Adım 4: Korelasyona bakınız.
##################################

pt[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pt[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)


##################################
# GÖREV 5: Feature Extraction uygulayın.
##################################

pt.head()
pt["min"] = pt[num_cols].min(axis=1)
pt["max"] = pt[num_cols].max(axis=1)
pt["sum"] = pt[num_cols].sum(axis=1)
pt["mean"] = pt[num_cols].mean(axis=1)
pt["median"] = pt[num_cols].median(axis=1)

pt.head()
pt["mentality"] = pt["position_id"].apply(lambda x: "defender" if (x == 2) | (x == 5) | (x == 3) | (x == 4) else "attacker")


pt.head()
pt.shape

for i in pt.columns[3:-6]:
    threshold = pt[i].mean() + pt[i].std()

    lst = pt[i].apply(lambda x: 0 if x < threshold else 1)
    pt[str(i) + "_FLAG"] = lst

pt.shape

flagCols = [col for col in pt.columns if "_FLAG" in col]

pt["counts"] = pt[flagCols].sum(axis=1)

pt["countRatio"] = pt["counts"] / len(flagCols)

pt.head()



"""avgDf = pt.groupby(["position_id", "potential_label"]).agg(["mean"]).reset_index()
avgDf = avgDf[avgDf["potential_label"] == "average"]
avgDf.columns[3:-41][0][0]"""

"""# HATA VAR
for i in range(2, 11):
    sozluk = {}
    for j in avgDf.columns[3:-41]:
        df = pt[pt["position_id"] == i]
        sozluk[j[0]] = df[j[0]].mean()
    print(i, sozluk)
    #pt[i[0]]

pt["4322"][3]"""


#
#pt[pt["counts"] == 0]["potential_label"].value_counts()
#"""
#average        93
#highlighted    15
#Name: potential_label, dtype: int64
#"""
#
#pt[pt["counts"] != 0]["potential_label"].value_counts()
#"""
#average        122
#highlighted     41
#Name: potential_label, dtype: int64
#"""
#
#########################################################

pt.head()

################################################################
# Görev 6:  Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
################################################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


labelEncoderCols = ["potential_label","mentality"]

for col in labelEncoderCols:
    pt = label_encoder(pt, col)

pt.head()


################################################################
# Görev 7: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için standardScaler uygulayınız.
################################################################

pt.head()
lst = ["counts", "countRatio","min","max","sum","mean","median"]
num_cols = list(num_cols)

for i in lst:
    num_cols.append(i)

scaler = StandardScaler()
pt[num_cols] = scaler.fit_transform(pt[num_cols])

pt.head()


################################################################
# Görev 8: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz.
################################################################


y = pt["potential_label"]
X = pt.drop(["potential_label", "player_id"], axis=1)

pt.head()
models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   #("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   #('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]



for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))
"""

LR
roc_auc score:0.8582683982683983
f1 score:0.6233061383061383
precision score:0.8071428571428572
recall score:0.55
accuracy score:0.871031746031746


LR
roc_auc score:0.8452886002886002
f1 score:0.5684648684648684
precision score:0.7738095238095238
recall score:0.49000000000000005
accuracy score:0.8525132275132276






KNN
roc_auc score:0.7391125541125542
f1 score:0.44740259740259747
precision score:0.8099999999999999
recall score:0.3266666666666666
accuracy score:0.8414021164021165

KNN
roc_auc score:0.7256998556998557
f1 score:0.4278571428571428
precision score:0.775
recall score:0.30999999999999994
accuracy score:0.8449735449735449





SVC
roc_auc score:0.859112554112554
f1 score:0.20238095238095238
precision score:0.6
recall score:0.12333333333333334
accuracy score:0.8191798941798941

SVC
roc_auc score:0.8439105339105339
f1 score:0.03333333333333334
precision score:0.1
recall score:0.02
accuracy score:0.797089947089947




CART
roc_auc score:0.7265584415584415
f1 score:0.590905760905761
precision score:0.5240079365079365
recall score:0.57
accuracy score:0.8078042328042327

CART
roc_auc score:0.7118398268398268
f1 score:0.5659995559995561
precision score:0.5709126984126984
recall score:0.5933333333333333
accuracy score:0.8375661375661376





RF
roc_auc score:0.8990367965367966
f1 score:0.5803751803751804
precision score:0.8666666666666668
recall score:0.47000000000000003
accuracy score:0.8783068783068784

RF
roc_auc score:0.9054797979797978
f1 score:0.5857142857142856
precision score:0.925
recall score:0.4533333333333333
accuracy score:0.8781746031746032








GBM
roc_auc score:0.848968253968254
f1 score:0.5327489177489177
precision score:0.6866666666666666
recall score:0.4533333333333333
accuracy score:0.8488095238095237

GBM
roc_auc score:0.8803968253968254
f1 score:0.5772799422799422
precision score:0.8016666666666667
recall score:0.5233333333333333
accuracy score:0.8597883597883598






XGBoost
roc_auc score:0.8651010101010101
f1 score:0.6123199023199024
precision score:0.7169047619047619
recall score:0.5733333333333333
accuracy score:0.8563492063492063

XGBoost
roc_auc score:0.8558080808080808
f1 score:0.611030081030081
precision score:0.7397619047619048
recall score:0.5766666666666665
accuracy score:0.8563492063492063







CatBoost
roc_auc score:0.905122655122655
f1 score:0.6079220779220779
precision score:0.9099999999999999
recall score:0.48666666666666664
accuracy score:0.8818783068783069

CatBoost
roc_auc score:0.9040981240981241
f1 score:0.5937662337662338
precision score:0.93
recall score:0.47000000000000003
accuracy score:0.8817460317460318





LightGBM
roc_auc score:0.8910822510822511
f1 score:0.6175613275613275
precision score:0.7513095238095238
recall score:0.5733333333333333
accuracy score:0.8634920634920634

LightGBM
roc_auc score:0.8982034632034631
f1 score:0.6632539682539682
precision score:0.8071428571428572
recall score:0.5933333333333333
accuracy score:0.8817460317460318













RF
roc_auc score:0.9045382395382395
f1 score:0.6121789321789322
precision score:0.8416666666666666
recall score:0.49333333333333335
accuracy score:0.8781746031746032
GBM
roc_auc score:0.8737301587301587
f1 score:0.5732323232323233
precision score:0.7483333333333333
recall score:0.4533333333333333
accuracy score:0.859920634920635
XGBoost
roc_auc score:0.8766233766233766
f1 score:0.6451770451770451
precision score:0.7835714285714285
recall score:0.5933333333333333
accuracy score:0.8744708994708995
LightGBM
roc_auc score:0.8910533910533911
f1 score:0.6409307359307359
precision score:0.8266666666666668
recall score:0.5633333333333332
accuracy score:0.8818783068783069




"""
pt.head()


################################################################
# Görev 9: Hiperparametre Optimizasyonu yapınız.
################################################################

lgbm_model = LGBMClassifier(random_state=46)

#rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
             }



lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

# normal y cv süresi: 16.2s
# scale edilmiş y ile: 13.8s

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(rmse)




################################################################
# Görev 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")

#model = LGBMClassifier()
#model.fit(X, y)

plot_importance(final_model, X)
plot_importance(model, X)


