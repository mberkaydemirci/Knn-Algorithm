import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#dataseti çektim
dataset = pd.read_csv("C:/Users/mehme/Desktop/alışveriş.csv")
#data sette x değişkenine yaş,tahmini maaş ve satin aldı mıyı verdim
X = dataset.iloc[:, [-1,2,3]].values
#yye cinsiyet verdim cinsiyeti tahmin etmeye çalışıyoruz
y = dataset.iloc[:, 4].values
#test boyutunu %20 olarak ayarladım ve rasgele durumu 0 a eşitledim 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = 0.2)
#komşuluk sınıflandırmasında minkowskiyi kullandım komşu sayisini 3 girdim
#çünkü en doğru sonucu 3de veriyor ve öklid kullandım
knn = KNeighborsClassifier(metric='minkowski', n_neighbors=5, p=2,)
#birimler farklı olduğu için normalize ettim
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
#k:fold cross validation kullandım 10 parçaya böldüm ort skoru hesaplattım
knn = KNeighborsClassifier(n_neighbors= 1)
scores = cross_val_score(knn, X, y , cv = 10, scoring= 'accuracy')
print("Çapraz Geçerleme Skorları")
print(scores)
print("Çapraz Geçerleme Skorı Ortalamaları")
print(scores.mean())
#en doğru sonuç için forla gezdim en iyi sonucu 1de verdiğini öğrendim tüm aralıkları denedim
#net gözükmesi için grafikte 1 ile 50 arası yaptım foru 
k_range = range(1, 50)
#k skorlarını diziye atadım aşağıda ondan burda tanımladım
k_scores = []
for k in k_range:
  knn = KNeighborsClassifier(n_neighbors= k)
  scores = cross_val_score(knn, X, y , cv = 5, scoring= 'accuracy')
  k_scores.append(scores.mean())
#karmaşıklık matrisini hesaplatıp yazdırdım
result = confusion_matrix(y_test, y_pred)
print("Karmaşıklık Matrisi:")
print(result)
#raporlamayı kütüphane üzerinden çektim tüm değerleri hesaplaması için
result1 = classification_report(y_test, y_pred)
print("Sınıflandırma Raporu:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Doğruluk:",result2)
#grafiği 
sn.lineplot(x = k_range, y = k_scores);
