#Diabetes Project AI10+2
import pandas as pd
df=pd.read_csv(r'diabetes_data.csv')
print(df.columns)
df.head()
df.describe()
columns_to_replace=['Glucose','BloodPressure', 'SkinThickness','Insulin','BMI']
for col in columns_to_replace:
 df[col].replace(0,df[col].mean(),inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='39'.2f'39')
plt.title('Correlation Heatmap')
plt.show()
from sklearn.preprocessing import StandardScaler
scale_X=StandardScaler()
X=scale_X.fit_transform(df.drop(['Outcome'],axis=1))
X
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,df[&#39;Outcome&#39;],test_size=0.2,random_state=42,shuffle=T
rue)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

k_values=range(1,16)
knn_models=[KNeighborsClassifier(n_neighbors=k) for k in k_values]
cv_scores = [cross_val_score(model,X_train,Y_train,cv=5).mean()
for model in knn_models]
best_k=k_values[cv_scores.index(max(cv_scores))]
{best_k}
best_knn=KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train,Y_train)
accuracy=best_knn.score (X_test,Y_test)
print(f&quot;Accuracy of the selected model:{accuracy:.2f}&quot;)
