import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plot
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ucimlrepo import fetch_ucirepo
spambase = fetch_ucirepo(id=94)



X = spambase.data.features
y = spambase.data.targets
y = y.values.flatten()
column_names = spambase.variables['name'].tolist()
df_features = pd.DataFrame(X, columns=column_names[:-1])
df_targets = pd.Series(y, name=column_names[-1])
df = pd.concat([df_features, df_targets], axis=1)

# balanceamento e escala
df_positives = df[df['Class'] == 1]
df_negatives = df[df['Class'] == 0]
df_negatives = resample(df_negatives, replace=False, n_samples=1813, random_state=1)
df_downsampled = pd.concat([df_positives, df_negatives], axis=0)
X = df_downsampled[column_names[:-1]]
y = df_downsampled['Class']
X_scaled = scale(X)

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
C_values = numpy.logspace(-1, 2, 6)

accuracies = []
precisions = []
recalls = []
f1_scores = []

print("debug1\n")


for C in C_values:
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []
    print(f'debug 2 testando C = {C:.2f}...')

    for fold_index, (train_index, test_index) in enumerate(kf.split(X_scaled), start=1):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        svm = SVC(kernel='linear', C=C)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1_scores.append(f1)

        print(f"  Fold {fold_index}: acuracia = {accuracy:.4f}, precisao = {precision:.4f}, revocacao = {recall:.4f}, f1 = {f1:.4f}")

    avg_accuracy = numpy.mean(fold_accuracies)
    avg_precision = numpy.mean(fold_precisions)
    avg_recall = numpy.mean(fold_recalls)
    avg_f1 = numpy.mean(fold_f1_scores)

    accuracies.append(avg_accuracy)
    precisions.append(avg_precision)
    recalls.append(avg_recall)
    f1_scores.append(avg_f1)

    print(f'\ncuracia media para C = {C:.2f}: {avg_accuracy:.4f}')
    print(f'precrecisão media para C = {C:.2f}: {avg_precision:.4f}')
    print(f'revocação media para C = {C:.2f}: {avg_recall:.4f}')
    print(f'F1 media para C = {C:.2f}: {avg_f1:.4f}\n')

plot.figure(figsize=(8, 6))
plot.plot(C_values, accuracies, marker='o', color='b', label='Acurácia')
plot.plot(C_values, precisions, marker='o', color='g', label='Precisão')
plot.plot(C_values, recalls, marker='o', color='r', label='Revocação')
plot.plot(C_values, f1_scores, marker='o', color='purple', label='F1-Score')
plot.xscale('log')
plot.title('Desempenho da SVM com Variação de C')
plot.xlabel('Hiperparâmetro C (escala logarítmica)')
plot.ylabel('Média das Métricas de Desempenho')
plot.legend()
plot.grid(True)
plot.show()

best_index = numpy.argmax(accuracies)
best_C = C_values[best_index]
print(f'C com base na acurca: {best_C}')
print(f'Acuracia media para o melhor C = {best_C:.2f}: {accuracies[best_index]:.4f}')
print(f'precisao media para o melhor C = {best_C:.2f}: {precisions[best_index]:.4f}')
print(f'revoc media para o melhor C = {best_C:.2f}: {recalls[best_index]:.4f}')
print(f'F1 media para o melhor C = {best_C:.2f}: {f1_scores[best_index]:.4f}')



#factcheck n usar

#df = pd.DataFrame(spambase.data.features, columns=spambase.data.feature_names)
#df["class"] = spambase.data.targets
#spam_count = df[df["class"] == 1].shape[0]
#non_spam_count = df[df["class"] == 0].shape[0]
#total_instances = df.shape[0]
#spam_percentage = (spam_count / total_instances) * 100
#non_spam_percentage = (non_spam_count / total_instances) * 100

#print(spam_percentage")
#print(non_spam_percentage)