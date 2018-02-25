import pandas as pd
from collections import Counter
from sklearn.model_selection import cross_val_score
import numpy as np

#https://archive.ics.uci.edu/ml/datasets/Automobile
'''1. symboling: -3, -2, -1, 0, 1, 2, 3. 
2. normalized-losses: continuous from 65 to 256. 
3. make: 
alfa-romero, audi, bmw, chevrolet, dodge, honda, 
isuzu, jaguar, mazda, mercedes-benz, mercury, 
mitsubishi, nissan, peugot, plymouth, porsche, 
renault, saab, subaru, toyota, volkswagen, volvo 

4. fuel-type: diesel, gas. 
5. aspiration: std, turbo. 
6. num-of-doors: four, two. 
7. body-style: hardtop, wagon, sedan, hatchback, convertible. 
8. drive-wheels: 4wd, fwd, rwd. 
9. engine-location: front, rear. 
10. wheel-base: continuous from 86.6 120.9. 
11. length: continuous from 141.1 to 208.1. 
12. width: continuous from 60.3 to 72.3. 
13. height: continuous from 47.8 to 59.8. 
14. curb-weight: continuous from 1488 to 4066. 
15. engine-type: dohc, dohcv, l, ohc, ohcf, ohcv, rotor. 
16. num-of-cylinders: eight, five, four, six, three, twelve, two. 
17. engine-size: continuous from 61 to 326. 
18. fuel-system: 1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi. 
19. bore: continuous from 2.54 to 3.94. 
20. stroke: continuous from 2.07 to 4.17. 
21. compression-ratio: continuous from 7 to 23. 
22. horsepower: continuous from 48 to 288. 
23. peak-rpm: continuous from 4150 to 6600. 
24. city-mpg: continuous from 13 to 49. 
25. highway-mpg: continuous from 16 to 54. 
26. price: continuous from 5118 to 45400.'''
#==============================================================================

train_df = pd.read_csv("automobile.csv")

'''print("Tratando nulos")
lista = []
for c in train_df.columns:
	for a in train_df[c]:
		if a == '?':
			lista.append(c)
	if c in lista:
		#print(c, 'possui', lista.count(c), 'campos nulos')
		print(c, ':', lista.count(c))'''

def limparNuloEConverter(field, decimal):
	soma = 0
	for value in train_df[field]:
		if value != '?':
			soma += float(value)
	media = soma/len(train_df[field])
	'''print(field,'media:', media)'''
	convetNumeroInt = lambda value: int(float(value)) if value != '?' else int(media)
	convetNumeroFloat = lambda value: float(value) if value != '?' else media
	train_df[field] = train_df[field].apply(convetNumeroFloat) if decimal == True else train_df[field].apply(convetNumeroInt)

limparNuloEConverter('normalized-losses', False)
limparNuloEConverter('symboling', False)
limparNuloEConverter('wheel-base', True)
limparNuloEConverter('length', True)
limparNuloEConverter('width', True)
limparNuloEConverter('height', True)
limparNuloEConverter('curb-weight', False)
limparNuloEConverter('engine-size', False)
limparNuloEConverter('curb-weight', False)
limparNuloEConverter('bore', True)
limparNuloEConverter('stroke', True)
limparNuloEConverter('horsepower', False)
limparNuloEConverter('peak-rpm', False)
limparNuloEConverter('price', False)

print('Limpando portas')
train_df['num-of-doors'] = train_df['num-of-doors'].apply(lambda v: 'two' if v=='?' else v)

print("Tratando campos multivalorados")
X_df = train_df[['fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']]
Xdumies_df = pd.get_dummies(X_df).astype(int)


train_df.__delitem__('body-style')
train_df.__delitem__('num-of-cylinders')
train_df.__delitem__('fuel-type')
train_df.__delitem__('aspiration')
train_df.__delitem__('num-of-doors')
train_df.__delitem__('drive-wheels')
train_df.__delitem__('engine-location')
train_df.__delitem__('engine-type')
train_df.__delitem__('fuel-system')
train_df = pd.concat([train_df,Xdumies_df], axis=1)

print("Convertendo a empresa em um numero")
make = list(train_df.make.unique())
convert_make = lambda m: make.index(m['make'])
train_df["make"] = train_df.apply(convert_make, axis=1)

print("Separando dados de treino e teste")

X_df = train_df
Y_df = train_df['symboling']
X_df.__delitem__('symboling')

X = X_df.values
Y = Y_df.values

tamanho_de_treino = 150

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[tamanho_de_treino:]
teste_marcacoes = Y[tamanho_de_treino:]

'''
for x in X_df:
	print(x, list(X_df[x]))
print(Y)'''

print("Aplicando algoritimos")
def fit_and_predict(nome,modelo, treino_dados, treino_marcacoes,teste_dados,teste_marcacoes):
	modelo.fit(treino_dados, treino_marcacoes)
	resultado = modelo.predict(teste_dados)
	acertos = 0
	tamanho = len(teste_marcacoes)
	for i in range(tamanho):
		if teste_marcacoes[i] == resultado[i]:
			acertos = acertos+1
	print('%s: %.2f' % (nome,(acertos*100/tamanho)))

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados,teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
resultadoOneVesRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.multiclass import OneVsOneClassifier
modelOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
resultadoOneVsOne = fit_and_predict("OneVSOne", modelOneVsOne, treino_dados, treino_marcacoes, teste_dados,teste_marcacoes)

from sklearn.ensemble import RandomForestClassifier
modelRandomForest = RandomForestClassifier()
resultadoRandomForest = fit_and_predict("Random Forest", modelRandomForest, treino_dados, treino_marcacoes, teste_dados,teste_marcacoes)


from sklearn.cluster  import KMeans 
modelKMeans = KMeans(n_clusters = 3, init = 'random')
resultadoKMeans = fit_and_predict("KMeans", modelKMeans, treino_dados, treino_marcacoes, teste_dados,teste_marcacoes)

'''for i in range(0,50):
	modelKMeans = KMeans(n_clusters = 3, init = 'random')
	resultadoKMeans = fit_and_predict("KMeans", modelKMeans, treino_dados, treino_marcacoes, teste_dados,teste_marcacoes)
'''
#====== 10 ====================#
print("Algoritimo com folding")
k = 7
def fit_and_predict_folding(nome,modelo, treino_dados, treino_marcacoes,k):
	scores = cross_val_score(modelo,treino_dados, treino_marcacoes, cv=k)
	taxa_de_acerto = np.mean(scores)
	print('%s: %.2f' % (nome,taxa_de_acerto*100))

fit_and_predict_folding("MultinomialNB - Folding", modeloMultinomial, treino_dados,treino_marcacoes,k)
fit_and_predict_folding("AdaBoostClassifier - Folding", modeloAdaBoost, treino_dados,treino_marcacoes,k)
fit_and_predict_folding("OneVsRest - Folding", modeloOneVsRest, treino_dados,treino_marcacoes,k)
fit_and_predict_folding("OneVSOne - Folding", modelOneVsOne, treino_dados,treino_marcacoes,k)
fit_and_predict_folding("Random Forest - Folding", modelRandomForest, treino_dados,treino_marcacoes,k)

