import numpy as np
from numpy import genfromtxt
from functools import reduce
#--- przerabianie grzybow ---#
mushroom = genfromtxt('dane/mushroom.csv', delimiter=',' ,dtype = str)

classes = mushroom[:,0].copy()
last = mushroom[:,-1].copy()
training_set= mushroom
training_set[:,0]= last
training_set[:,-1]=classes

def f(x):
    if x=='p':
        return 1
    if x =='e':
        return 0

binary_classes = [f(x) for x in classes]
b  = np.reshape(binary_classes,(8124, 1))


def toBinaryFeatures(features):
    COLUMNS = features.shape[1]
    v = [x + str(i % COLUMNS) for i, x in enumerate(features.flatten())]
    l = features.tolist() 
    uv = list(set(v)) # unique values of all features
    
    mv = {} # mapping to unique powers of 2
    for i,x in enumerate(uv):
        mv[x] = 2**i
    
    as_numbers = [reduce((lambda x, y: x | y), [mv[x + str(i)] for i, x in enumerate(row)]) for row in l]
    TO_BIN = "{0:0" + str(len(mv)) +"b}"
    flattened_features = [[int(char) for char in TO_BIN.format(number)] for number in as_numbers]
   
    return np.array(flattened_features)
a= toBinaryFeatures(training_set)
grzyby_zestaw_uczacy = np.append(a,b,axis=1)
#--- koniec przerabiania grzybow ---#
#--- pobieranie danych ---#
training_set = np.array([[0, 1, 0, 1, 0, 1],[0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 1, 0],[1, 0, 1, 0, 1, 0],[0, 1, 1, 1, 0, 1],[0, 1, 0, 0, 1, 1],[1, 1, 1, 0, 0, 0],[1, 1, 1, 1, 0, 1],[0, 1, 1, 0, 1, 0],[1, 1, 0, 0, 0, 1],[1, 0, 0, 0, 1, 0]])
test1 = genfromtxt('dane/test1.csv', delimiter=',')
dane1 = genfromtxt('dane/dane1.csv', delimiter=',')
udacity_set1 = np.array(
        [[1,1,1,0],
         [1,0,1,0],
         [0,1,0,1],
         [1,0,0,1]])
#--- koniec pobierania danych ---#



#--- liczenie entropii ---#
def H(X):
    if X.shape[0] == 0:
        return 0 #nie wiem jaka jest entropia zrobic jak macierz jest pusta
    fx = X[:,-1]
    zeros = len([j for j in fx if j == 0])
    ones = len([j for j in fx if j == 1])
    p0 = zeros/ len(fx)
    p1 = 1-p0
    if p0 == 0 or p0 == 1:
        return 0      
    return -p0*np.log2(p0)-p1*np.log2(p1)
   
def select_attribute(X,i,v):
    list = X[np.where(X[:,i] == v)]
    return list

def Q(X,i,v):
    return select_attribute(X,i,v).shape[0]/X.shape[0]
    
def IG(X,i): 
    return H(X)-Q(X,i,0)*H(select_attribute(X,i,0))-Q(X,i,1)*H(select_attribute(X,i,1))
    
def ID3(S, recursion = 1, tree = {}):
    result = np.array([])
    rows = S.shape[0]
    columns = S.shape[1]

    decision_column = S[:,columns-1]        
    if np.all(decision_column==0):
        tree[recursion] = int(0)
        return 
    if np.all(decision_column==1):
        tree[recursion] = int(1)
        return
    for i in range(columns-1):
        ig = IG(S,i)
        result = np.append(result, ig)
    j = result.argmax()
    leaf_label2 = str(int(j))
    tree[recursion] = leaf_label2 
    S0 = select_attribute(S,j,0)
    S1 = select_attribute(S,j,1)
    ID3(S0,recursion*2,tree)   
    ID3(S1,recursion*2+1,tree)
    return tree


def zwroc_wierzcholki(X):
    drzewo_lista = list(X.keys())
    wynik = {}
    for i in range(0, len(drzewo_lista)):
        index = drzewo_lista[i]
        if X[index] != 1 and X[index] != 0:
            wynik[index] = X[index]
    return sorted(list(wynik.keys()))

def poddrzewa(drzewo):
    wszystkie_drzewa = []
    wierzcholki = zwroc_wierzcholki(drzewo)
    drzewo_klucze = sorted(list(drzewo.keys()))
    for wierzcholek in wierzcholki[1:]:
        pojedyncze = {}
        for j in range(0,drzewo_klucze.index(wierzcholek)):
            wartosc = drzewo_klucze[j]
            pojedyncze[wartosc]=drzewo[wartosc]            
        wszystkie_drzewa.append(pojedyncze)
    wszystkie_drzewa.append(drzewo)
    return wszystkie_drzewa

def zapis2(drzewo):
    a = [None for i in range(max(drzewo.keys()))]
    for i in drzewo.keys():
        a[i-1]=drzewo[i]
    return a

def drzewa_zapisy(drzewo):
    return list(drzewo.items()), list(drzewo.keys()), list(drzewo.values()), zapis2(drzewo)

def poddrzewa2(drzewo):
    wszystkie = []
    chwilowe= []
    drzewo = zapis2(drzewo)
    for i in drzewo:        
        chwilowe.append(i)
        if type(i) == int:
            wszystkie.append(chwilowe.copy())
    return wszystkie


def blad_drugiego_rodzaju(drzewa, training_set):
    wynik = []
    iloscbledow=0
    for drzewo in drzewa:
        test_set = training_set.copy()

        for i in range(0,len(drzewo)-1):
            if i%2 == 1: #negatywne
                if type(drzewo[i]) == str:
                    atrybut = int(drzewo[i])                    
                    test_set = select_attribute(test_set,atrybut,1)
                              
                elif drzewo[i] == 1:
                    iloscbledow = len([j for j in test_set[:,-1] if j==1])  
                    print(test_set)
                    print(iloscbledow)
                elif drzewo[i] == 0:
                    iloscbledow = len([j for j in test_set[:,-1] if j==0])
                    print(test_set)
                    print(iloscbledow)
            else: #pozytywne
                if type(drzewo[i]) == str:
                    atrybut = int(drzewo[i])  
                    test_set = select_attribute(test_set,atrybut,0)
                                                    
                elif drzewo[i] == 1:
                    iloscbledow = len([j for j in test_set[:,-1] if j==1]) 
                    print(test_set)
                    print(iloscbledow)
                elif drzewo[i] == 0:
                    iloscbledow = len([j for j in test_set[:,-1] if j==0])
                    print(test_set)
                    print(iloscbledow)
                wynik.append(iloscbledow)
    return wynik




        






























































#def treetoarray(tree):
#    a=0
#    array = max(tree.keys())
#    treearray =  np.empty([2,len])
#    for i in tree.keys():        
#        treearray[1]
#        a = a+1
#
#

#
#def liczbalisci(drzewo):
#    z = list(drzewo.values()) 
#    f = z.count(1)
#    g = z.count(0)
#    return f+g
#
#def blad(drzewo,S):
#    
    
    
    
#def alfai(T,Ti,S):    
#    return blad(Ti,S)-blad(T,S)/liczbalisci(T)-liczbalisci(Ti)
#    
#def wybierzdrzewo(drzewa,S):
#    wyniki= []
#    glowne_drzewo = drzewa[len(drzewa)-1]
#    for drzewo in drzewa:
#        wyniki.append(alfai(glowne_drzewo,drzewo,S))
#    indexdrzewa= wyniki.index(max(wyniki))
#    return drzewa[indexdrzewa]
#        
#    
 
#from sklearn.feature_extraction import DictVectorizer
#dvec = DictVectorizer(sparse=False)
#
#X = dvec.fit_transform(mushroom.transpose().to_dict().values())
#
#data = pd.DataFrame({'0': ['u']})
#res = pd.get_dummies(mushroom)
#res.to_csv('output.csv')
#print(res)