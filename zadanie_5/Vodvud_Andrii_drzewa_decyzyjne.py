
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split

#wczytanie zbioru danych
iris = load_iris()

X = iris.data
y = iris.target
train, test, train_targets, test_targets = train_test_split(X, y,
                                 test_size=0.50, random_state=42)


#drzewo klasyfikacyjne z domyslnymi wartosciami parametrow
clf = tree.DecisionTreeClassifier()

#uczenie drzewa klasyfikacyjnego
clf = clf.fit(train, train_targets)

#testowanie - klasyfikacja elementow zbioru uczacego
y=clf.predict(test)
print 'Uzyskane w toku testowania etykiety:'
print y

#wygenerowanie pliku pdf reprezentujacego drzewo
#wymaga pakietu pydot oraz oprogramowania Graphviz
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("drzewo_iris_klasyfikujace.pdf")

print 'wartosci parametrow drzewa klasyfikacyjnego'
print dot_data.getvalue()

#uzyskana sprawnosc
score_gini = clf.score(test, test_targets)
print 'Sprawnosc na zbiorze testowym gini'
print score_gini

#obliczamy liczbe niepoprawnych zaklasyfikowan i procentowa sprawnosc na zbiorze uczacym
print 'Uzyskana sprawnosc na zbiorze testowym:'
niepopr_zaklas=(test_targets!=y).sum()
print 'Niepoprawnych zaklasyfikowan:'
print niepopr_zaklas
CR=float(niepopr_zaklas)/len(y)
print CR


print 'obliczamy liczbe poprawnych zaklasyfikowan i procentowa sprawnosc na zbiorze uczacym'
clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(train, train_targets)

from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("drzewo_iris_pruning_2.pdf")

#uzyskana sprawnosc
score = clf.score(test, test_targets)
print 'Sprawnosc na zbiorze testowym do 2'
print score

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(train, train_targets)

from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("drzewo_iris_pruning_3.pdf")

#uzyskana sprawnosc
score = clf.score(test, test_targets)
print 'Sprawnosc na zbiorze testowym do 3'
print score


## criterion='entropy'
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(train, train_targets)
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("drzewo_iris_entropy.pdf")

score_entropy = clf.score(test, test_targets)
print 'Sprawnosc na zbiorze testowym entropy'
print score_entropy

print "roznica poprawnosci zaklasyfikowania drzewa uzyskanego z kryteriem entropy oraz gini"
print score_entropy-score_gini