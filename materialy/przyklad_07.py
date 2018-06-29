
from sklearn.datasets import load_iris
from sklearn import tree

#wczytanie zbioru danych
iris = load_iris()

#drzewo klasyfikacyjne z domyslnymi wartosciami parametrow
clf = tree.DecisionTreeClassifier()

#uczenie drzewa klasyfikacyjnego
clf = clf.fit(iris.data, iris.target)

#testowanie - klasyfikacja elementow zbioru uczacego
y=clf.predict(iris.data)
print 'Uzyskane w toku testowania etykiety:'
print y

#wygenerowanie pliku pdf reprezentujacego drzewo
#wymaga pakietu pydot oraz oprogramowania Graphviz
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("drzewo_iris.pdf")

