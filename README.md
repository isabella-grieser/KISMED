# Repository der Gruppe "Beats by Dr.DeepAI"
Code zum Projektseminar "Wettbewerb künstliche Intelligenz in der Medizin" 
SoSe 2022 von der Gruppe "Beats by Dr.DeepAI". 
Mitglieder der Gruppe waren Abhishek Deshmukh, Sebastian Perle und Isabella Nunes Grieser.

In diesem Wettbewerb, welcher im Rahmen eines Projektseminars stattgefunden hat, 
war die Aufgabe abnormale Verläufe in EKG-Messungen zu detektieren und diese dementsprechend zu klassifizieren.
Hierbei wurde der Fokus auf die Detektion von Vorhofkammerflimmern (engl. Atrial Fibrillation) in einem EKG gelegt.
Daraus entstanden zwei Klassifizierungsprobleme die bearbeitet werden konnten.

Das zwei Klassen Problem, also die Unterscheidung zwischen einem EKG mit Atrial Fibrillation Charakteristik und einem gesunden,
hat dabei die Hauptaufgabe dargestellt. Als Nebenaufgabe konnte man zusätzlich das vier Klassen Problem lösen,
bei dem zu den bereits genannten Kategorien, die Kategorien zu stark verrauscht ("Noise") oder anders abnormales EKG ("Other Signal Type") detektiert werden sollten.

Um diese Aufgaben zu lösen haben wir zwei Modelle erstellt. Das erste Modell ist ein RandomForestClassifier,
welcher Features aus dem EKG extrahiert und diese nutzt um ein EKG mit Atrial Fibrillation Charakteristik von einem gesunden zu unterscheiden.
Das zweite Modell ist ein Convolutional Neural Network, welches das aufgezeichnete Signal des EKGs in Form eines Spektrogramms untersucht und
anschließend entscheidet, ob das Signal des Typs "Normal Heartbeat", "Atrial Fibrillation", "Other Signal Type" oder "Noise" ist.

#TODO: noch die Besonderheiten erklären



### Projektstruktur

Das Projekt ist in mehrere Ordner unterteilt. Die wichtigsten Ordner werden im folgenden genannt:

#### data
Der Datenordner besteht aus mehreren Unterordnern, die gleichzeitig auch verschiedene Mess-Quellen von EKGs darstellen.
Die Unterordner sind folgendermaßen unterteilt:
- **training**: Hier liegen die uns ursprünglich gegebenen Daten.
- **others**: Hier liegen die von uns gefundenen Daten, die auf das für das zum Lösen der Aufgabe wichtige reduziert wurden:
    - **CINC**: Hier liegen die Daten der PhysioNet/CinC 2017 Challenge (Teilweise Daten aus "training")
    - **CPSC**: Hier liegen die Daten der China Physiological Signal Challenge 2018 
    - **CU_SPH**: Hier liegen die Daten der Chapman University und des Shaoxing People’s Hospital
> Bevor die Daten verwendet werden können, müssen zuerst die zip-Files extrahiert werden.

#### explain
Hier sind die Hilfsfunktionen für die "Explainable AI" Nebenaufgabe. Die zwei Dateien sind:
- **explanationtexts.py**: Enthält Methode zur Generierung der Beschreibungstexte.
- **plots.py**: Enthält Methode zur Generierung der Visualisierung der Features.

#### model_weights
Hier sind alle trainierten Modelle gespeichert, die von uns erstellt wurden. Jedes Modell hat 
seinen eigenen Unterordner.

#### models
Hier sind die Implementierungen aller Modelle, die von uns getestet wurden. Modelle, die bei der finalen Abgabe nicht 
berücksichtigt wurden, wurden mit *DEPRECEATED* angegeben

#### Notebooks
Mehrere Notebooks, die von uns in dem Modellbildungsprozess erstellt wurden.

#### preprocessing
Enthält alle Preprocessing-Methoden. Wichtige Dateien sind:
- **augmentation.py**: Enthält alle Augmentierungsmethoden, die von uns benutzt wurden. 
- **features.py**: Enthält alle Methoden zur Feature-Extrahierung.
- **padding.py**: Enthält alle Methoden, die für die Normalisierung der Signale auf dieselbe Länge benutzt wurden.
- **preprocessing.py**: Enthält alle sonstigen Methoden, die für das Preprocessing benutzt wurden.

#### utils
Enthält sonstige Utility Methoden. Wichtige Dateien sind:
- **crossvalidation.py**: Enthält die Methode zur Crossvalidierung des FreqCNNModels
- **plotutils.py**: Enthält Methoden zum Plotten von Signalen

#### Sonstige Dateien

Weitere wichtige Dateien sind:

- **config.py**: Enthält alle Konfigurationsparameter für das Trainieren/die Benutzung des FreqCNNModels.
- **noisesidequest.py**: Skript für die Analyse der Rauschen-Nebenaufgabe.
- **train.py**: Skript für das Trainieren der beiden finalen Modelle.