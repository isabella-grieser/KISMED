# Repository der Gruppe "Beats by Dr.DeepAI"
Code zum Projektseminar "Wettbewerb künstliche Intelligenz in der Medizin" 
SoSe 2022 von der Gruppe "Beats by Dr.DeepAI". 
Mitglieder der Gruppe waren Abhishek Deshmukh, Sebastian Perle und Isabella Nunes Grieser.

Für das Projektseminar wurden zwei Modelle erstellt. Das erste Modell ist ein RandomForestClassifier, welches mit
Features, die von dem Signal extrahiert werden, prognostiziert, ob ein Signal zu der Klasse "Atrial Fibrillation" ist
oder von einem normalen Herzen gemessen wurde.
Das zweite Modell ist ein Convolutional Neural Network, welches ein Spektogramm vom Modell als Eingabe erhält und
anschließend entscheidet, ob das Signal des Typs "Normal Heartbeat", "Atrial Fibrillation", "Other Signal Type" 
oder "Noise" ist.



## Projektstruktur

Das Projekt ist in mehrere Ordner unterteilt. Die wichtigsten Ordner werden im folgenden genannt:

#### data
Der Datenordner. Dieser ist in mehreren Unterordnern unterteilt:
- **training**: Hier liegen die uns ursprünglich gegebenen Daten.
- **others**: Hier liegen die von uns gefundenen Daten:
    - **CINC**: Hier liegen die Daten von ...
    - **CPSC**: Hier liegen die Daten von ...
    - **CU_SPH**: Hier liegen die Daten von ...

#### model_weights
Hier sind alle trainierten Modelle gespeichert, die von uns erstellt wurden. Jedes Modell hat 
seinen eigenen Unterordner.

#### models
Hier sind die Implementierungen aller Modelle, die von uns getestet wurden. Modelle, die bei der finalen Abgabe nicht 
berücksichtigt wurden, wurden mit *DEPRECEATED* angegeben

#### Notebooks
Mehrere Notebooks, die von uns in dem Modellbildungsprozess erstellt wurden.

#### preprocessing
Erhält alle Preprocessing-Methoden. Wichtige Dateien sind:
- **augmentation.py**: Erhält alle Augmentierungsmethoden, die von uns benutzt wurden. 
- **features.py**: Erhält alle Methoden zur Feature-Extrahierung.
- **padding.py**: Erhält alle Methoden, die für die Normalisierung der Signale auf dieselbe Länge benutzt wurden.
- **preprocessing.py**: Erhält alle sonstigen Methoden, die für das Preprocessing benutzt wurden.

#### Sonstige Dateien

Weitere wichtige Dateien sind:

- **config.py**: Erhält alle Konfigurationsparameter für das Trainieren/die Benutzung des FreqCNNModels.
- **noisesidequest.py**: Skript für die Analyse der Rauschen-Nebenaufgabe.
- **train.py**: Skript für das Trainieren der beiden finalen Modelle.