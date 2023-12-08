# Masc_Detection# Maskenerkennung mit YOLOv5

## Übersicht über die Datensätze

In dieser Arbeit wurden zwei Kaggle-Datensätze zur Maskenerkennung verwendet:

1. **Face Mask Detection Dataset:**
   - [Face Mask Detection Dataset](https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset)
   - JSON-Annotationen

2. **Kaggle Face Mask Detection Full:**
   - [Kaggle Face Mask Detection Full](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
   - Erweitert durch Augmentation für schwierige Lichtverhältnisse und nächtliche Bedingungen
   - XML-Annotationen


## Klassen im nicht-dunklen Datensatz (20 Klassen):
| Klassenindex | Klasse              	|
|--------------|-------------------------|
| 0        	| face_no_mask        	|
| 1        	| face_with_mask      	|
| 2        	| mask_surgical       	|
| 3        	| hat                 	|
| 4        	| eyeglasses          	|
| 5        	| face_other_covering 	|
| 6        	| face_with_mask_incorrect|
| 7        	| mask_colorful       	|
| 8        	| helmet              	|
| 9        	| sunglasses          	|
| 10       	| scarf_bandana       	|
| 11       	| hair_net            	|
| 12       	| goggles             	|
| 13       	| face_shield         	|
| 14       	| hijab_niqab         	|
| 15       	| turban              	|
| 16       	| balaclava_ski_mask  	|
| 17       	| gas_mask            	|
| 18       	| hood                	|
| 19       	| other               	|

## Klassen im dunklen Datensatz (3 Klassen):
| Klassenindex | Klasse              	|
|--------------|-------------------------|
| 0        	| without_mask        	|
| 1        	| with_mask           	|
| 2        	| mask_weared_incorrect   |


# Struktur der zu generierenden Datensätze für YOLO-Training

Die Datenvorbereitung umfasst die Konvertierung und Aufbereitung der Datensätze für das YOLO-Training. Die generierten Datensätze werden im Verzeichnis "YOLO/dataset" strukturiert abgelegt und umfassen folgende Untergruppen:

1. **all_classes:** Enthält alle Klassen und wird beim Training auf dem nicht-dunklen Datensatz verwendet.
2. **face_classes:** Berücksichtigt nur die Gesichtsklassen von den insgesamt 20 Klassen während des Trainings auf dem nicht-dunklen Datensatz.
3. **single_person:** Bezieht sich ausschließlich auf die Gesichtsklassen und einzelne Personen beim Training auf dem nicht-dunklen Datensatz.
4. **dark_dataset:** Umfasst alle Klassen und wird beim Training auf dem dunklen Datensatz verwendet.


# Konvertierung ins YOLO-Format
Für das YOLO-Training müssen Annotationen ins YOLO-Format (txt) konvertiert werden. Die Konvertierung von JSON-Daten folgt diesen Regeln:

- Erste Spalte: Klassenindizes
- Zweite bis fünfte Spalte: Koordinaten der Bounding Box (x, y, w, h), diese müssen normalisiert werden.


### Erstellung des All_Classes-Datensatzes für das Training

In diesem Abschnitt werden alle 20 Klassen für das Training berücksichtigt. Der Datensatz wird für das YOLO-Training in 80% Trainings- und 20% Validierungsdatensatz aufgeteilt. Die Annotationen im Datensatz sind im JSON-Format gespeichert und müssen für das YOLO-Training noch ins YOLO-Format umgewandelt werden.

### Erstellung des Face_Classes-Datensatzes für das Training

In diesem Abschnitt werden nur die 4 Face-Klassen (siehe unten) aus dem nicht-dunklen Datensatz betrachtet. Hierfür werden nicht relevante Bilder und Labels im Datensatz identifiziert und gelöscht. Anschließend erfolgt die Aufteilung der Daten in 80% Training und 20% Validierungsdatensatz.

| Klassenindex | Klasse              	|
|--------------|-------------------------|
| 0        	| face_no_mask        	|
| 1        	| face_with_mask      	|
| 2        	| face_other_covering 	|
| 3        	| face_with_mask_incorrect|

### Einzel-Personen-Datensatz

In diesem Datensatz wird ebenfalls der nicht-dunkle Datensatz mit den JSON-Annotationen verarbeitet. Der Fokus liegt hier ausschließlich auf einzelnen Gesichtern. Dabei werden nur die 4 Face_Classes berücksichtigt. Es ist bekannt, dass in einer TXT-Datei mehrere Face_Classes darauf hinweisen, dass mehrere Personen erkannt wurden. Solche TXT-Dateien und die dazugehörigen Bilder werden dann in den Ordner "single_person" kopiert.


### Dunkler Datensatz

Der dunkle Datensatz wurde durch Augmentation verdunkelt, da in Kaggle keine dunklen Datensätze zu finden sind. Das Ziel besteht darin, ein Modell zu trainieren, das auch im Dunkeln Masken erkennen kann. In diesem Datensatz werden alle Klassen berücksichtigt, da es 3 relevante Klassen gibt. Es ist wichtig zu beachten, dass die Annotationen für diesen Datensatz als XML-Dateien vorliegen und für das YOLO-Training in das YOLO-Format konvertiert werden müssen.

### Installation von YOLOv5 

Die Installation von YOLOv5 erfolgt mittels Git und pip. Es wurden spezifische Paketversionen für eine effiziente GPU-Unterstützung verwendet. Hier sind die Schritte zur Installation:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -U -r requirements.txt
```

### WandB Integration in YOLOv5

Die Integration von WandB (Weights and Biases) ermöglicht die effiziente Überwachung und Analyse des YOLOv5-Modelltrainings. WandB bietet Experimentenverfolgung, Visualisierung und Modellvergleiche. In diesem Projekt wird WandB genutzt, um Metriken wie Recall, Precision und mAP während des Trainings zu erfassen.

## Zusammenfassung der Trainingsergebnisse

Hier sind die Ergebnisse der YOLOv5-Trainingsläufe für verschiedene Datensätze in einer übersichtlichen Tabelle:

| Experiment      | Datensatz         | Recall | Precision | mAP_0.5 | mAP_0.95 |
|------------------|-------------------|--------|-----------|---------|----------|
| All_Classes      | Nicht-dunkel      | 0.44   | 0.71      | 0.46    | 0.29     |
| Face_Classes     | Nicht-dunkel      | 0.60   | 0.89      | 0.67    | 0.49     |
| Single_Person    | Nicht-dunkel      | 0.65   | 0.69      | 0.68    | 0.50     |
| Dark_Dataset     | Dunkel            | 0.55   | 0.63      | 0.56    | 0.33     |

Die detaillierte Analyse der Ergebnisse bietet Einblicke in die Leistung des YOLOv5-Modells bei unterschiedlichen Trainingsansätzen und Datensätzen.

### All_Classes
Der All_Classes-Datensatz, der alle 20 Klassen umfasst, zeigt die geringste Leistung. Die Herausforderung hier liegt wahrscheinlich in der Datenmenge für jede Klasse, was zu einer ungleichmäßigen Lernkurve führt. Die Klassen könnten aufgrund der Größe des Datensatzes und der Anzahl der Bilder pro Klasse unzureichend repräsentiert sein.

### Face_Classes
Das Modell, das sich auf die 4 Face_Classes konzentriert, zeigt erhebliche Verbesserungen. Hier spielt der Fokus auf spezifische Gesichtsmerkmale und Maskentypen eine entscheidende Rolle. Die Generalisierung ist besser, da das Modell auf die relevanten Merkmale gezielt trainiert wird.

### Single_Person
Die Betrachtung einzelner Personen führt zu weiteren Verbesserungen. Der Fokus auf Einzelpersonen reduziert die Komplexität der Aufgabe, da Überlappungen mit anderen Klassen vermieden werden. Dies trägt zu höheren Recall-, Precision- und mAP-Werten bei.

### Dark_Dataset
Das Training auf einem verdunkelten Datensatz birgt Herausforderungen. Die vermehrte Identifikation von Objekten als "background" deutet darauf hin, dass die Dunkelheit des Datensatzes zu Unsicherheiten führt. Eine mögliche Lösung könnte in einer präziseren Verdunkelung oder speziellen Anpassungen für schwierige Lichtverhältnisse liegen.

Die Erkenntnisse verdeutlichen, dass eine gezielte Datenvorbereitung und Auswahl der Trainingsklassen entscheidend sind. Das YOLOv5-Modell profitiert von ausreichenden Datenmengen pro Klasse und spezifischen Fokusgebieten, um eine optimale Leistung zu erzielen.

Für detailliertere Ergebnisse, umfassendere Analysen und Einblicke in den Trainingsprozess, empfehle ich einen Blick in das beigefügte Jupyter Notebook zu werfen. Das Notebook bietet eine Schritt-für-Schritt-Dokumentation des Trainingsprozesses, der Datenvorbereitung, der Modellevaluation und beinhaltet visuelle Darstellungen zur Veranschaulichung der Ergebnisse.

Das [Jupyter Notebook](YOLO/Datenvorbereitung_YOLOv5.ipynb) steht zur Verfügung und bietet die Möglichkeit, die einzelnen Schritte nachzuvollziehen. 





