# Katzen und Hunde klassifizieren
## Modell trainieren
- Öffne https://teachablemachine.withgoogle.com/train/image
- Benenne `Class 1` und `Class 2` in `Katze` und `Hund` (o.ä.) um.
- Lade die Trainingsdaten aus `catsvsdogs/train` hoch. Klicke dazu bei der jeweiligen Klasse auf `Hochladen` und ziehe den jeweiligen Unterordner hinein.
- Trainiere dein Modell. Dies dauert ca. eine Minute.

## Modell testen
- Wechsle im rechten Fenster auf `Datei` und ziehe ein Bild aus `catsvsdogs/test` hinein.
- Unten wird nun angegeben, mit welcher Wahrscheinlichkeit das Modell das Bild in die jeweilige Klasse eingeordnet hat.

## Aufgabe 1: Richtige und falsche Klassifizierungen
- Teste das Modell auf einigen Katzen und Hunden.
- Finde ein Bild, welches falsch klassifiziert wird.
- Wieso wird es falsch klassifiziert?
- Was macht dieses Bild besonders schwierig zu klassifizieren?

## Aufgabe 2: Testdaten aus unbekannten Klassen
### 2.1 ohne Webcam
- Lade ein Bild von einem anderen Tier hoch. Du kannst dazu entweder selber ein Bild aus dem Internet nehmen oder eines aus dem Ordner `catsvsdogs/test/other`. Wird es als Hund oder als Katze klassifizert?
- Beides wäre offensichtlich falsch. Was könnte man am Training oder am Datensatz ändern, um solche Fehlklassifizierungen zu vermeiden?
### 2.2 mit Webcam
- Wechsle zur Webcam.
- Wirst du als Hund oder als Katze klassifiziert? Schaffst du es, durch z.B. Kopfbewegungen o.ä. das Netzwerk umzustimmen?
- Beides ist offensichtlich falsch. Was könnte man am Training oder am Datensatz ändern, um solche Fehlklassifizierungen zu vermeiden?

## Aufgabe 3: Einfluss der Trainingsdaten
- Verändere die Trainingsdaten und trainiere das Netzwerk neu.
- Was passiert z.B. wenn du von einer Klasse sehr viel mehr Trainingsbilder verwendest als von der anderen?
- Was passiert wenn du von beiden Klassen nur sehr wenige Bilder verwendest?
- Wie sind deine Beobachtungen zu erklären?
