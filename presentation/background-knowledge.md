# Background Knowledge – Pacman mit Reinforcement Learning

## 1. Überblick über das Projekt

### Was ist das Ziel des Projekts?

Dieses Projekt entwickelt einen künstlichen Agenten, der das klassische Arcade-Spiel Pacman selbstständig spielen lernt. Der Agent soll nicht durch vorprogrammierte Regeln gesteuert werden, sondern durch eigene Erfahrung lernen, welche Aktionen in welchen Situationen am besten sind.

Konkret bedeutet das: Der Agent startet ohne jegliches Wissen über das Spiel. Er weiß anfangs nicht, dass Dots Punkte bringen, dass Geister gefährlich sind oder dass Power-Pills die Geister fressbar machen. All das muss er durch wiederholtes Spielen und Auswerten von Belohnungen selbst herausfinden.

### Warum Pacman?

Pacman eignet sich hervorragend als Testumgebung für Reinforcement Learning aus mehreren Gründen:

- **Klare Regeln**: Das Spiel hat eindeutige Spielregeln und Ziele (alle Dots sammeln, Geistern ausweichen).
- **Übersichtlicher Zustandsraum**: Die Position von Pacman und den Geistern lässt sich klar erfassen.
- **Echtzeit-Entscheidungen**: Der Agent muss kontinuierlich Entscheidungen treffen, nicht nur einmal pro Runde.
- **Balance aus Einfachheit und Komplexität**: Das Spiel ist einfach genug, um es zu verstehen, aber komplex genug, um interessante Lernprobleme zu bieten.
- **Gute Dokumentation**: Als klassisches Spiel ist Pacman gut erforscht und dokumentiert.

### Was soll der Agent lernen?

Der Agent soll lernen:

- Dots effizient zu sammeln, um Punkte zu erhalten
- Geistern auszuweichen, um nicht zu sterben
- Power-Pills strategisch einzusetzen, um Geister zu fressen
- Das Level zu gewinnen, indem alle 244 Dots gesammelt werden

### Abgrenzung: Was ist das Projekt bewusst NICHT?

- **Keine perfekte Lösung**: Das Ziel ist nicht, Pacman perfekt zu spielen, sondern zu zeigen, dass ein Agent durch Erfahrung lernen kann.
- **Keine vorprogrammierte Strategie**: Der Agent erhält keine Regeln wie "weiche nach links aus, wenn ein Geist von rechts kommt".
- **Kein Supervised Learning**: Es gibt keinen Datensatz von "guten" Spielzügen, von dem der Agent lernt.
- **Kein komplettes Pacman-Spiel**: Das Projekt verwendet einen vereinfachten Modus mit reduzierter Geisteranzahl.

## 2. Grundidee: Lernen durch Erfahrung

### Was bedeutet "Reinforcement Learning"?

Reinforcement Learning (verstärkendes Lernen) ist ein Teilbereich des maschinellen Lernens. Die Grundidee ist einfach: Ein Agent lernt durch Versuch und Irrtum, welches Verhalten in einer Umgebung zu guten Ergebnissen führt.

**Eine Analogie**: Stell dir vor, ein Kind lernt Fahrradfahren. Niemand kann ihm genau erklären, welche Muskeln es wie bewegen muss. Stattdessen probiert es aus: Wenn es fällt (negative Erfahrung), merkt es sich, was es falsch gemacht hat. Wenn es erfolgreich fährt (positive Erfahrung), verstärkt es dieses Verhalten. Nach vielen Versuchen kann es Fahrradfahren – ohne dass je jemand die exakte Lösung programmiert hat.

Genau so funktioniert Reinforcement Learning:

1. Der Agent befindet sich in einem Zustand
2. Er wählt eine Aktion
3. Die Umgebung reagiert mit einem neuen Zustand und einer Belohnung (Reward)
4. Der Agent lernt aus dieser Erfahrung

### Unterschied zu klassischem Programmieren

Beim klassischen Programmieren gibt ein Entwickler dem Computer explizite Regeln vor:

- "Wenn ein Geist von rechts kommt, gehe nach links"
- "Wenn ein Dot vor dir ist, gehe geradeaus"

Das Problem: Der Entwickler muss alle möglichen Situationen vorhersehen und passende Regeln definieren. Bei komplexen Spielen ist das praktisch unmöglich.

Beim Reinforcement Learning definiert der Entwickler nur:

- Welche Aktionen möglich sind (hoch, runter, links, rechts)
- Welche Belohnungen es gibt (Punkte für Dots, Strafe für Tod)

Die Strategie, wie man spielt, entwickelt der Agent selbst.

### Unterschied zu Supervised Learning

Beim Supervised Learning (überwachtes Lernen) gibt es einen Datensatz mit Beispielen und korrekten Antworten:

- Input: Ein Bild einer Katze
- Korrekte Antwort: "Katze"

Der Algorithmus lernt, die richtigen Antworten zu den Inputs zu finden.

Beim Reinforcement Learning gibt es keine "korrekten Antworten". Der Agent weiß nicht im Voraus, welche Aktion die beste ist. Er erfährt nur im Nachhinein durch die Belohnung, ob seine Entscheidung gut war. Außerdem haben Entscheidungen oft verzögerte Konsequenzen: Eine Aktion, die jetzt harmlos erscheint, kann später zum Tod führen.

### Agent-Umwelt-Interaktion anschaulich erklärt

Die Interaktion zwischen Agent und Umgebung folgt einem einfachen Kreislauf:

1. **Zustand beobachten**: Der Agent "sieht" die aktuelle Spielsituation (wo ist Pacman? wo sind die Geister? wo sind die Dots?)

2. **Aktion wählen**: Basierend auf seinem aktuellen Wissen wählt der Agent eine Aktion (z.B. "nach oben gehen")

3. **Aktion ausführen**: Die Umgebung (das Spiel) führt die Aktion aus

4. **Feedback erhalten**: Der Agent erhält:
   - Den neuen Zustand (neue Positionen)
   - Eine Belohnung (z.B. +15 für einen Dot, -200 für Tod)

5. **Lernen**: Der Agent passt sein Wissen an, basierend auf der erhaltenen Belohnung

6. **Wiederholen**: Der Zyklus beginnt von vorne

Dieser Kreislauf wiederholt sich tausende Male während des Trainings. Mit jeder Wiederholung verbessert der Agent sein Verständnis davon, welche Aktionen in welchen Situationen gut sind.

## 3. Zentrale Bausteine von Reinforcement Learning

### Agent

Der **Agent** ist der Lernende – in unserem Fall der Pacman-Spieler. Er trifft Entscheidungen basierend auf seinem aktuellen Wissen und lernt aus den Konsequenzen.

**Im Projekt**: Der Agent ist ein Python-Programm, das entweder eine Q-Tabelle (bei Q-Learning) oder ein neuronales Netz (bei Deep Q-Learning) verwendet, um Entscheidungen zu treffen.

### Environment

Das **Environment** (Umgebung) ist alles, womit der Agent interagiert – das Spielfeld, die Regeln, die Geister.

**Im Projekt**: Das Environment ist das Pacman-Spiel, implementiert in der Programmiersprache C. Der Agent kommuniziert mit dem Spiel über eine HTTP-Schnittstelle.

### State

Der **State** (Zustand) beschreibt die aktuelle Situation im Spiel. Er enthält alle Informationen, die der Agent braucht, um eine Entscheidung zu treffen.

**Im Projekt** enthält der State:

- Position von Pacman
- Positionen und Zustände der Geister (gefährlich oder fliehend)
- Verbleibende Dots auf dem Spielfeld
- Anzahl der verbleibenden Leben
- Ob der Power-Modus aktiv ist

### Action

Eine **Action** (Aktion) ist eine Handlung, die der Agent ausführen kann.

**Im Projekt** gibt es vier mögliche Aktionen:

- Hoch (up)
- Runter (down)
- Links (left)
- Rechts (right)

### Reward

Der **Reward** (Belohnung) ist ein numerischer Wert, der dem Agenten mitteilt, wie gut oder schlecht seine letzte Aktion war.

**Im Projekt**:

- +15 für das Sammeln eines Dots
- +40 für das Sammeln einer Power-Pill
- +100 bis +1700 für das Fressen eines Geistes (steigt mit Kombos)
- -10 bis -200 für den Tod (abhängig davon, wie lange man überlebt hat)
- +2000 bis +5000 für das Gewinnen des Levels

### Episode

Eine **Episode** ist ein vollständiger Durchgang des Spiels – vom Start bis zum Game Over (oder bis zum Gewinn des Levels).

**Im Projekt**: Eine Episode umfasst alle drei Leben. Der Agent startet mit zwei Reserveleben, kann also insgesamt dreimal sterben, bevor die Episode endet.

### Policy

Die **Policy** (Strategie) ist die Entscheidungsregel des Agenten. Sie bestimmt, welche Aktion in welchem Zustand gewählt wird.

**Im Projekt**: Die Policy wird durch die Q-Werte bestimmt. Der Agent wählt (meistens) die Aktion mit dem höchsten Q-Wert für den aktuellen Zustand.

## 4. Q-Learning – das Basiskonzept

### Was ist Q-Learning?

Q-Learning ist ein klassischer Algorithmus aus dem Reinforcement Learning, entwickelt 1989 von Christopher Watkins. Das "Q" steht für "Quality" – die Qualität einer Aktion in einem bestimmten Zustand.

Die zentrale Idee: Für jede Kombination aus Zustand und Aktion wird ein Q-Wert gespeichert. Dieser Q-Wert schätzt, wie viel Belohnung der Agent erwarten kann, wenn er in diesem Zustand diese Aktion wählt und danach optimal weiterspielt.

### Was bedeutet die Q-Tabelle intuitiv?

Stell dir die Q-Tabelle als ein großes Nachschlagewerk vor:

- Jede Zeile entspricht einem möglichen Spielzustand
- Jede Spalte entspricht einer möglichen Aktion (hoch, runter, links, rechts)
- In jeder Zelle steht ein Wert: "Wie gut ist es, in diesem Zustand diese Aktion zu wählen?"

Beispiel:

| Zustand                        | Hoch | Runter | Links | Rechts |
|--------------------------------|------|--------|-------|--------|
| Geist oben, Dot rechts         | -50  | +10    | +5    | +20    |
| Keine Gefahr, Dots überall     | +15  | +15    | +15   | +15    |
| Geist links, Ausgang oben      | +30  | -10    | -80   | +5     |

Der Agent schaut in der Tabelle nach und wählt die Aktion mit dem höchsten Wert.

### Wie "merkt" sich der Agent gute Aktionen?

Der Lernprozess funktioniert so:

1. Der Agent ist in Zustand S und wählt Aktion A
2. Er erhält Belohnung R und landet in neuem Zustand S'
3. Er aktualisiert den Q-Wert für (S, A):
   - Der neue Wert berücksichtigt die erhaltene Belohnung
   - Und den geschätzten Wert des neuen Zustands
4. Über viele Wiederholungen nähern sich die Q-Werte den wahren Werten an

Die Aktualisierung folgt der Formel:

Q(S, A) ← Q(S, A) + α × (R + γ × max Q(S', a) - Q(S, A))

Dabei ist:

- α (Alpha): Die Lernrate – wie stark neue Informationen gewichtet werden
- γ (Gamma): Der Discount-Faktor – wie wichtig zukünftige Belohnungen sind
- max Q(S', a): Der beste Q-Wert im Folgezustand

### Vorteile und Grenzen von Q-Learning

**Vorteile**:

- Einfach zu verstehen und zu implementieren
- Konvergiert bei genügend Training zur optimalen Strategie
- Benötigt keine Kenntnis der Umgebungsdynamik

**Grenzen**:

- Speichert jeden Zustand einzeln – bei vielen Zuständen wird die Tabelle riesig
- Kann nicht generalisieren: Ähnliche Zustände werden komplett getrennt behandelt
- Bei kontinuierlichen Zuständen nicht direkt anwendbar

### Warum Q-Learning bei großen Zustandsräumen problematisch wird

Das fundamentale Problem: Die Anzahl möglicher Zustände wächst exponentiell.

In Pacman gibt es:

- 28 × 36 = 1008 mögliche Positionen für Pacman
- 1008 Positionen für jeden der 4 Geister
- 244 Dots, die jeweils da sein können oder nicht

Die Gesamtzahl möglicher Zustände wäre astronomisch groß. Eine Q-Tabelle für all diese Zustände würde nicht in den Speicher passen.

**Lösung im Projekt**: Die Zustände werden vereinfacht (diskretisiert). Das Spielfeld wird in 16 Zonen eingeteilt statt einzelne Positionen zu unterscheiden. Geisterabstände werden in grobe Kategorien eingeteilt (nah, mittel, fern). Dadurch wird die Tabelle handhabbar – aber Detailinformationen gehen verloren.

## 5. Deep Q-Learning – Erweiterung des Ansatzes

### Motivation: Warum reicht Q-Learning nicht aus?

Wie oben beschrieben, scheitert Q-Learning an großen Zustandsräumen. Selbst mit Vereinfachungen kann die Q-Tabelle im Projekt zehntausende Einträge haben. Und trotz der Vereinfachungen fehlt dem Agenten oft wichtige Detailinformation.

Eine weitere Schwäche: Q-Learning kann nicht generalisieren. Wenn der Agent lernt, dass in Zustand A die Aktion "links" gut ist, hilft ihm das nichts für den fast identischen Zustand B. Er muss jeden Zustand einzeln lernen.

### Idee: Neuronales Netz als Funktionsapproximator

Deep Q-Learning ersetzt die Q-Tabelle durch ein neuronales Netz. Statt die Q-Werte explizit zu speichern, lernt das Netz eine Funktion, die Zustände auf Q-Werte abbildet.

Das Netz erhält als Input eine Beschreibung des Zustands (Zahlen, die Position, Geisterabstände etc. kodieren) und gibt als Output die Q-Werte für alle möglichen Aktionen aus.

### Was lernt das Netz konkret?

Das neuronale Netz lernt Muster in den Eingabedaten zu erkennen:

- "Wenn der Geist in Richtung X ist, sind Aktionen in Richtung -X meist besser"
- "Wenn viele Dots in einer Richtung sind, ist diese Richtung wertvoll"
- "Wenn der Power-Modus aktiv ist, werden Aktionen zum Geist hin belohnt"

Diese Muster werden in den Gewichten des Netzes gespeichert. Das Netz muss nicht jeden einzelnen Zustand gesehen haben – es kann das Gelernte auf neue, ähnliche Zustände anwenden.

### Unterschied Q-Learning vs. Deep Q-Learning

| Aspekt                  | Q-Learning                      | Deep Q-Learning                    |
|-------------------------|--------------------------------|-----------------------------------|
| Speicherung             | Tabelle mit expliziten Werten  | Neuronales Netz mit Gewichten     |
| Zustandsdarstellung     | Vereinfachte, diskrete Werte   | Detaillierte Zahlenvektoren       |
| Generalisierung         | Keine – jeder Zustand einzeln  | Ja – ähnliche Zustände ähnlich    |
| Speicherbedarf          | Wächst mit Zustandsanzahl      | Konstant (Netzgröße)              |
| Trainingszeit pro Schritt | Schnell                       | Langsamer (Netz-Update)           |
| Komplexität             | Einfach                        | Komplexer                         |

### Warum Deep Q-Learning besser generalisieren kann

Generalisierung bedeutet: Das Gelernte auf neue, ungesehene Situationen anwenden.

Beispiel: Der Agent hat gelernt, dass er bei einem Geist 3 Felder rechts nach links ausweichen soll. Mit Q-Learning muss er dieselbe Lektion für "Geist 4 Felder rechts", "Geist 2 Felder rechts" usw. einzeln lernen.

Mit Deep Q-Learning erfasst das Netz das abstrakte Muster: "Geist rechts → links gut". Dieses Muster gilt für alle Varianten der Situation.

Das führt zu:

- Schnellerem Lernen (muss weniger Einzelfälle sehen)
- Robusterem Verhalten (funktioniert auch in neuen Situationen)
- Besserer Nutzung der Trainingsdaten

## 6. Pacman als Lernumgebung

### Warum eignet sich Pacman gut für RL?

Pacman ist ein ideales Testbed für Reinforcement Learning:

**Klare Belohnungsstruktur**: Dots sammeln = gut, sterben = schlecht. Der Agent bekommt eindeutiges Feedback.

**Deterministisches Spielfeld**: Das Labyrinth ändert sich nicht. Der Agent kann Muster lernen, die immer gelten.

**Überschaubarer Aktionsraum**: Nur vier mögliche Aktionen reduzieren die Komplexität.

**Sichtbarer Fortschritt**: Man kann direkt beobachten, ob der Agent besser wird (mehr Dots, längeres Überleben).

**Bekanntes Problem**: Pacman ist gut erforscht; es gibt Vergleichswerte und etablierte Ansätze.

### Welche Herausforderungen gibt es?

**Gegner (Geister)**: Die Geister verfolgen Pacman aktiv. Der Agent muss lernen, ihre Bewegungen zu antizipieren und auszuweichen. Das macht die Umgebung dynamisch und erfordert vorausschauendes Planen.

**Große Zustandsräume**: Obwohl Pacman einfach aussieht, ist die Anzahl möglicher Spielsituationen enorm. Die exakte Konstellation von Pacman, vier Geistern und 244 Dots erzeugt eine riesige Anzahl unterschiedlicher Zustände.

**Belohnungsdesign (Reward Shaping)**: Die richtige Gestaltung der Belohnungen ist kritisch. Falsche Anreize können zu unerwünschtem Verhalten führen:

- Zu hohe Strafe für Tod → Agent wird übervorsichtig und sammelt keine Dots
- Zu geringe Strafe für Tod → Agent lernt nicht, Geistern auszuweichen
- Keine Zwischenbelohnungen → Agent bekommt zu wenig Feedback zum Lernen

**Verzögerte Konsequenzen**: Manche Entscheidungen haben Auswirkungen erst viel später. Eine Sackgasse betreten ist erst problematisch, wenn ein Geist kommt – aber dann ist es zu spät.

### Welche Entscheidungen muss der Agent treffen?

Bei jedem Schritt muss der Agent entscheiden:

- **Richtung wählen**: Hoch, runter, links oder rechts?
- **Risiko abwägen**: Dot in der Nähe eines Geistes holen oder sicher spielen?
- **Power-Pills strategisch nutzen**: Jetzt nehmen oder für später aufsparen?
- **Escape-Routen beachten**: Immer einen Fluchtweg offenhalten

Im Projekt trifft der Agent Entscheidungen an "Junctions" – Kreuzungen, wo mehrere Richtungen möglich sind. In Gängen ohne Abzweigung gibt es keine echte Entscheidung zu treffen.

## 7. Reward-Design im Projekt

### Warum Rewards entscheidend sind

Rewards sind das einzige Signal, durch das der Agent lernt, was gut und was schlecht ist. Ein schlecht designtes Reward-System führt zu Agenten, die technisch ihr Ziel erfüllen, aber nicht das tun, was der Entwickler eigentlich wollte.

Ein klassisches Beispiel: Gibt man nur Belohnung fürs Überleben, lernt der Agent vielleicht, sich in einer Ecke zu verstecken – er überlebt lange, sammelt aber keine Punkte.

### Welche Belohnungen/Bestrafungen im Projekt verwendet wurden

**Positive Belohnungen**:

- +15 für jeden gesammelten Dot
- +40 für eine Power-Pill
- +100 bis +1700 für das Fressen eines Geistes (steigt bei Kombos)
- +100 Bonus bei 200+ gesammelten Dots (Meilenstein)
- +200 Bonus bei 220+ gesammelten Dots
- +500 Bonus bei 235+ gesammelten Dots
- +2000 bis +5000 für das Gewinnen des Levels

**Negative Belohnungen (Strafen)**:

- -0.2 bis -0.5 pro Zeitschritt (kleine Kosten fürs Warten)
- -1 bis -15 für Nähe zu gefährlichen Geistern (je näher, desto mehr)
- -10 bis -200 für den Tod (abhängig von der Überlebenszeit)

**Besonderheit – Survival-basierte Todesstrafe**:

Die Strafe für den Tod ist nicht fix, sondern hängt davon ab, wie lange der Agent überlebt hat:

- Schneller Tod (wenige Schritte) → hohe Strafe (-200)
- Langes Überleben vor dem Tod → geringe Strafe (-10)

Diese Gestaltung ermutigt den Agenten, länger zu überleben, statt ihn für jeden Tod gleich zu bestrafen.

### Wie Rewards das Verhalten des Agents steuern

Die Rewards formen das Verhalten direkt:

- **Dot-Belohnungen** motivieren zum Sammeln
- **Geister-Nähe-Strafen** lehren Ausweichen, bevor es zu spät ist
- **Meilenstein-Boni** motivieren, viele Dots zu sammeln statt früh aufzugeben
- **Step-Kosten** verhindern, dass der Agent einfach stillsteht
- **Win-Bonus** macht das Gewinnen zum wichtigsten Ziel

### Typische Fehler beim Reward-Design

**Zu sparse Rewards**: Nur Belohnung am Ende des Levels → Agent bekommt zu wenig Feedback, um zu lernen.

**Widersprüchliche Anreize**: Strafe für Geister-Nähe + Belohnung für Geister-Fressen → Agent weiß nicht, ob er zum Geist soll oder weg.

**Reward Hacking**: Agent findet Schlupflöcher. Beispiel: Belohnung für "neue Tiles besuchen" → Agent pendelt zwischen zwei Tiles hin und her.

**Zu starke Strafen**: Sehr hohe Todesstrafe → Agent wird passiv und riskiert nichts.

Im Projekt wurden diese Probleme durch iteratives Anpassen der Rewards gelöst. Die finale Reward-Struktur ist das Ergebnis vieler Experimente.

## 8. Technischer Aufbau des Projekts

### Überblick über die Projektarchitektur

Das Projekt besteht aus zwei Hauptkomponenten:

1. **Das Pacman-Spiel** (in C implementiert)
2. **Die Machine-Learning-Agenten** (in Python implementiert)

Diese kommunizieren über eine HTTP-Schnittstelle miteinander.

### Rolle von Python

Python ist die Sprache für die Machine-Learning-Komponenten:

- **pacman_env.py**: Definiert die Umgebung aus Sicht des Agenten
- **qlearn.py**: Implementiert den Q-Learning-Agenten mit Q-Tabelle
- **dqn_agent.py**: Implementiert den Deep Q-Learning-Agenten
- **dqn_model.py**: Definiert die Architektur des neuronalen Netzes
- **replay_buffer.py**: Speichert vergangene Erfahrungen für das Training
- **train.py**: Steuert den Trainingsprozess

Für Deep Q-Learning wird PyTorch verwendet, eine Bibliothek für neuronale Netze.

### Rolle von Mongoose

Mongoose ist eine eingebettete HTTP-Server-Bibliothek für C/C++. Sie ermöglicht:

- Das Pacman-Spiel startet einen lokalen Webserver
- Python-Agenten können HTTP-Anfragen an das Spiel senden
- Aktionen werden per POST-Request übermittelt
- Spielzustände werden als JSON zurückgegeben

Diese Architektur entkoppelt Spiel und Agent vollständig. Das Spiel muss nichts über Machine Learning wissen; es bietet einfach eine Schnittstelle für externe Steuerung.

### Wie die Komponenten miteinander kommunizieren

Der Ablauf einer Aktion:

1. Agent fordert aktuellen Zustand an → GET /api/state
2. Spiel antwortet mit JSON (Positionen, Punkte, etc.)
3. Agent analysiert Zustand und wählt Aktion
4. Agent sendet Aktion → POST /api/step mit {"direction": "left"}
5. Spiel führt Aktion aus und antwortet mit neuem Zustand
6. Agent berechnet Reward und lernt

Weitere wichtige Endpunkte:

- /api/restart → Startet das Spiel neu
- /api/start → Beginnt eine neue Runde

### Grober Ablauf eines Trainingsdurchlaufs (Step-by-Step)

1. **Initialisierung**
   - Spiel wird gestartet mit aktivierter API
   - Agent wird initialisiert (leere Q-Tabelle oder untrainiertes Netz)

2. **Episode starten**
   - Spiel wird zurückgesetzt
   - Agent erhält initialen Zustand

3. **Spiel-Schleife** (wiederholt sich hunderte Male pro Episode)
   - Agent beobachtet aktuellen Zustand
   - Agent wählt Aktion (anfangs oft zufällig, später basierend auf Wissen)
   - Aktion wird an Spiel gesendet
   - Neuer Zustand und Reward werden empfangen
   - Agent aktualisiert sein Wissen (Q-Tabelle oder Netz-Gewichte)

4. **Episode endet** (durch Tod oder Level-Gewinn)
   - Statistiken werden gespeichert
   - Exploration-Rate wird angepasst

5. **Nächste Episode** (zurück zu Schritt 2)

6. **Nach vielen Episoden**
   - Modell wird gespeichert
   - Training ist abgeschlossen

## 9. Trainingsprozess

### Was passiert während des Trainings?

Das Training besteht aus tausenden Episoden, in denen der Agent immer wieder Pacman spielt. Anfangs spielt er schlecht – er läuft ziellos umher und stirbt schnell. Mit der Zeit lernt er Muster:

- "Wenn ich in diese Richtung gehe und da ist ein Geist, sterbe ich oft"
- "Wenn ich Dots sammle, bekomme ich Belohnung"
- "An dieser Kreuzung nach links führt zu vielen Dots"

Diese Muster manifestieren sich in den Q-Werten (bei Q-Learning) bzw. den Netzwerk-Gewichten (bei Deep Q-Learning).

### Exploration vs. Exploitation (intuitiv erklärt)

Ein fundamentales Dilemma im Reinforcement Learning:

**Exploration** = Neues ausprobieren, um vielleicht bessere Strategien zu finden

**Exploitation** = Das nutzen, was man bereits als gut erkannt hat

**Analogie**: In einem Restaurant:

- Exploration: Immer neue Gerichte bestellen – man könnte etwas Besseres finden
- Exploitation: Immer das Lieblingsgericht bestellen – man weiß, dass es schmeckt

Zu viel Exploration: Der Agent probiert endlos Neues und nutzt sein Wissen nie.
Zu viel Exploitation: Der Agent bleibt bei der ersten halbwegs guten Strategie und findet nie die optimale.

**Lösung im Projekt – Epsilon-Greedy**:

Mit Wahrscheinlichkeit ε (Epsilon) wählt der Agent zufällig (Exploration).
Mit Wahrscheinlichkeit 1-ε wählt er die beste bekannte Aktion (Exploitation).

ε startet hoch (z.B. 1.0 = 100% zufällig) und sinkt über das Training (z.B. auf 0.05 = 5% zufällig). So erkundet der Agent anfangs viel und nutzt später sein Wissen.

### Warum Training Zeit braucht

Mehrere Faktoren machen das Training zeitaufwändig:

- **Viele Erfahrungen nötig**: Der Agent muss tausende Situationen erleben
- **Verzögertes Feedback**: Der Wert einer Aktion zeigt sich oft erst Schritte später
- **Statistisches Lernen**: Aus zufälligen Ereignissen muss ein Muster erkannt werden
- **Langsame Kommunikation**: HTTP-Anfragen zwischen Python und C kosten Zeit
- **Komplexe Berechnungen**: Besonders bei Deep Q-Learning sind Netz-Updates rechenintensiv

Typische Trainingszeiten im Projekt:

- Q-Learning: 2-3 Stunden für 5000 Episoden
- Deep Q-Learning: 4-6 Stunden für 2000 Episoden

### Was bedeutet "konvergieren" im Kontext des Projekts?

Konvergenz bedeutet, dass sich das Verhalten des Agenten stabilisiert. Am Anfang verbessert er sich schnell, dann langsamer, bis er ein Plateau erreicht.

Erkennbar an:

- Durchschnittliche Punktzahl steigt nicht mehr signifikant
- Epsilon hat seinen Minimalwert erreicht
- Q-Werte ändern sich nur noch minimal

Wichtig: Konvergenz bedeutet nicht perfektes Spiel. Der Agent erreicht ein lokales Optimum – das beste Verhalten, das er mit seinem Lernansatz finden konnte.

## 10. Ergebnisse & Interpretation

### Wie wurden Ergebnisse gemessen?

Die Leistung der Agenten wurde über mehrere Metriken erfasst:

- **Score**: Gesamtpunktzahl pro Episode (Dots, Geister, etc.)
- **Dots eaten**: Anzahl der gesammelten Dots (max. 244)
- **Deaths**: Anzahl der Tode pro Episode (max. 3)
- **Wins**: Episoden, in denen alle 244 Dots gesammelt wurden

Die Werte wurden über Episoden-Fenster gemittelt (z.B. die letzten 500 Episoden), um Trends sichtbar zu machen.

### Bedeutung von durchschnittliche Dots und Best Dots

**Durchschnittliche Dots** zeigen die typische Leistung des Agenten. Ein höherer Durchschnitt bedeutet konsistent gutes Spiel.

**Best Dots** zeigen das Potenzial des Agenten. Die maximale Dot-Anzahl einer einzelnen Episode zeigt, was unter günstigen Umständen möglich ist.

Beide Werte sind wichtig:

- Hoher Durchschnitt, niedriges Maximum → Konsistent, aber limitiert
- Niedriger Durchschnitt, hohes Maximum → Inkonsistent, aber fähig zu Spitzenleistung

### Ergebnisse im Projekt

**Q-Learning (nach 5000 Episoden)**:

- Durchschnittlicher Score: 1238.3
- Durchschnittliche Dots: 102.0
- Bester Score: 3500
- Beste Dots: 218
- Episoden mit 200+ Dots: 7

**Deep Q-Learning (nach 2000 Episoden)**:

- Durchschnittlicher Score: 1981.7
- Durchschnittliche Dots: 163.5
- Bester Score: 4490
- Beste Dots: 244 (Level gewonnen)
- Gewonnene Runden: 48 (2.4%)

### Warum Deep Q-Learning besser abschneidet als Q-Learning

Die Ergebnisse zeigen klar: Deep Q-Learning übertrifft Q-Learning deutlich:

- 60% höherer durchschnittlicher Dot-Wert
- Fast doppelt so hoher durchschnittlicher Score
- Regelmäßige Level-Gewinne (bei Q-Learning sehr selten)

Gründe für die Überlegenheit:

1. **Bessere Zustandsdarstellung**: Das neuronale Netz erhält detailliertere Informationen (exakte Positionen statt Zonen).

2. **Generalisierung**: Ähnliche Situationen werden ähnlich behandelt. Gelerntes überträgt sich auf verwandte Zustände.

3. **Effizientere Nutzung der Erfahrungen**: Experience Replay nutzt jede Erfahrung mehrfach für das Training.

4. **Stabileres Lernen**: Target Networks und Double DQN verhindern oszillierende Q-Werte.

### Was man aus den Ergebnissen lernen kann

- **RL funktioniert**: Beide Agenten lernen ohne vorgegebene Regeln, Pacman zu spielen
- **Architektur zählt**: Die Wahl des Algorithmus hat großen Einfluss auf die Leistung
- **Reward-Design ist kritisch**: Die sorgfältig designten Belohnungen ermöglichen effektives Lernen
- **Training braucht Zeit**: Tausende Episoden sind nötig für gute Ergebnisse
- **Perfektes Spiel ist schwer**: Selbst der beste Agent gewinnt nur in 2.4% der Fälle

## 11. Typische Fragen & Antworten (Q&A)

### Warum nicht einfach Regeln programmieren?

Man könnte Pacman-Regeln von Hand programmieren ("wenn Geist links, gehe rechts"). Das hätte Vorteile: schneller, deterministisch, verständlich. Aber es gibt Gründe für den RL-Ansatz:

- **Komplexität**: Alle Situationen manuell abzudecken ist praktisch unmöglich
- **Flexibilität**: RL-Agenten können sich an Änderungen anpassen
- **Forschungszweck**: Das Projekt dient dem Erlernen von RL-Techniken
- **Skalierbarkeit**: Der Ansatz lässt sich auf komplexere Spiele übertragen

### Lernt der Agent "wirklich"?

Das hängt von der Definition von "Lernen" ab. Der Agent:

- Passt sein Verhalten basierend auf Erfahrung an
- Verbessert messbar seine Leistung über die Zeit
- Generalisiert (bei DQN) auf neue Situationen

Was der Agent nicht tut:

- "Verstehen" im menschlichen Sinn
- Bewusstes Strategisieren
- Kreatives Problemlösen außerhalb des trainierten Bereichs

Der Agent lernt statistische Muster in seinen Daten – nicht mehr, nicht weniger.

### Kann das System auf andere Spiele übertragen werden?

Grundsätzlich ja, mit Anpassungen:

**Was übertragbar ist**:

- Die grundlegende RL-Architektur
- Q-Learning und Deep Q-Learning Algorithmen
- Experience Replay, Target Networks, etc.

**Was angepasst werden muss**:

- Zustandsdarstellung (jedes Spiel hat andere relevante Informationen)
- Aktionsraum (andere Spiele haben andere mögliche Aktionen)
- Reward-Design (andere Ziele erfordern andere Belohnungen)
- Hyperparameter (Lernrate, Epsilon-Decay, etc.)

DeepMind hat mit ähnlichen Techniken 49 Atari-Spiele gelernt – die Methoden sind generell anwendbar.

### Was passiert, wenn sich die Umgebung ändert?

Das hängt von der Art der Änderung ab:

**Kleine Änderungen** (z.B. leicht andere Geister-Geschwindigkeit):

- Der Agent kann sich eventuell adaptieren
- Performance sinkt kurzfristig, erholt sich dann

**Große Änderungen** (z.B. komplett neues Labyrinth):

- Der Agent muss neu trainiert werden
- Vorwissen hilft wenig bis gar nicht

**Transfer Learning** könnte helfen: Ein vortrainierter Agent als Startpunkt für neue Umgebungen. Das wurde in diesem Projekt jedoch nicht implementiert.

### Welche Grenzen hat der Ansatz?

**Praktische Grenzen**:

- Lange Trainingszeiten
- Viel Trial-and-Error beim Hyperparameter-Tuning
- Deterministisches Geisterverhalten vereinfacht das Problem
- Nur ein Geist im Training (Mehr-Geister-Szenarien sind schwieriger)

**Konzeptionelle Grenzen**:

- Keine Übertragung auf andere Spiele ohne Neutraining
- Kein "Verständnis" des Spiels, nur Mustererkennung
- Suboptimale Lösungen (findet nicht unbedingt die beste Strategie)
- Reward-Hacking möglich (Agent findet Schlupflöcher)

**Skalierungsgrenzen**:

- Bei komplexeren Spielen werden noch mehr Erfahrungen benötigt
- Höherdimensionale Zustandsräume erfordern größere Netze
- Ressourcenbedarf steigt erheblich

## 12. Zusammenfassung

### Kernaussagen des Projekts

1. **Reinforcement Learning ermöglicht lernende Spielagenten**: Ohne explizite Programmierung von Regeln lernt der Agent durch Erfahrung, Pacman zu spielen.

2. **Q-Learning funktioniert für vereinfachte Probleme**: Mit diskretisiertem Zustandsraum und Junction-basierten Entscheidungen erreicht tabellenbasiertes Q-Learning solide Ergebnisse.

3. **Deep Q-Learning skaliert besser**: Das neuronale Netz kann mit detaillierteren Zuständen arbeiten und generalisiert besser, was zu deutlich höherer Leistung führt.

4. **Reward-Design ist entscheidend**: Die sorgfältige Gestaltung der Belohnungen – von Dot-Boni über Survival-basierte Todesstrafen bis zu Meilenstein-Boni – ist zentral für den Lernerfolg.

5. **Die Architektur ermöglicht saubere Trennung**: Durch die HTTP-API kann das C-Spiel unabhängig von den Python-Agenten entwickelt und getestet werden.

### Wichtigste Learnings

- **Exploration vs. Exploitation**: Der richtige Balanceakt ist essenziell. Epsilon-Greedy mit Decay funktioniert gut in der Praxis.

- **State-Repräsentation beeinflusst alles**: Was der Agent "sieht", bestimmt, was er lernen kann. Zu wenig Information → suboptimales Lernen. Zu viel → langsames Lernen.

- **Stabilität beim Training**: Techniken wie Target Networks, Experience Replay und Gradient Clipping verhindern instabiles Lernen.

- **Iteratives Vorgehen**: Die finale Reward-Struktur und Hyperparameter entstanden durch viele Experimente, nicht durch einmalige Festlegung.

- **Geduld ist nötig**: Tausende Episoden über Stunden sind nötig. Frühe Ergebnisse sagen wenig über finale Leistung aus.

### Warum das Projekt ein gutes Beispiel für Reinforcement Learning ist

**Didaktisch wertvoll**:

- Pacman ist verständlich und anschaulich
- Fortschritt ist sichtbar (mehr Dots = besser)
- Alle RL-Grundkonzepte kommen vor

**Technisch relevant**:

- Q-Learning als Basisalgorithmus
- Deep Q-Learning als moderner Ansatz
- Praktische Herausforderungen (Zustandsraum, Reward-Design, Training)

**Realitätsnah**:

- Echtzeitentscheidungen erforderlich
- Gegner verhalten sich adversarial
- Langfristige Konsequenzen von Entscheidungen

Das Projekt demonstriert eindrucksvoll, wie ein Agent ohne Vorwissen lernen kann, ein Spiel zu spielen – allein durch die Erfahrung von Erfolg und Misserfolg.