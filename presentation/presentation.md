# Pacman Reinforcement Learning - Praesentationsskript

**Hochschule fuer Technik Stuttgart**  
**Modul: Machine Learning und Data Mining**  
**Wintersemester 2025/26**

---

## Sprecheraufteilung

| Folie | Sprecher | Dauer |
|-------|----------|-------|
| 1 - Titel | Paul Durz | 20 Sek. |
| 2 - Aufgabenstellung | Paul Durz | 45 Sek. |
| 3 - Loesungsansatz | Manuel Holm | 50 Sek. |
| 4 - Architektur | Manuel Holm | 50 Sek. |
| 5 - Training | Ron Seifried | 60 Sek. |
| 6 - Ergebnisse | Ron Seifried | 45 Sek. |
| 7 - Demo | Paul Durz | 40 Sek. |
| 8 - Fazit | Manuel Holm | 30 Sek. |

**Gesamtdauer: ca. 5 Minuten**

---

## Folie 1 - Titelfolie

**Sprecher: Paul Durz**  
**Dauer: ca. 20 Sekunden**

### Sprechtext

Willkommen zu unserer Praesentation ueber Pacman Machine Learning. Wir sind Paul Durz, Manuel Holm und Ron Seifried. In den naechsten fuenf Minuten zeigen wir Ihnen, wie wir mit Reinforcement Learning einen Agenten trainiert haben, der das klassische Pacman-Spiel selbststaendig spielen lernt.

### Background / Wissen

- **Reinforcement Learning (RL):** Ein Teilgebiet des maschinellen Lernens, bei dem ein Agent durch Interaktion mit einer Umgebung lernt. Der Agent erhaelt Belohnungen oder Bestrafungen fuer seine Aktionen und optimiert seine Strategie, um die kumulative Belohnung zu maximieren.
- **Pacman als Testumgebung:** Pacman ist ein klassisches Arcade-Spiel von 1980. Es eignet sich hervorragend als RL-Testumgebung, weil es klar definierte Regeln hat, Echtzeit-Entscheidungen erfordert und eine komplexe Dynamik durch die Geister bietet.

---

## Folie 2 - Aufgabenstellung und Motivation

**Sprecher: Paul Durz**  
**Dauer: ca. 45 Sekunden**

### Sprechtext

Unsere Projektaufgabe war es, einen lernenden Agenten fuer ein Spiel zu entwickeln. Wir haben uns fuer den Themenbereich Reinforcement Learning entschieden. Der Agent soll seine Umgebung analysieren und basierend auf einem Belohnungssystem autonom Entscheidungen treffen.

Warum haben wir Pacman gewaehlt? Pacman ist ein klassisches Testbed fuer Game AI mit klar definierten Regeln. Das Spiel erfordert Echtzeit-Entscheidungen und bietet durch die vier Geister eine komplexe Gegner-Dynamik.

Unser Ziel war es, einen Agenten zu entwickeln, der durch Erfahrung lernt, moeglichst viele Punkte zu sammeln und idealerweise das Level zu gewinnen.

### Background / Wissen

- **Agent:** Eine Software-Entitaet, die ihre Umgebung wahrnimmt und Aktionen ausfuehrt, um Ziele zu erreichen.
- **Belohnungssystem (Reward):** Numerische Signale, die dem Agenten mitteilen, wie gut oder schlecht eine Aktion war. Positive Rewards verstaerken Verhalten, negative Rewards entmutigen es.
- **Game AI:** Kuenstliche Intelligenz fuer Spiele, traditionell oft regelbasiert, heute zunehmend durch ML-Methoden ersetzt.
- Die komplexe Gegner-Dynamik bei Pacman entsteht dadurch, dass jeder Geist ein unterschiedliches Verfolgungsmuster hat.

---

## Folie 3 - Loesungsansatz

**Sprecher: Manuel Holm**  
**Dauer: ca. 50 Sekunden**

### Sprechtext

Unser Loesungsansatz besteht aus vier Hauptkomponenten. Erstens haben wir einen Fork des Open-Source-Spiels pacman.c verwendet und mit Mongoose erweitert, einer leichtgewichtigen HTTP-Bibliothek, die uns eine API zur externen Steuerung bereitstellt.

Zweitens haben wir ein Python-Framework entwickelt, in dem unsere ML-Agenten laufen. Die Agenten kommunizieren ueber HTTP mit dem laufenden Spiel.

Drittens haben wir klassisches Q-Learning implementiert. Das ist ein tabellenbasierter RL-Ansatz, bei dem der Agent nur an Kreuzungen Entscheidungen trifft.

Viertens haben wir Deep Q-Learning eingesetzt. Dabei ersetzt ein neuronales Netz die Q-Tabelle und ermoeglicht eine bessere Generalisierung.

Unsere Technologie-Stack umfasst C99 fuer das Spiel, Python und PyTorch fuer das ML-Framework, Mongoose fuer die HTTP-Schnittstelle und Poetry fuer das Dependency-Management.

### Background / Wissen

- **Mongoose:** Eine eingebettete HTTP-Server-Bibliothek fuer C. Ermoeglicht es, REST-APIs direkt in C-Anwendungen zu integrieren.
- **Q-Learning:** Ein modellfreier RL-Algorithmus, der eine Q-Funktion lernt. Die Q-Funktion schaetzt den erwarteten zukuenftigen Reward fuer jede Zustand-Aktion-Kombination.
- **Deep Q-Learning (DQN):** Kombination von Q-Learning mit tiefen neuronalen Netzen. Eingefuehrt von DeepMind 2013 fuer Atari-Spiele.
- **Junction-basierte Entscheidungen:** Anstatt bei jedem Spielschritt zu entscheiden, trifft der Agent nur an Kreuzungen Entscheidungen. Dazwischen laeuft er geradeaus weiter. Dies reduziert die Komplexitaet erheblich.
- **Dueling-Architektur:** Eine DQN-Variante, die den Q-Wert in einen Zustandswert V(s) und einen Aktionsvorteil A(s,a) aufteilt. Ermoeglicht besseres Lernen, welche Zustaende generell gut sind.

---

## Folie 4 - Architektur und RL-Pipeline

**Sprecher: Manuel Holm**  
**Dauer: ca. 50 Sekunden**

### Sprechtext

Hier sehen Sie unsere RL-Pipeline. Der Agent erhaelt vom Environment einen Zustand, der Informationen ueber die Position von Pacman, die Positionen der Geister und die verbleibenden Dots enthaelt.

Basierend auf diesem Zustand waehlt der Agent eine Aktion aus vier moeglichen Richtungen: hoch, runter, links oder rechts. Dazu verwendet er entweder eine Q-Tabelle beim klassischen Q-Learning oder ein neuronales Netz beim Deep Q-Learning.

Die Aktion wird an das Environment - also das Pacman-Spiel - gesendet. Das Environment fuehrt die Aktion aus und gibt einen Reward zurueck. Zum Beispiel plus 15 fuer einen gesammelten Dot oder minus 200 bei Tod.

Mit diesem Reward aktualisiert der Agent seine Policy, also seine Strategie. Dieser Kreislauf wiederholt sich kontinuierlich waehrend des Trainings.

### Background / Wissen

- **State (Zustand):** Eine Repraesentation der aktuellen Spielsituation. Bei uns: Position in einer von 16 Zonen, verfuegbare Richtungen, Geister-Distanz und -Richtung, naechste Dot-Richtung, ob Power-Modus aktiv ist.
- **Action (Aktion):** Die moeglichen Entscheidungen des Agenten. Hier: vier Bewegungsrichtungen.
- **Environment:** Die Umgebung, mit der der Agent interagiert. In unserem Fall das Pacman-Spiel.
- **Policy:** Die Strategie des Agenten, die festlegt, welche Aktion in welchem Zustand gewaehlt wird.
- **Training Loop:** Der wiederholte Zyklus aus Beobachten, Handeln, Reward erhalten und Lernen.

---

## Folie 5 - Training und Reward Design

**Sprecher: Ron Seifried**  
**Dauer: ca. 60 Sekunden**

### Sprechtext

Das Training eines RL-Agenten steht und faellt mit dem Reward Design. Unsere State-Repraesentation unterteilt das Spielfeld in 16 Zonen und erfasst verfuegbare Ausgaenge, Geister-Distanz und -Richtung, die naechste Food-Richtung und ob der Power-Modus aktiv ist.

Bei den Reward-Signalen haben wir folgendes System entwickelt: Plus 15 fuer jeden gesammelten Dot, plus 40 fuer Power-Pills, zwischen 100 und 1700 Punkte fuer gefressene Geister je nach Combo, und plus 2000 fuer ein gewonnenes Level. Bei Tod gibt es eine Strafe zwischen minus 10 und minus 200, abhaengig davon, wie lange der Agent ueberlebt hat.

Besonders wichtig war das 3-Leben-System. Eine Episode umfasst alle drei Leben, wobei die Strafe beim Tod davon abhaengt, wie lange man ueberlebt hat. Das nennen wir Survival-basierte Penalty. Ausserdem gibt es progressive Milestones: Bei 200, 220 und 235 gesammelten Dots erhaelt der Agent Bonuspunkte.

Die Q-Update-Formel zeigt die mathematische Grundlage: Der Q-Wert wird inkrementell mit einer Lernrate alpha angepasst, wobei der zukuenftige Reward mit gamma diskontiert wird.

### Background / Wissen

- **Reward Shaping:** Die Kunst, Belohnungen so zu gestalten, dass der Agent das gewuenschte Verhalten lernt. Zu einfaches Reward-Design fuehrt oft zu unerwuenschtem Verhalten.
- **Alpha (Lernrate):** Bestimmt, wie stark neue Erfahrungen alte Q-Werte ueberschreiben. Typische Werte: 0.1 bis 0.5.
- **Gamma (Discount Factor):** Bestimmt, wie stark zukuenftige Rewards gewichtet werden. Gamma nahe 1 bedeutet weitsichtiges Verhalten.
- **Epsilon-Greedy:** Eine Explorationsstrategie, bei der der Agent mit Wahrscheinlichkeit epsilon zufaellig handelt und sonst die beste bekannte Aktion waehlt.
- **Survival-basierte Penalty:** Je laenger der Agent ueberlebt hat, desto geringer die Strafe beim Tod. Dies vermeidet, dass der Agent Angst vor dem Spielen entwickelt.
- **Experience Replay:** Beim DQN werden Erfahrungen in einem Buffer gespeichert und spaeter in zufaelliger Reihenfolge zum Lernen verwendet. Dies bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen.

---

## Folie 6 - Ergebnisse und Beobachtungen

**Sprecher: Ron Seifried**  
**Dauer: ca. 45 Sekunden**

### Sprechtext

Hier sind unsere Trainingsergebnisse im Vergleich. Beim klassischen Q-Learning mit 5000 Trainingsepisoden stieg der durchschnittliche Score von anfaenglich 699 auf 1238 Punkte. Der beste erreichte Score war 3500 Punkte.

Beim Deep Q-Learning mit 2000 Episoden erreichten wir einen deutlich hoeheren durchschnittlichen Score von fast 2000 Punkten. Der beste Score lag bei 4490 Punkten, und der Agent hat 48 Runden, also 2,4 Prozent, komplett gewonnen.

Die Kernbeobachtung ist eindeutig: Deep Q-Learning uebertrifft das klassische Q-Learning deutlich. Der Grund liegt in der besseren Generalisierung. Waehrend die Q-Tabelle jeden Zustand einzeln lernen muss, kann das neuronale Netz aehnliche Zustaende erkennen und Wissen uebertragen.

### Background / Wissen

- **Episode:** Ein kompletter Spieldurchlauf vom Start bis zum Game Over oder Levelgewinn.
- **Generalisierung:** Die Faehigkeit, aus bekannten Situationen auf unbekannte zu schliessen. Neuronale Netze koennen aehnliche Zustaende zusammenfassen.
- **Curse of Dimensionality:** Das Problem, dass der Zustandsraum exponentiell mit der Anzahl der Features waechst. Q-Tabellen skalieren schlecht, DQN loest dieses Problem.
- Die Gewinnrate von 2,4 Prozent mag niedrig erscheinen, ist aber beachtlich, da Pacman ein sehr schwieriges Spiel ist. Professionelle Spieler haben auch keine 100-prozentige Gewinnrate.

---

## Folie 7 - Demo: Trainierte Agenten

**Sprecher: Paul Durz**  
**Dauer: ca. 40 Sekunden**

### Sprechtext

Hier sehen Sie beide Agenten in Aktion. Links der Q-Learning Agent mit seiner tabellenbasierten Strategie, rechts der Deep Q-Learning Agent mit dem neuronalen Netz.

Beobachten Sie, wie der gelbe Pacman sich durch das Labyrinth bewegt und Dots sammelt. Die weissen Punkte sind die Dots, die er einsammeln muss. Die farbigen Figuren sind die Geister, vor denen er fliehen oder die er nach einer Power-Pill jagen kann.

Sie koennen sehen, dass der DQN-Agent tendenziell fluessiger und effizienter spielt. Er reagiert besser auf Geister und findet schneller Wege zu den verbleibenden Dots.

### Background / Wissen

- Die Demos zeigen Timelapse-Aufnahmen des Trainings, also beschleunigte Aufnahmen mehrerer Episoden.
- **Power-Pill (Energizer):** Die grossen Punkte in den Ecken. Nach dem Essen werden Geister temporaer blau und koennen gefressen werden.
- Der DQN-Agent zeigt weniger zufaelliges Verhalten, weil er bereits gelernt hat, welche Aktionen in welchen Situationen gut sind.
- Bei laengerem Training wuerden die Unterschiede noch deutlicher werden.

---

## Folie 8 - Fazit

**Sprecher: Manuel Holm**  
**Dauer: ca. 30 Sekunden**

### Sprechtext

Zum Abschluss unsere wichtigsten Erkenntnisse. Reinforcement Learning funktioniert fuer Echtzeit-Spiele wie Pacman. Das Reward-Shaping, also das Design der Belohnungsfunktion, ist dabei entscheidend fuer den Erfolg.

Deep Q-Learning skaliert deutlich besser als klassische Q-Tables, weil es aehnliche Zustaende generalisieren kann. Unsere Junction-basierten Entscheidungen haben die Komplexitaet erheblich reduziert.

Bei den Limitationen muessen wir erwaehnen, dass das Training sehr zeitaufwaendig ist. Das Hyperparameter-Tuning erfordert viele Experimente. Ausserdem haben wir mit deterministischem Geisterverhalten und nur einem Geist trainiert. Multi-Ghost-Szenarien waeren deutlich komplexer.

Vielen Dank fuer Ihre Aufmerksamkeit.

### Background / Wissen

- **Hyperparameter:** Parameter des Lernalgorithmus, die vor dem Training festgelegt werden muessen: Lernrate, Discount-Faktor, Netzwerkgroesse, Batch-Groesse, etc.
- **Deterministisches Geisterverhalten:** Die Geister folgen festen Regeln, sind also vorhersehbar. Im Original-Pacman haben die vier Geister unterschiedliche Persoenlichkeiten.
- **Multi-Ghost-Szenarien:** Mit mehreren Geistern waechst die Komplexitaet stark, da der Agent gleichzeitig mehrere Bedrohungen beruecksichtigen muss.
- Moegliche Erweiterungen: A3C oder PPO Algorithmen, Transfer Learning auf andere Level, menschenaehnlichere Geister-KI.

---

## Potenzielle Rueckfragen und Antworten

### Was ist der Unterschied zwischen Q-Learning und Deep Q-Learning?

Q-Learning speichert Q-Werte in einer Tabelle mit einem Eintrag pro Zustand-Aktion-Paar. Deep Q-Learning ersetzt diese Tabelle durch ein neuronales Netz, das die Q-Funktion approximiert. Das ermoeglicht Generalisierung ueber aehnliche Zustaende.

### Warum habt ihr Mongoose verwendet?

Mongoose ist eine leichtgewichtige, eingebettete HTTP-Server-Bibliothek fuer C. Sie ermoeglicht es uns, das C-Spiel um eine REST-API zu erweitern, ueber die Python-Agenten das Spiel steuern koennen.

### Was ist Dueling DQN?

Dueling DQN ist eine Architekturvariante, die den Q-Wert Q(s,a) in zwei Teile aufspaltet: den Zustandswert V(s) und den Aktionsvorteil A(s,a). Dies hilft dem Netz zu lernen, welche Zustaende generell gut oder schlecht sind, unabhaengig von der spezifischen Aktion.

### Warum nur an Junctions entscheiden?

In Gaengen gibt es nur eine sinnvolle Richtung - geradeaus. Entscheidungen an jedem Schritt wuerden den Zustandsraum unnoetig aufblaahen und das Lernen verlangsamen. Junction-basierte Entscheidungen fokussieren das Lernen auf die relevanten Entscheidungspunkte.

### Wie lange dauert das Training?

5000 Episoden Q-Learning dauern etwa 2-3 Stunden. 2000 Episoden DQN dauern etwa 4-6 Stunden, je nach Hardware. Die Trainingszeit ist ein wesentlicher Faktor bei RL-Experimenten.
