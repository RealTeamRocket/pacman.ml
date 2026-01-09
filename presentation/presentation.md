# Präsentationsskript (5 Minuten) – Pacman Reinforcement Learning

## Sprecheraufteilung
- **Paul Durz:** Folie 1–3 (Einführung, Motivation, Lösungsansatz)
- **Manuel Holm:** Folie 4–5 (Architektur, Training & Reward Design)
- **Ron Seifried:** Folie 6–8 (Ergebnisse, Demo, Fazit)

---

# Paul Durz – Folien 1–3

---

## Folie 1 – Pacman Machine Learning (≈ 30s)

**Stichworte**
- Begrüßung
- Projektname: Pacman Machine Learning
- Thema: Reinforcement Learning
- Zwei Ansätze: Q-Learning und Deep Q-Learning
- HFT Stuttgart, Modul ML & Data Mining
- Gruppenarbeit: Paul, Manuel, Ron

**Sprechtext**

Guten Tag und herzlich willkommen zu unserer Präsentation. Wir sind Paul, Manuel und Ron, und wir stellen heute unser Projekt „Pacman Machine Learning" vor.

In diesem Projekt haben wir Reinforcement Learning auf das klassische Pacman-Spiel angewandt. Dabei haben wir zwei verschiedene Ansätze implementiert: Q-Learning und Deep Q-Learning.

Diese Arbeit entstand im Rahmen des Moduls „Machine Learning und Data Mining" an der Hochschule für Technik Stuttgart.

**Background Info**
- Reinforcement Learning (RL): Ein Teilbereich des maschinellen Lernens, bei dem ein Agent durch Interaktion mit einer Umgebung lernt, optimale Entscheidungen zu treffen.
- Q-Learning: Tabellenbasierter RL-Algorithmus (1989, Watkins).
- Deep Q-Learning: Kombination von Q-Learning mit neuronalen Netzen (2013, DeepMind).

---

## Folie 2 – Aufgabenstellung und Motivation (≈ 45s)

**Stichworte**
- Themenbereich: Reinforcement Learning
- Lernende Agenten für Spiele entwickeln
- Belohnungssystem zur Umgebungsanalyse
- Autonome Entscheidungsfindung
- Warum Pacman? Klassisches Testbed
- Klare Regeln, Echtzeit-Entscheidungen
- Komplexe Gegner-Dynamik (Geister)
- Ziel: Agent lernt Punkte sammeln + Level gewinnen

**Sprechtext**

Unsere Aufgabe war es, lernende Agenten zu entwickeln, die das Pacman-Spiel selbstständig spielen können. Der Agent soll durch ein Belohnungssystem lernen, die Umgebung zu analysieren und autonome Entscheidungen zu treffen.

Warum haben wir Pacman gewählt? Pacman ist ein klassisches Testbed für Game AI. Es hat klar definierte Regeln und Ziele, erfordert Echtzeit-Entscheidungen und bietet mit den Geistern eine komplexe Gegner-Dynamik.

Unser konkretes Ziel war: Ein Agent, der durch Erfahrung lernt, möglichst viele Punkte zu sammeln und das Level zu gewinnen.

**Background Info**
- Testbed: Eine kontrollierte Umgebung zum Testen von Algorithmen.
- Game AI: Künstliche Intelligenz für Spielumgebungen – oft verwendet für RL-Forschung, da Spiele klare Belohnungsstrukturen haben.
- Echtzeit-Entscheidungen: Der Agent muss kontinuierlich handeln; keine Zeit für langes Nachdenken.
- Rückfrage „Warum nicht ein anderes Spiel?": Pacman bietet eine gute Balance aus Komplexität und Übersichtlichkeit, ist gut dokumentiert und hat eine überschaubare Zustandsraum-Größe.

---

## Folie 3 – Lösungsansatz (≈ 45s)

**Stichworte**
- 🎮 Pacman in C: Fork von pacman.c, Mongoose für externe Steuerung
- 🐍 Python ML-Framework: HTTP-Kommunikation mit dem Spiel
- 📊 Q-Learning: Tabellenbasiert, Junction-Entscheidungen
- 🧠 Deep Q-Learning: Neuronales Netz, Dueling-Architektur
- Tech-Stack: C99, Python, PyTorch, Mongoose, Poetry

**Sprechtext**

Unser Lösungsansatz besteht aus vier Komponenten:

Erstens: Das Pacman-Spiel selbst. Wir haben einen Fork des Open-Source-Projekts pacman.c verwendet und diesen mit Mongoose erweitert, um eine externe Steuerung zu ermöglichen.

Zweitens: Ein Python-basiertes ML-Framework. Unsere Agenten kommunizieren via HTTP mit dem C-Spiel.

Drittens: Ein Q-Learning-Agent. Dieser nutzt eine tabellenbasierte Strategie und trifft Entscheidungen an Kreuzungen – sogenannten Junctions.

Viertens: Ein Deep Q-Learning-Agent. Dieser verwendet ein neuronales Netz mit Dueling-Architektur für bessere Generalisierung.

Unser Tech-Stack umfasst C99 für das Spiel, Python für die Agenten, PyTorch für Deep Learning, Mongoose für die HTTP-API und Poetry für das Dependency-Management.

**Background Info**
- pacman.c: Ein minimalistischer Pacman-Klon in C99, ursprünglich von Flooh entwickelt.
- Mongoose: Eine eingebettete HTTP-Server-Bibliothek für C/C++.
- Dueling-Architektur: Eine DQN-Variante, die Value und Advantage getrennt schätzt (Wang et al., 2016).
- Junction-Entscheidungen: Der Agent entscheidet nur an Kreuzungen, nicht bei jedem Frame – reduziert die Komplexität erheblich.
- Rückfrage „Warum HTTP statt direkter Integration?": Entkopplung ermöglicht unabhängige Entwicklung und einfaches Debugging.

---

# Manuel Holm – Folien 4–5

---

## Folie 4 – Architektur und RL-Pipeline (≈ 45s)

**Stichworte**
- Zustand: Position, Geister, Dots
- Agent: Q-Table oder DQN
- Aktion: Hoch, Runter, Links, Rechts
- Environment: Pacman.c
- Reward: +15 Dot, -200 Tod
- Trainingsschleife: Zustand → Aktion → Reward → Update

**Sprechtext**

Hier sehen Sie unsere Reinforcement-Learning-Pipeline.

Der Agent erhält einen Zustand aus der Umgebung. Dieser Zustand enthält Informationen über die aktuelle Position, die Geister und die verbleibenden Dots.

Basierend auf diesem Zustand wählt der Agent eine Aktion – entweder über eine Q-Table beim klassischen Q-Learning oder über ein neuronales Netz beim Deep Q-Learning. Die möglichen Aktionen sind: Hoch, Runter, Links, Rechts.

Die Aktion wird an das Environment – also das Pacman-Spiel – gesendet. Das Environment gibt einen Reward zurück. Zum Beispiel plus 15 für einen gesammelten Dot oder minus 200 bei Tod.

Dieser Reward wird genutzt, um die Policy des Agenten zu aktualisieren. Dann wiederholt sich der Zyklus.

**Background Info**
- Policy: Die Strategie des Agenten, die Zuständen Aktionen zuordnet.
- Q-Table: Eine Tabelle, die für jedes Zustands-Aktions-Paar einen Q-Wert speichert.
- DQN (Deep Q-Network): Ein neuronales Netz, das Q-Werte approximiert, wenn der Zustandsraum zu groß für eine Tabelle ist.
- Rückfrage „Warum nur 4 Aktionen?": Pacman bewegt sich diskret in einem Gitter; diagonale Bewegungen gibt es nicht.

---

## Folie 5 – Training und Reward Design (≈ 45s)

**Stichworte**
- State-Repräsentation: 16 Zonen, Ausgänge, Geister-Distanz, Food-Richtung, Power-Modus
- Reward-Signale: +15 Dot, +40 Power-Pill, +100–1700 Geist, -10 bis -200 Tod, +2000 Level gewonnen
- Besonderheiten: 3-Leben-System, Survival-Penalty, Dot-Milestones, Junction-Entscheidungen
- Q-Update-Formel

**Sprechtext**

Für das Training haben wir eine kompakte State-Repräsentation entwickelt. Das Spielfeld ist in 16 Zonen eingeteilt. Der Zustand enthält: verfügbare Ausgänge, Distanz und Richtung zum nächsten Geist, Richtung zum nächsten Food, und ob der Power-Modus aktiv ist.

Beim Reward Design haben wir differenzierte Signale definiert: Plus 15 pro Dot, plus 40 für eine Power-Pill, plus 100 bis 1700 für das Fressen eines Geistes – je nachdem wie viele Geister hintereinander gefressen werden. Minus 10 bis minus 200 bei Tod, abhängig von der Spielphase. Und plus 2000 für ein gewonnenes Level.

Zu den Besonderheiten: Jede Episode hat 3 Leben. Es gibt eine Survival-basierte Penalty, progressive Dot-Milestones als Zwischenbelohnungen, und der Agent entscheidet nur an Junctions.

Die Q-Update-Formel sehen Sie hier: Q von s,a wird aktualisiert mit Alpha mal der Differenz aus erhaltenem Reward plus diskontiertem maximalem Q-Wert des Folgezustands minus dem aktuellen Q-Wert.

**Background Info**
- Alpha (Lernrate): Bestimmt, wie stark neue Informationen gewichtet werden (typisch: 0.1–0.5).
- Gamma (Discount-Faktor): Gewichtet zukünftige Rewards (typisch: 0.9–0.99).
- Power-Modus: Nach Aufnahme einer Power-Pill kann Pacman für kurze Zeit Geister fressen.
- Survival-Penalty: Längeres Überleben ohne Punkte wird bestraft, um passives Verhalten zu vermeiden.
- Rückfrage „Warum Junction-basiert?": Reduziert die Anzahl der Entscheidungspunkte erheblich und macht das Learning effizienter.

---

# Ron Seifried – Folien 6–8

---

## Folie 6 – Ergebnisse und Beobachtungen (≈ 45s)

**Stichworte**
- Q-Learning (5000 Episoden):
  - Durchschn. Score: 1238.3
  - Durchschn. Dots: 102.0
  - Best Score: 3500
  - Best Dots: 218
  - 200+ Dots: 7 Episoden
- Deep Q-Learning (2000 Episoden):
  - Durchschn. Score: 1981.7
  - Durchschn. Dots: 163.5
  - Best Score: 4490
  - Best Dots: 244
  - Gewonnene Runden: 48 (2.4%)
- Kernbeobachtung: Deep Q-Learning übertrifft Q-Learning, generalisiert besser

**Sprechtext**

Kommen wir zu unseren Ergebnissen.

Bei Q-Learning nach 5000 Episoden erreichten wir einen durchschnittlichen Score von 1238 Punkten und durchschnittlich 102 gesammelte Dots. Der beste Score lag bei 3500, die beste Dot-Anzahl bei 218. In 7 Episoden wurden mehr als 200 Dots gesammelt.

Bei Deep Q-Learning – hier nach 2000 Episoden – sehen wir deutlich bessere Werte: Ein durchschnittlicher Score von fast 1982, durchschnittlich 163 Dots. Der beste Score erreichte 4490 Punkte, die beste Dot-Anzahl 244. Und wir hatten 48 gewonnene Runden, das entspricht 2,4 Prozent.

Die Kernbeobachtung: Deep Q-Learning übertrifft Q-Learning deutlich. Es generalisiert besser über den Zustandsraum, weil das neuronale Netz ähnliche Zustände ähnlich behandeln kann.

**Background Info**
- Episoden: Ein komplettes Spiel von Start bis Game Over (oder Level gewonnen).
- Dots: Es gibt 244 Dots im Level; alle zu sammeln bedeutet Level gewonnen.
- Win-Rate 2.4%: Klingt niedrig, ist aber für RL-Agenten ohne Vorwissen ein respektables Ergebnis.
- Rückfrage „Warum weniger Episoden bei DQN?": DQN lernt effizienter pro Episode, benötigt aber mehr Rechenzeit pro Episode.

---

## Folie 7 – Demo: Trainierte Agenten (≈ 30s)

**Stichworte**
- Zwei GIFs nebeneinander
- Links: Q-Learning Agent (tabellenbasiert)
- Rechts: Deep Q-Learning Agent (Dueling DQN)
- Legende: Gelb = Pacman, Weiß = Dots, Rot = Geister
- Beobachtung: Unterschiedliche Spielstile

**Sprechtext**

Hier sehen Sie unsere beiden trainierten Agenten in Aktion.

Links der Q-Learning-Agent mit seiner tabellenbasierten Strategie. Rechts der Deep Q-Learning-Agent mit der Dueling-Architektur.

Gelb ist Pacman – also unser Agent. Die weißen Punkte sind die Dots, die gesammelt werden müssen. Und rot sind die Geister, denen der Agent ausweichen muss.

Beachten Sie die unterschiedlichen Spielstile: Der Q-Learning-Agent folgt oft festen Mustern, während der Deep Q-Learning-Agent flexibler auf Situationen reagiert.

**Background Info**
- Die GIFs zeigen Timelapse-Aufnahmen aus dem Training.
- Tabellenbasiert: Deterministisch bei gleichem Zustand; kann zu repetitiven Mustern führen.
- DQN: Kann generalisieren; ähnliche Zustände führen zu ähnlichem Verhalten.
- Rückfrage „Wie lange dauert eine Episode?": Typisch 30–60 Sekunden Echtzeit, je nach Spielerfolg.

---

## Folie 8 – Fazit (≈ 30s)

**Stichworte**
- Erkenntnisse:
  - RL funktioniert für Echtzeit-Spiele
  - Reward-Shaping ist entscheidend
  - Deep Q-Learning skaliert besser als Q-Tables
  - Junction-Entscheidungen reduzieren Komplexität
- Limitationen:
  - Lange Trainingszeiten
  - Hyperparameter-Tuning aufwändig
  - Deterministisches Geisterverhalten
  - Multi-Ghost-Szenarien

**Sprechtext**

Zum Abschluss unser Fazit.

Unsere wichtigsten Erkenntnisse: Reinforcement Learning funktioniert für Echtzeit-Spiele wie Pacman. Das Reward-Shaping – also das Design der Belohnungssignale – ist entscheidend für den Lernerfolg. Deep Q-Learning skaliert besser als tabellenbasierte Ansätze. Und die Beschränkung auf Junction-Entscheidungen reduziert die Komplexität erheblich.

Zu den Limitationen: Das Training benötigt viel Zeit. Das Hyperparameter-Tuning ist aufwändig. In unserem Setup verhalten sich die Geister deterministisch, was das Problem vereinfacht. Und Multi-Ghost-Szenarien mit mehr als einem Geist sind deutlich schwieriger.

Vielen Dank für Ihre Aufmerksamkeit. Wir freuen uns auf Ihre Fragen.

**Background Info**
- Trainingszeit: Q-Learning ca. 2–3 Stunden für 5000 Episoden; DQN ca. 4–6 Stunden für 2000 Episoden (auf CPU).
- Deterministisches Geisterverhalten: In unserem Setup folgen Geister festen Regeln; im Original-Pacman gibt es auch zufällige Elemente.
- Rückfrage „Was wären nächste Schritte?": Mehr Geister, nicht-deterministisches Verhalten, Transfer auf andere Level, Multi-Agent-Szenarien.
- Rückfrage „Könnte der Agent das Spiel perfekt spielen?": Theoretisch ja, praktisch limitiert durch Zustandsraum-Größe und Trainingszeit.

---

# Timing-Übersicht

| Folie | Titel                          | Sprecher | Zeit   |
|-------|--------------------------------|----------|--------|
| 1     | Pacman Machine Learning        | Paul     | 30s    |
| 2     | Aufgabenstellung und Motivation| Paul     | 45s    |
| 3     | Lösungsansatz                  | Paul     | 45s    |
| 4     | Architektur und RL-Pipeline   | Manuel   | 45s    |
| 5     | Training und Reward Design     | Manuel   | 45s    |
| 6     | Ergebnisse und Beobachtungen   | Ron      | 45s    |
| 7     | Demo: Trainierte Agenten       | Ron      | 30s    |
| 8     | Fazit                          | Ron      | 30s    |
| **Σ** |                                |          | **5:15** |

*Hinweis: Pufferzeit von ca. 15 Sekunden für Übergänge und eventuelle Nachfragen einplanen.*