# Zeitkritische Aufgaben

Ein Dispatcher mit Warteschlange kann Aufgaben nach Deadlines sortieren.
Ein typisches Beispiel ist die fristgerechte Berichtserstellung. Der Nutzer
reicht einen Task mit Deadline ein und der Dispatcher startet ihn rechtzeitig
vor Ablauf. Abgelaufene Tasks werden als `expired` markiert und nicht mehr
verarbeitet.
