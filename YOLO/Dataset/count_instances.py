
import glob, os

TextList = []

for file in glob.glob("*.txt"):
    with open(file, 'r') as f:
	    for line in f:
		    TextList.append(line.split(None,1)[0])

capacity = len(TextList)
index = 0

while index != capacity:
	line = TextList[index]

	for word in line.split():
		index += 1

def count_instance():

    Giocatore = 0
    Guardalinee = 0
    Palla = 0
    Arbitro = 0

    for i in TextList:
        if i == str(0):
            Giocatore += 1
        if i == str(1):
            Arbitro += 1
        if i == str(2):
            Palla += 1
        if i == str(3):
            Guardalinee += 1

    print("CONTEGGIO ISTANZE NEL DATASET\n\n Giocatore: {} \n Guardalinee: {} "
          "\n Palla: {} \n Arbitro: {}"
               .format(Giocatore, Guardalinee, Palla, Arbitro))
    
    f = open("outputCount.txt", "a")
    print("CONTEGGIO ISTANZE NEL DATASET\n\n Giocatore: {} \n Guardalinee: {} "
          "\n Palla: {} \n Arbitro: {}"
               .format(Giocatore, Guardalinee, Palla, Arbitro), file=f)
    f.close()

count_instance()




