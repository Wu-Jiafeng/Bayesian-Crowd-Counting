import random

def write(filepath,datas):
    file=open(filepath, "w", encoding="utf-8")
    for data in datas:
        file.write(data+"\n")
    file.close()

if __name__=="__main__":
    train_set, val_set = [], []
    for i in range(1, 301):
        filepath = "IMG_" + str(i) + ".jpg"
        if random.random() <= 0.9:
            train_set.append(filepath)
        else:
            val_set.append(filepath)

    random.shuffle(train_set), random.shuffle(val_set)
    write("./train.txt",train_set),write("./val.txt",val_set)
