
def main():
    syöte = input("Give a list of integers separated by space:").strip()\
        .split(" ")

    numerot = [eval(i) for i in syöte]
    numerot.sort()
    print("Given numbers sorted: ", numerot)


if __name__ == "__main__":
    main()
