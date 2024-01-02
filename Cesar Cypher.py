from timeit import default_timer as timer
list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def shift(text, shift2):
    result = ""
    if abs(shift2) > len(list) - 1:
        if shift2 > 0:
            while shift2 > len(list) - 1:
                shift2 = shift2 - len(list)
        else:
            while shift2 < -len(list) + 1:
                shift2 = shift2 + len(list)

    for letter in text:
        if list.count(letter) == 0:
            result += letter
        else:
            new_spot = list.index(letter) + shift2
            if new_spot > len(list) - 1:
                new_spot -= len(list)
            elif new_spot < -len(list) + 1:
                new_spot += len(list)
            result += list[new_spot]
    return result


def enc_or_dec(result, enc_or_dec_input, shift_key=0, plaintxt=""):

    if enc_or_dec_input == 1:
        result = shift(plaintxt, shift_key)

    elif enc_or_dec_input == 2:
        result = shift(plaintxt, -shift_key)

    elif enc_or_dec_input == 3:

        for key in range(0, len(list)):
            result += str(key) + " " + shift(plaintxt, key) + "\n"

    return result

def main():

    while True:
        enc_or_dec_input = int(input("Enter 1 to encrypt, 2 to decrypt or 3 to crack: "))
        result = ""
        if enc_or_dec_input != 3:
            keys = input("Enter shift key(s): ").split(" ")
            plaintxt = input("Input text: ").lower()

            t0 = timer()
            for shift_key in keys:
                result = enc_or_dec(result, enc_or_dec_input, int(shift_key),
                                    plaintxt)
                plaintxt = result
            t1 = timer()
        else:
            plaintxt = input("Input text: ").lower()
            t0 = timer()
            result = enc_or_dec(result, enc_or_dec_input, None,  plaintxt)
            t1 = timer()

        print(result, " : ", t1 - t0, " s")

        cont = input("Continue?(Y/N) ").lower()
        if cont == "n":
            return False


if __name__ == '__main__':
    main()
