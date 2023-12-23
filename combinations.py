import numeric

import numpy as np


def iter_spins(number_of_spins, number_of_positive_spins):
    if number_of_positive_spins == 0:
        yield [-1] * number_of_spins
    elif number_of_positive_spins == number_of_spins:
        yield [1] * number_of_spins
    else:
        for spins in iter_spins(number_of_spins - 1, number_of_positive_spins - 1):
            yield [1] + spins
        for spins in iter_spins(number_of_spins - 1, number_of_positive_spins):
            yield [-1] + spins


def get_flips(number_of_spins, number_of_positive_spins):
    number_of_flips_to_count = {}
    for spins in iter_spins(number_of_spins, number_of_positive_spins):
        mul = 0
        for i in range(len(spins)):
            if i == len(spins) - 1:
                continue
            mul += spins[i] * spins[i + 1]

        number_of_flips = (number_of_spins - 1 - mul) / 2
        if number_of_flips not in number_of_flips_to_count:
            number_of_flips_to_count[number_of_flips] = 1
        else:
            number_of_flips_to_count[number_of_flips] += 1

    print(sorted(number_of_flips_to_count.items()))
    print(f"total_cases = {sum(number_of_flips_to_count.values())}")


if __name__ == "__main__":
    get_flips(10, 3)
