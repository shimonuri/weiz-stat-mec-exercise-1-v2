import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
import pandas as pd


plt.rcParams.update({"font.size": 22})


@dataclass
class Result:
    partition_function: float
    lambda_max: float
    f_stat_mech: float
    f_approx: float
    f_thermodynamics: float
    energy: float


def main(magnetic_field, magnetization_coefficient, number_of_particles, ax=None):
    f_stat_mech = []
    f_approx = []
    f_thermodynamics = []
    energies = []
    temperatures = np.linspace(0.28e-1, 1, 100)
    for temperature in temperatures:
        result = calculate(
            number_of_particles,
            magnetic_field=magnetic_field,
            magnetization_coefficient=magnetization_coefficient,
            temperature=temperature,
        )
        f_stat_mech.append(result.f_stat_mech)
        f_approx.append(result.f_approx)
        f_thermodynamics.append(result.f_thermodynamics)
        energies.append(result.energy)

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(f"$N = {number_of_particles}$")
    ax.plot(
        temperatures, abs(np.array(f_stat_mech) - np.array(f_approx)), label="energy"
    )
    # ax.plot(temperatures, f_stat_mech, label="f_stat_mech")
    # ax.plot(temperatures, f_approx, label="f_approx")
    # ax.plot(temperatures, f_thermodynamics, label="f_thermodynamics")


def calculate(
    number_of_particles, magnetic_field, magnetization_coefficient, temperature
):
    partition_function = get_partition_function(
        number_of_particles, magnetic_field, magnetization_coefficient, temperature
    )
    energy = get_average_energy(
        magnetic_field, magnetization_coefficient, number_of_particles, temperature
    )
    lambda_max = get_lambda_max(magnetic_field, magnetization_coefficient, temperature)
    f_entropy = get_free_energy_from_entropy(
        number_of_particles,
        magnetic_field,
        magnetization_coefficient,
        temperature,
        energy,
    )
    return Result(
        partition_function=partition_function,
        lambda_max=lambda_max,
        f_stat_mech=-temperature * np.log(partition_function),
        f_approx=-temperature * number_of_particles * np.log(lambda_max),
        f_thermodynamics=f_entropy,
        energy=energy,
    )


def get_partition_function(
    number_of_spins, magnetic_field, magnetization_coefficient, temperature
):
    partition_function = 0
    beta = 1 / temperature
    for spins in iter_spins(number_of_spins):
        configuration_energy = get_energy(
            spins, magnetic_field, magnetization_coefficient
        )
        partition_function += np.exp(-configuration_energy * beta)

    return partition_function


def iter_spins(number_of_spins):
    for i in range(2**number_of_spins):
        spins = []
        for j in range(number_of_spins):
            spins.append(1) if i >> j & 1 == 1 else spins.append(-1)

        yield spins


def get_energy(spins, magnetic_field, magnetization_coefficient):
    energy = 0
    for i in range(len(spins)):
        energy += (
            -(magnetization_coefficient / 2) * spins[i] * spins[(i + 1) % len(spins)]
        )
        energy += -magnetic_field * spins[i]

    return energy


def get_lambda_max(magnetic_field, magnetization_coefficient, temperature):
    beta = 1 / temperature
    return np.exp(beta * magnetization_coefficient) * np.cosh(
        beta * magnetic_field
    ) + np.sqrt(
        np.exp(2 * beta * magnetization_coefficient)
        * np.power(np.cosh(beta * magnetic_field), 2)
        - 2 * np.sinh(2 * beta * magnetic_field)
    )


def get_free_energy_from_entropy(
    number_of_spins, magnetic_field, magnetization_coefficient, temperature, energy
):
    entropy = get_entropy(
        energy, magnetic_field, magnetization_coefficient, number_of_spins
    )

    return energy - temperature * entropy


def get_entropy(energy, magnetic_field, magnetization_coefficient, number_of_spins):
    number_of_combinations = get_number_of_combinations(
        energy, magnetic_field, magnetization_coefficient, number_of_spins
    )
    # print(number_of_combinations)
    if number_of_combinations == 0:
        return np.nan
    return np.log(number_of_combinations)


def get_gibbs_entropy(
    temperature, magnetic_field, magnetization_coefficient, number_of_spins
):
    entropy = 0
    beta = 1 / temperature
    partition_function = get_partition_function(
        number_of_spins, magnetic_field, magnetization_coefficient, temperature
    )
    for spins in iter_spins(number_of_spins):
        configuration_energy = get_energy(
            spins, magnetic_field, magnetization_coefficient
        )
        p_i = np.exp(-configuration_energy * beta) / partition_function
        entropy += -p_i * np.log(p_i)

    return entropy


def get_number_of_combinations(
    energy, magnetic_field, magnetization_coefficient, number_of_spins
):
    number_of_combinations = 0
    for spins in iter_spins(number_of_spins):
        configuration_energy = get_energy(
            spins, magnetic_field, magnetization_coefficient
        )
        if abs(configuration_energy - energy) < max(
            abs(magnetization_coefficient), abs(magnetic_field)
        ):
            number_of_combinations += 1
            # print(spins)

    return number_of_combinations


def get_average_energy(
    magnetic_field, magnetization_coefficient, number_of_spins, temperature
):
    partition_function = get_partition_function(
        number_of_spins, magnetic_field, magnetization_coefficient, temperature
    )
    energy = 0
    beta = 1 / temperature
    for spins in iter_spins(number_of_spins):
        configuration_energy = get_energy(
            spins, magnetic_field, magnetization_coefficient
        )
        energy += configuration_energy * np.exp(-configuration_energy * beta)

    return energy / partition_function


flag = False
if __name__ == "__main__":
    if flag:
        numbers_of_particles = range(1, 13)
        subplots = plt.subplots(4, 3, figsize=(15, 15), layout="tight")
    else:
        numbers_of_particles = [12]

    for number_of_particles in numbers_of_particles:
        subplot_x = (number_of_particles - 1) // 3
        subplot_y = (number_of_particles - 1) % 3
        main(
            magnetic_field=-1,
            magnetization_coefficient=1,
            number_of_particles=number_of_particles,
            ax=subplots[1][subplot_x][subplot_y]
            if len(numbers_of_particles) is not 1
            else None,
        )

    plt.savefig("n12_diff.png", dpi=300, bbox_inches="tight")
    plt.show()
