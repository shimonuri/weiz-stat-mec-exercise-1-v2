import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class Result:
    partition_function: float
    lambda_max: float
    f_stat_mech: float
    f_approx: float
    f_entropy: float


def main(magnetic_field, magnetization_coefficient, number_of_particles):
    f_stat_mech = []
    f_approx = []
    f_entropy = []
    temperatures = np.linspace(10, 100, 5)
    for temperature in temperatures:
        result = calculate(
            number_of_particles,
            magnetic_field=magnetic_field,
            magnetization_coefficient=magnetization_coefficient,
            temperature=temperature,
        )
        f_stat_mech.append(result.f_stat_mech)
        f_approx.append(result.f_approx)
        f_entropy.append(result.f_entropy)

    plt.title(f"Number of Particles = {number_of_particles}")
    plt.plot(temperatures, f_stat_mech, label="f_stat_mech")
    # plt.plot(temperatures, f_approx, label="f_approx")
    plt.plot(temperatures, f_entropy, label="f_entropy")
    plt.legend()
    plt.show()


def calculate(
    number_of_particles, magnetic_field, magnetization_coefficient, temperature
):
    partition_function = get_partition_function(
        number_of_particles, magnetic_field, magnetization_coefficient, temperature
    )
    lambda_max = get_lambda_max(magnetic_field, magnetization_coefficient, temperature)
    return Result(
        partition_function=partition_function,
        lambda_max=lambda_max,
        f_stat_mech=-np.log(partition_function),
        f_approx=-np.log(lambda_max**number_of_particles),
        f_entropy=get_free_energy_from_entropy(
            number_of_particles, magnetic_field, magnetization_coefficient, temperature
        ),
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
        energy += -magnetization_coefficient * spins[i] * spins[(i + 1) % len(spins)]
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
    number_of_spins, magnetic_field, magnetization_coefficient, temperature
):
    energy = get_average_energy(
        magnetic_field, magnetization_coefficient, number_of_spins, temperature
    )
    giibs_entropy = get_gibbs_entropy(
        temperature, magnetic_field, magnetization_coefficient, number_of_spins
    )
    entropy = get_entropy(
        energy, magnetic_field, magnetization_coefficient, number_of_spins
    )

    return energy - temperature * entropy


def get_entropy(energy, magnetic_field, magnetization_coefficient, number_of_spins):
    number_of_combinations = get_number_of_combinations(
        energy, magnetic_field, magnetization_coefficient, number_of_spins
    )
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
        if abs(configuration_energy - energy) < 1:
            number_of_combinations += 1

    return number_of_combinations


def get_average_energy(
    magnetic_field, magnetization_coefficient, number_of_spins, temperature
):
    partition_function = get_partition_function(
        number_of_spins, magnetic_field, magnetization_coefficient, temperature
    )
    energy = 0
    beta = 1 / temperature
    energies = []
    for spins in iter_spins(number_of_spins):
        configuration_energy = get_energy(
            spins, magnetic_field, magnetization_coefficient
        )
        energies.append(configuration_energy)
        energy += configuration_energy * np.exp(-configuration_energy * beta)

    return energy / partition_function


if __name__ == "__main__":
    for number_of_particles in [7]:
        main(
            magnetic_field=-1,
            magnetization_coefficient=0,
            number_of_particles=number_of_particles,
        )
