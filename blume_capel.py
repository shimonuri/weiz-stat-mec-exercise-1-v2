import numpy as np
import matplotlib.pyplot as plt

# set font size
plt.rcParams.update({"font.size": 22})


def get_mean_magnetization(
    magnetic_field,
    magnetization_coefficient,
    lamda,
    temperature,
    number_of_particles,
    diff_accuracy=1e-6,
):
    beta = 1 / temperature

    def get_lambda(B):
        matrix = np.array(
            [
                [
                    np.exp(beta * (magnetization_coefficient + B)),
                    np.exp((beta / 2) * (B + lamda)),
                    np.exp(-beta * magnetization_coefficient),
                ],
                [
                    np.exp((beta / 2) * (B + lamda)),
                    np.exp(beta * lamda),
                    np.exp((beta / 2) * (-B + lamda)),
                ],
                [
                    np.exp(-beta * magnetization_coefficient),
                    np.exp((beta / 2) * (-B + lamda)),
                    np.exp(beta * (magnetization_coefficient - B)),
                ],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return max(eigenvalues)

    # deffrintiate lambda with respect to B
    diff = (
        np.log(get_lambda(magnetic_field + diff_accuracy))
        - np.log(get_lambda(magnetic_field))
    ) / diff_accuracy
    return number_of_particles * temperature * diff


def plot(
    magnetization_coefficient,
    lamda,
    temperature,
    number_of_particles,
):
    magnetic_fields = np.linspace(-1, 1, 1000)
    plt.plot(
        magnetic_fields,
        [
            get_mean_magnetization(
                magnetic_field,
                magnetization_coefficient,
                lamda,
                temperature,
                number_of_particles,
            )
            for magnetic_field in magnetic_fields
        ],
        label="$\\lambda = {:.2f}$".format(lamda),
    )


if __name__ == "__main__":
    for lamda in np.linspace(1, -1, 5):
        plot(
            magnetization_coefficient=1,
            lamda=lamda,
            temperature=1,
            number_of_particles=1,
        )
    plt.xlabel("$B$")
    plt.ylabel("$\\langle M \\rangle$")
    plt.legend(fontsize=16)
    plt.savefig("output/blume_capel.png", bbox_inches="tight")
    plt.show()
