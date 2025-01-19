from src.func import *
from src.fss import FishSchoolSearch
from src.draw import draw_2d


def main():
    # params
    iterations_num = 1000
    population_size = 50
    weight_scale = 200.0
    selected_optimum = True  # True - minimum, False - maximum

    """
    params: function name, dimension, flag name, min range, max range, init individual step
    """
    functions = [
        (Himmelblau, 2, "Himmelblau", -4, 4, 0.05),
        (Rosenbrock, 2, "Rosenbrock", -4, 4, 0.05),
        (Easom, 2, "Easom", -10, 10, 0.1),
        (CrossInTray, 2, "CrossInTray", -10, 10, 0.1),
        (Booth, 3, "Booth", -10, 10, 0.1),
        (GriffithStokes, 3, "GriffithStokes", -4, 4, 0.05),
        (Sphere, 3, "Sphere", -10, 10, 0.1)
    ]

    print("Select function:")
    print("0: All function")
    for idx, (func_class, dim, flag, min_range, max_range, individual_step) in enumerate(functions):
        print(f"{idx + 1}: {func_class.__name__} (dimension {dim})")
    choice = int(input("Enter function number (0-7): "))
    if choice < 0 or choice > len(functions):
        choice = 0

    if choice == 0:
        for func_class, coord_count, flag, min_range, max_range, individual_step in functions:
            print(f"\nOptimization function: {func_class.__name__} with dimension: {coord_count}")

            fss = FishSchoolSearch(iterations_num, population_size, individual_step,
                                   func_class, selected_optimum, coord_count, weight_scale, min_range, max_range)
            best_fish, coords, neval = fss.run()

            # print("School results:")
            # for i, fish in enumerate(fss.fishes):
            #     print(f"Fish {i} result: position {fish.init_position}, fitness {fish.init_fitness}")

            print(f"Neval: {neval}")
            print(f"Best X: {best_fish.init_position}")
            print(f"Best F(X): {best_fish.init_fitness}")

            draw_2d([(pos[0], pos[1]) for pos in coords],
                    (best_fish.init_position[0], best_fish.init_position[1]),
                    iterations_num,
                    lambda x: func_class.calculate([x[0], x[1]] + [0] * (coord_count - 2) if coord_count > 2 else x),
                    min_range, max_range, flag)
    else:
        func_class, coord_count, flag, min_range, max_range, individual_step = functions[choice - 1]

        print(f"\nOptimization function: {func_class.__name__} with dimension: {coord_count}")

        fss = FishSchoolSearch(iterations_num, population_size, individual_step,
                               func_class, selected_optimum, coord_count, weight_scale, min_range, max_range)
        best_fish, coords, neval = fss.run()

        # print("School results:")
        # for i, fish in enumerate(fss.fishes):
        #     print(f"Fish {i} result: position {fish.init_position}, fitness {fish.init_fitness}")

        print(f"Neval: {neval}")
        print(f"Best X: {best_fish.init_position}")
        print(f"Best F(X): {best_fish.init_fitness}")

        draw_2d([(pos[0], pos[1]) for pos in coords],
                (best_fish.init_position[0], best_fish.init_position[1]),
                iterations_num,
                lambda x: func_class.calculate([x[0], x[1]] + [0] * (coord_count - 2) if coord_count > 2 else x),
                min_range, max_range, flag)


if __name__ == "__main__":
    main()
