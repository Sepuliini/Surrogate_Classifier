import numpy as np
import pandas as pd
import os
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
from scipy.stats.distributions import norm

import numpy as np
import pandas as pd
import os
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
from scipy.stats.distributions import norm

problems = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
num_vars_list = [6, 10, 20, 30]
num_obj_list = [3, 4, 5] 
num_samples_list = [100, 500, 1000, 2000]
distribution_types = ["uniform", "normal"]
noise_mean = 0
noise_std = 0.1

data_dir = os.path.join(os.getcwd(), "data")
dtlz_data_dir = os.path.join(data_dir, "dtlz_data")

if not os.path.exists(dtlz_data_dir):
    os.makedirs(dtlz_data_dir)

def generate_datasets(problem, num_vars, num_obj, num_samples, distribution):
    dtlz_problem = test_problem_builder(problem, n_of_objectives=num_obj, n_of_variables=num_vars)
    
    if distribution == "uniform":
        sample_data = lhs(num_vars, samples=num_samples)
    elif distribution == "normal":
        sample_data = norm(loc=0.5, scale=0.15).ppf(lhs(num_vars, samples=num_samples))
        np.clip(sample_data, 0, 1, out=sample_data)

    objective_values_list = []
    for sample in sample_data:
        evaluated = dtlz_problem.evaluate(sample)
        if hasattr(evaluated, 'objectives'):
            objective_values = evaluated.objectives
            if objective_values is not None:
                objective_values_list.append(objective_values[0])
        else:
            raise TypeError("EvaluationResults object does not contain objective values.")

    data = np.hstack((sample_data, np.array(objective_values_list)))
    if data is not None:
        data += np.random.normal(noise_mean, noise_std, data.shape)

    columns = [f"x{i}" for i in range(1, num_vars + 1)] + [f"f{j}" for j in range(1, num_obj + 1)]
    df = pd.DataFrame(data, columns=columns)
    filename = f"{problem}_{num_vars}var_{num_obj}obj_{num_samples}samples_{distribution}.csv"
    df.to_csv(os.path.join(dtlz_data_dir, filename), index=False)
    print(f"Saved dataset to {os.path.join(dtlz_data_dir, filename)}")

for problem in problems:
    for num_vars in num_vars_list:
        for num_obj in num_obj_list:  # Iterate over the num_obj_list
            for num_samples in num_samples_list:
                for distribution in distribution_types:
                    generate_datasets(problem, num_vars, num_obj, num_samples, distribution)
