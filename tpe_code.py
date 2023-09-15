
import numpy as np


class TPE:
    def __init__(self, direction):
        self.__direction = 1 if direction.lower() == 'maximize' else -1 if direction.lower() == 'minimize' else 0
        if self.__direction == 0:
            self.__direction = 1
            raise ValueError ('direction must be maximize or minimize')
        self.__direction_string = direction
        self.__iterations = 0
    @property
    def n_trials(self):
        return self.__iterations
    @property
    def trials(self):
        if hasattr(self, '_TPE__trials_parameters_dict_array'):
            return np.array(list(zip(self.__trials_parameters_dict_array, self.__values*self.__direction)), dtype = object)
            
        else:
            return np.array([np.empty(0), np.empty(0)])
    @property
    def best_trial(self):
        if self.__array:
            return np.array([self.__best_parameter, self.__best_value], dtype = object)
        else:
            return np.array([self.__best_parameter_dict, self.__best_value], dtype = object)
        
    def __dict_to_array(self, bounds):
        bounds_array = np.array(list(bounds.values()))
        return bounds_array
    
    def __array_to_dict(self, values, keys, types, use_types, types_array):
        if use_types:
            if types_array:
                return {key: type_(value) for key, value, type_ in zip(keys, values, types)}
            else:
                return {key: types(value) for key, value in zip(keys, values)}
        else:
            return {key: value for key, value in zip(keys, values)}
    
    def __filter_pos(self, params, params_tested):
        mask = np.all(np.expand_dims(params, 1) == params_tested, axis = 2)
        elements = np.sum(mask, axis=1) == 0
        return elements
    
    def __kd(self, X, x, bandwidth = 1):
        diff = (np.expand_dims(x, axis=1) - X) / bandwidth
        kde = (1 / (bandwidth * np.sqrt(2 * np.pi))) ** x.shape[1] * np.exp(-0.5 * np.sum(diff**2, axis=2))
        kde_values = np.sum(kde, axis=1)
        return kde_values / X.shape[0]
    
    def __generate_random_params(self, bounds, num_samples, steps):
        num_params = len(bounds)
        samples = np.random.random_sample(size=(num_samples, len(bounds)))
        samples = samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        random_params = np.round(samples.reshape((num_samples, num_params)) / steps) * steps

        return random_params
    
    def __ei(self, params, kde_good, kde_bad, params_tested, bandwidth):
        

        good_density = self.__kd(kde_good, params)
        
        bad_density = self.__kd(kde_bad, params)

        density = good_density / ((good_density + bad_density) + 1e-16)
        
        density = np.where(self.__filter_pos(params, params_tested), density, 0)
        
        return -density
    
    
    
    def __pso(self, func, bounds, steps, args = (), num_particles=30, max_iter=100, w=0.5, c1=1, c2=2, tol = 100):
        dim = bounds.shape[0]
        particles = self.__generate_random_params(bounds, num_particles, steps)
        velocities = np.zeros((num_particles, dim))
        
        best_positions = np.copy(particles)
        
        best_fitness = func(particles, *args)
        

        swarm_best_position = best_positions[np.argmin(best_fitness)]
        swarm_best_fitness = np.min(best_fitness)
        count = 0
        
        for i in range(max_iter):
            r1 = np.random.uniform(0, 1, (num_particles, dim))
            r2 = np.random.uniform(0, 1, (num_particles, dim))
            velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)
            velocities = np.round(velocities / steps) * steps
            particles += velocities
            fitness_values = func(particles, *args)
            
            particles = np.clip(particles, bounds[:, 0], bounds[:, 1])
            
            
            improved_indices = np.where(fitness_values < best_fitness)
            best_positions[improved_indices] = particles[improved_indices]
            best_fitness[improved_indices] = fitness_values[improved_indices]
            if np.min(fitness_values) < swarm_best_fitness:
                swarm_best_position = best_positions[np.argmin(best_fitness), :]
                swarm_best_fitness = np.min(fitness_values)
                count = 0
            else:
                count += 1
            if count >= tol:
                break

        return swarm_best_position, swarm_best_fitness
    
    
    def __get_next_parameters(self, kde_good, kde_bad, bounds, steps, parameters_tested, pso_population = 10, pso_iterations = 100, pso_tol = 10, w = 0.5, c1 = 1.0, c2 = 2.0, tol = 10, bandwidth = 1.0):
        
        params, result = self.__pso(self.__ei, bounds, steps, (kde_good, kde_bad, parameters_tested, bandwidth), pso_population, pso_iterations, w, c1, c2, tol)
        return params
    

        
    def optimize(self, function, bounds, iterations = 100, debug = True, bandwidth = 1.0, gamma = 0.1, pso_population = 10, pso_iterations = 100, pso_tol = 10, pso_w = 0.5, pso_c1 = 1.0, pso_c2 = 2.0, types = None):
        """
        Parameters
        ----------
        function : optimize
            
        bounds : numpy array or dict
            Search space and step size, it can be a numpy array or a dict but it will be faster if it is a numpy array especially if there are several parameters, example: 
                bounds = {'x': (-50, 50, 1), 'y': (-50, 50, 1)}
                bounds = np.array([(-50, 50, 1), (-50, 50, 1)])
            The first element is the lower bound, the second is the upper bound, and the third is the step size.
        iterations : int, optional
            Number of iterations. The default is 100.
        debug : bool, optional
            If it is True it will print info about each iteration. The default is True.
        bandwidth : float, optional
            Bandwidth for kernel density. The default is 1.0.
        gamma : float, optional
            Gamma value to split the parameters into good and bad.. The default is 0.1.
        pso_population : int, optional
            Number of particles for the pso algorithm. The default is 10.
        pso_iterations : int, optional
            Number of iterations for the pso algorithm. The default is 100.
        pso_tol : int, optional
            Maximum number of iterations without improvement before stopping the pso algorithm. The default is 10.
        pso_w : float, optional
            Weight value for the pso algorithm. The default is 0.5.
        pso_c1 : float, optional
            c1 value for the pso algorithm. The default is 1.0.
        pso_c2 : float, optional
            c2 value for the pso algorithm. The default is 2.0.
        types : numpy array or types like int or float, optional
            Types of each parameter or for all the parameters if types is only one it will be faster. The default is None.
            Examples:
                bounds = np.array([(-50, 50, 1), (-50, 50, 1)])
                types = np.array([int, int])
                -----------------------------------------------
                bounds = np.array([(-50, 50, 1), (-50, 50, 1)])
                types = int
        """
        values = np.full((iterations), np.inf)
        
        use_types = False
        
        types_array = False
        
        if type(types) != type(None):
            use_types = True
            if type(types) == int or type(types) == float:
                types_array = False
            if type(types) == np.ndarray:
                types_array = True
        
        if type(bounds) == dict:
            values_keys = np.array(list(bounds.keys()))
            bounds = self.__dict_to_array(bounds).astype(np.float64)
            array = False
            
        elif not type(bounds) == np.ndarray:
            raise ValueError ('bounds must be a dict or a numpy array')
        else:
            array = True
        
        parameters = np.full((iterations, bounds.shape[0]), np.inf)
        
        parameters_dict = np.full(iterations, np.inf, dtype = object)
        
        steps = bounds[:, 2].astype(np.float64)
        bounds = bounds[:, :2].astype(np.float64)
        
        types_func = lambda param, type_: type_(param)
        
        self.__array = array
        
        if hasattr(self, '_TPE__trials_parameters'):
            self.__trials_parameters_dict_array = np.concatenate((self.__trials_parameters_dict_array, parameters_dict))
            self.__trials_parameters = np.concatenate((self.__trials_parameters, parameters))
        else:
            self.__trials_parameters_dict_array = parameters_dict
            self.__trials_parameters = parameters
        if hasattr(self, '_TPE__values'):
            self.__values = np.concatenate((self.__values, values))
        else:
            self.__values = values
        
        
        for _ in range(iterations):
            if gamma*self.__iterations < 1:
                next_parameter = self.__generate_random_params(bounds, 1, steps)[0]
            else:
                
                parameters = self.__trials_parameters[:self.__iterations]
                values = self.__values[:self.__iterations]
                
                sorted_indices = np.argsort(values)

                index = int(values.shape[0] * gamma)

                top_indices = sorted_indices[-index:]
                bottom_indices = sorted_indices[index:]
                
                top_parameters_x = parameters[top_indices]
                bottom_parameters_x = parameters[bottom_indices]

                
                
                next_parameter = self.__get_next_parameters(top_parameters_x, bottom_parameters_x, bounds, steps, parameters, pso_population, pso_iterations, pso_tol, pso_w, pso_c1, pso_c2, pso_tol, bandwidth)
            
            if not array:
                
                next_parameter_dict_array = self.__array_to_dict(next_parameter, values_keys, types, use_types, types_array)
                
                value = function(**next_parameter_dict_array)
                
            else:
                if use_types:
                    if types_array:
                        next_parameter_dict_array = np.vectorize(types_func)(next_parameter, types)
                    else:
                        next_parameter_dict_array = next_parameter.astype(types)
                else:
                    next_parameter_dict_array = next_parameter
                    
                value = function(*next_parameter_dict_array)
            
            self.__trials_parameters[self.__iterations] = next_parameter
            self.__trials_parameters_dict_array[self.__iterations] = next_parameter_dict_array
            self.__values[self.__iterations] = value*self.__direction
            best_index = np.argmax(self.__values[:self.__iterations+1])
            self.__best_parameter = self.__trials_parameters[:self.__iterations+1][best_index]
            self.__best_value = self.__values[:self.__iterations+1][best_index]
    
            self.__best_parameter_dict = self.__trials_parameters_dict_array[best_index]
            
            if debug:
                print('\n', 
                      'Iteration', self.__iterations, '\n', 
                      'Parameters: ', next_parameter_dict_array, '\n', 
                      'Result: ', value, '\n', 
                      'Best parameters: ', self.__trials_parameters_dict_array[best_index], '\n', 
                      'Best result: ', self.__best_value*self.__direction)
            
            self.__iterations += 1


if __name__ == '__main__':

    def func(x, y):
        term1 = -0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))
        term2 = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
        value = -20.0 * np.exp(term1) - np.exp(term2) + 20.0 + np.e
        return -value
    

    #bounds = {'x': (-50, 50, 1), 'y': (-50, 50, 1)}
    bounds = np.array([(-50, 50, 1), (-50, 50, 1)])
    types = np.array([int, int])
    tpe_object = TPE('maximize')
    
    tpe_object.optimize(func, bounds, 100, True, types = types)
    
    print(tpe_object.best_trial)

    import optuna

    def objective_function__(trial):
        x = trial.suggest_int('x', -50, 50)
        y = trial.suggest_int('y', -50, 50)
        return func(x, y)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_function__, n_trials=100,)
    





    
