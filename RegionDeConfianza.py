import numpy as np


def dogleg_exacto(f, grad_f, hess_f, x0, delta0, max_delta, eta, max_iter=100, tol=1e-6):
    """
    Método de Dogleg

    Parameters:
    - f: Función a minimizar.
    - grad_f: Gradiente de la funció.
    - hess_f: Hessiana de la función.
    - x0: Punto inicial.
    - delta0: Radio inicial.
    - max_delta: Radio máximo.
    - eta: Umbral de aceptación.
    - max_iter: Iteraciones máximas.
    - tol: Tolerancia.

    Returns:
    - x: Punto optimizado.
    - f(x): Valor de la función en el punto optimizado.
    """
    def dogleg_sub(grad, hess, delta):
        """
        Subrutina de Dogleg para obtener el paso.
        """
        # Punto de Cauchy
        p_u = - (np.dot(grad, grad) / np.dot(np.dot(grad, hess), grad)) * grad

        # Paso de Newton
        try:
            p_b = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            p_b = p_u  # Si la matriz es singular se usa solo el punto de Cauchy

        # Si P_b está dentro del radio de confianza
        if np.linalg.norm(p_b) <= delta:
            return p_b

        # Si P_u está fuera del radio de confianza
        if np.linalg.norm(p_u) >= delta:
            return delta * p_u / np.linalg.norm(p_u) # Se escala al radio de confianza

        # Si p_u esta dentro y p_u esta fuera, se calcula la intersección con el radio de confianza de la trayectoria
        p_diff = p_b - p_u
        a = np.dot(p_diff, p_diff)
        b = 2 * np.dot(p_u, p_diff)
        c = np.dot(p_u, p_u) - delta ** 2
        tau = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return p_u + tau * p_diff

    x = x0
    delta = delta0

    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)

        # Usamos la subrutina de Dogleg para obtener el paso
        p_k = dogleg_sub(grad, hess, delta)

        # Computamos rho_k
        reduccion_real = f(x) - f(x + p_k)
        reuccion_predecida = -np.dot(grad, p_k) - 0.5 * np.dot(np.dot(p_k, hess), p_k)
        rho_k = reduccion_real / reuccion_predecida if reuccion_predecida != 0 else 0

        # Actualizamos el radio de confianza

        #Caso 1
        if rho_k < 0.25:
            delta = 0.25 * np.linalg.norm(p_k)

        #Caso 2
        elif rho_k > 0.75 and np.linalg.norm(p_k) == delta:
            delta = min(2 * delta, max_delta)
        

        # Actualizamos x si se pasa el umbral
        if rho_k > eta:
            x = x + p_k

        # Revisamos convergencia en base a la tolerancia
        if np.linalg.norm(grad) < tol:
            break

    return x, f(x)

def dogleg_debug_exacto(f, grad_f, hess_f, x0, delta0, max_delta, eta, max_iter=100, tol=1e-6):
    """
    Método de Dogleg

    Parameters:
    - f: Función a minimizar.
    - grad_f: Gradiente de la funció.
    - hess_f: Hessiana de la función.
    - x0: Punto inicial.
    - delta0: Radio inicial.
    - max_delta: Radio máximo.
    - eta: Umbral de aceptación.
    - max_iter: Iteraciones máximas.
    - tol: Tolerancia.

    Returns:
    - x: Punto optimizado.
    - f(x): Valor de la función en el punto optimizado.
    - x_k: Puntos en cada iteración.
    - f(x_k): Valores de la función en cada iteración.
    - delta_k: Radios de confianza en cada iteración.
    - rho_k_list: Valores de rho en cada iteración.
    - p_k_list: Pasos en cada iteración.
    - p_k_tipo: Tipo de paso en cada iteración.
    - iter: Número de iteraciones.
    """
    def dogleg_sub(grad, hess, delta):
        """
        Subrutina de Dogleg para obtener el paso.
        """
        # Punto de Cauchy
        p_u = - (np.dot(grad, grad) / np.dot(np.dot(grad, hess), grad)) * grad

        # Paso de Newton
        try:
            p_b = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            p_b = p_u  # Si la matriz es singular se usa solo el punto de Cauchy

        # Si P_b está dentro del radio de confianza
        if np.linalg.norm(p_b) <= delta:
            return p_b, p_u, p_b, 'Paso Completo'

        # Si P_u está fuera del radio de confianza
        if np.linalg.norm(p_u) >= delta:
            return delta * p_u / np.linalg.norm(p_u), p_u, p_b, 'Escalado' # Se escala al radio de confianza

        # Si p_u esta dentro y p_u esta fuera, se calcula la intersección con el radio de confianza de la trayectoria
        p_diff = p_b - p_u
        a = np.dot(p_diff, p_diff)
        b = 2 * np.dot(p_u, p_diff)
        c = np.dot(p_u, p_u) - delta ** 2
        tau = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return p_u + tau * p_diff, p_u, p_b, 'Intersección'

    x = x0
    delta = delta0

    x_k = []
    f_k = []
    delta_k = []
    rho_k_list = []
    p_k_list = []
    p_u_list = []
    p_b_list = []
    p_k_tipo = []
    iter = 0

    for _ in range(max_iter):

        iter += 1
        x_k.append(x)
        f_k.append(f(x))
        delta_k.append(delta)

        grad = grad_f(x)
        hess = hess_f(x)

        # Usamos la subrutina de Dogleg para obtener el paso
        p_k, p_u, p_b, tipo = dogleg_sub(grad, hess, delta)

        p_u_list.append(p_u)
        p_b_list.append(p_b)
        p_k_tipo.append(tipo)
        p_k_list.append(p_k)

        # Computamos rho_k
        reduccion_real = f(x) - f(x + p_k)
        reuccion_predecida = -np.dot(grad, p_k) - 0.5 * np.dot(np.dot(p_k, hess), p_k)
        rho_k = reduccion_real / reuccion_predecida if reuccion_predecida != 0 else 0

        rho_k_list.append(rho_k)

        # Actualizamos el radio de confianza

        #Caso 1
        if rho_k < 0.25:
            delta = 0.25 * np.linalg.norm(p_k)

        #Caso 2
        elif rho_k > 0.75 and np.linalg.norm(p_k) == delta:
            delta = min(2 * delta, max_delta)
        

        # Actualizamos x si se pasa el umbral
        if rho_k > eta:
            x = x + p_k

        # Revisamos convergencia en base a la tolerancia
        if np.linalg.norm(grad) < tol:
            break

    return x, f(x), x_k, f_k, delta_k, rho_k_list, p_k_list, p_u_list, p_b_list, p_k_tipo, iter



def dogleg_approx(f, x0, delta0, max_delta, eta, max_iter=100, tol=1e-6):
    """
    Método de Dogleg

    Parameters:
    - f: Función a minimizar.
    - x0: Punto inicial.
    - delta0: Radio inicial.
    - max_delta: Radio máximo.
    - eta: Umbral de aceptación.
    - max_iter: Iteraciones máximas.
    - tol: Tolerancia.

    Returns:
    - x: Punto optimizado.
    - f(x): Valor de la función en el punto optimizado.
    """
    def gradiente_approx(f, x, epsilon):
        """
        Aproximar gradiente con diferencias finitas.
        """
        n = len(x)
        grad = np.zeros(n)
        for i in range(n):
            x_forward = np.copy(x)
            x_forward[i] += epsilon
            x_backward = np.copy(x)
            x_backward[i] -= epsilon
            grad[i] = (f(x_forward) - f(x_backward)) / (2 * epsilon)
        return grad

    def hessiana_approx(f, x, epsilon):
        """
        Aproximar hessiana con diferencias finitas.
        """
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_ij_plus = np.copy(x)
                x_ij_minus = np.copy(x)
                x_i_plus = np.copy(x)
                x_i_minus = np.copy(x)

                x_ij_plus[i] += epsilon
                x_ij_plus[j] += epsilon
                x_ij_minus[i] += epsilon
                x_ij_minus[j] -= epsilon
                x_i_plus[i] += epsilon
                x_i_minus[i] -= epsilon

                hess[i, j] = (
                    f(x_ij_plus) - f(x_ij_minus) - f(x_i_plus) + f(x_i_minus)
                ) / (4 * epsilon**2)
        return hess
    
    def dogleg_sub(grad, hess, delta):
        """
        Subrutina de Dogleg para obtener el paso.
        """
        # Punto de Cauchy
       
        p_u = - (np.dot(grad, grad) / np.dot(np.dot(grad, hess), grad)) * grad
        

        # Paso de Newton
        try:
            p_b = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            p_b = p_u  # Si la matriz es singular se usa solo el punto de Cauchy

        # Si P_b está dentro del radio de confianza
        if np.linalg.norm(p_b) <= delta:
            return p_b

        # Si P_u está fuera del radio de confianza
        if np.linalg.norm(p_u) >= delta:
            return delta * p_u / np.linalg.norm(p_u) # Se escala al radio de confianza

        # Si p_u esta dentro y p_u esta fuera, se calcula la intersección con el radio de confianza de la trayectoria
        p_diff = p_b - p_u
        a = np.dot(p_diff, p_diff)
        b = 2 * np.dot(p_u, p_diff)
        c = np.dot(p_u, p_u) - delta ** 2
        tau = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return p_u + tau * p_diff

    x = x0
    delta = delta0
 

    for _ in range(max_iter):
        grad = gradiente_approx(f, x, 1e-6)
        hess = hessiana_approx(f, x, 1e-6)
   

        # Usamos la subrutina de Dogleg para obtener el paso
        p_k = dogleg_sub(grad, hess, delta)

        # Computamos rho_k
        reduccion_real = f(x) - f(x + p_k)
        reuccion_predecida = -np.dot(grad, p_k) - 0.5 * np.dot(np.dot(p_k, hess), p_k)
        rho_k = reduccion_real / reuccion_predecida if reuccion_predecida != 0 else 0

        # Actualizamos el radio de confianza

        #Caso 1
        if rho_k < 0.25:
            delta = 0.25 * np.linalg.norm(p_k)

        #Caso 2
        elif rho_k > 0.75 and np.linalg.norm(p_k) == delta:
            delta = min(2 * delta, max_delta)
        

        # Actualizamos x si se pasa el umbral
        if rho_k > eta:
            x = x + p_k

        # Revisamos convergencia en base a la tolerancia
        if np.linalg.norm(grad) < tol:
            break

    return x, f(x)

def dogleg_debug_approx(f, x0, delta0, max_delta, eta, max_iter=100, tol=1e-6):
    """
    Método de Dogleg

    Parameters:
    - f: Función a minimizar.
    - x0: Punto inicial.
    - delta0: Radio inicial.
    - max_delta: Radio máximo.
    - eta: Umbral de aceptación.
    - max_iter: Iteraciones máximas.
    - tol: Tolerancia.

    Returns:
    - x: Punto optimizado.
    - f(x): Valor de la función en el punto optimizado.
    - x_k: Puntos en cada iteración.
    - f(x_k): Valores de la función en cada iteración.
    - delta_k: Radios de confianza en cada iteración.
    - rho_k_list: Valores de rho en cada iteración.
    - p_k_list: Pasos en cada iteración.
    - p_k_tipo: Tipo de paso en cada iteración.
    - iter: Número de iteraciones.
    """
    def gradiente_approx(f, x, epsilon):
        """
        Aproximar gradiente con diferencias finitas.
        """
        n = len(x)
        grad = np.zeros(n)
        for i in range(n):
            x_forward = np.copy(x)
            x_forward[i] += epsilon
            x_backward = np.copy(x)
            x_backward[i] -= epsilon
            grad[i] = (f(x_forward) - f(x_backward)) / (2 * epsilon)
        return grad

    def hessiana_approx(f, x, epsilon):
        """
        Aproximar hessiana con diferencias finitas.
        """
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_ij_plus = np.copy(x)
                x_ij_minus = np.copy(x)
                x_i_plus = np.copy(x)
                x_i_minus = np.copy(x)

                x_ij_plus[i] += epsilon
                x_ij_plus[j] += epsilon
                x_ij_minus[i] += epsilon
                x_ij_minus[j] -= epsilon
                x_i_plus[i] += epsilon
                x_i_minus[i] -= epsilon

                hess[i, j] = (
                    f(x_ij_plus) - f(x_ij_minus) - f(x_i_plus) + f(x_i_minus)
                ) / (4 * epsilon**2)
        return hess
    
    def dogleg_sub(grad, hess, delta):
        """
        Subrutina de Dogleg para obtener el paso.
        """
        # Punto de Cauchy
        p_u = - (np.dot(grad, grad) / np.dot(np.dot(grad, hess), grad)) * grad

        # Paso de Newton

        try:
            p_b = np.dot(-np.linalg.inv(hess), grad)
        except np.linalg.LinAlgError:
            p_b = p_u  # Si la matriz es singular se usa solo el punto de Cauchy

        # Si P_b está dentro del radio de confianza
        if np.linalg.norm(p_b) <= delta:
            return p_b, p_u, p_b, 'Paso Completo'

        # Si P_u está fuera del radio de confianza
        if np.linalg.norm(p_u) >= delta:
            return delta * p_u / np.linalg.norm(p_u), p_u, p_b, 'Escalado' # Se escala al radio de confianza

        # Si p_u esta dentro y p_u esta fuera, se calcula la intersección con el radio de confianza de la trayectoria
        p_diff = p_b - p_u
        a = np.dot(p_diff, p_diff)
        b = 2 * np.dot(p_u, p_diff)
        c = np.dot(p_u, p_u) - delta ** 2
        tau = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return p_u + tau * p_diff, p_u, p_b, 'Intersección'

    x = x0
    delta = delta0

    x_k = []
    f_k = []
    delta_k = []
    rho_k_list = []
    p_k_list = []
    p_u_list = []
    p_b_list = []
    p_k_tipo = []
    iter = 0

    for _ in range(max_iter):

        iter += 1
        x_k.append(x)
        f_k.append(f(x))
        delta_k.append(delta)

        grad = gradiente_approx(f, x, 1e-6)
        hess = hessiana_approx(f, x, 1e-6)

        # Usamos la subrutina de Dogleg para obtener el paso
        p_k, p_u, p_b, tipo = dogleg_sub(grad, hess, delta)

        p_u_list.append(p_u)
        p_b_list.append(p_b)
        p_k_tipo.append(tipo)
        p_k_list.append(p_k)

        # Computamos rho_k
        reduccion_real = f(x) - f(x + p_k)
        reuccion_predecida = -np.dot(grad, p_k) - 0.5 * np.dot(np.dot(p_k, hess), p_k)
        rho_k = reduccion_real / reuccion_predecida if reuccion_predecida != 0 else 0

        rho_k_list.append(rho_k)

        # Actualizamos el radio de confianza

        #Caso 1
        if rho_k < 0.25:
            delta = 0.25 * np.linalg.norm(p_k)

        #Caso 2
        elif rho_k > 0.75 and np.linalg.norm(p_k) == delta:
            delta = min(2 * delta, max_delta)
        

        # Actualizamos x si se pasa el umbral
        if rho_k > eta:
            x = x + p_k

        # Revisamos convergencia en base a la tolerancia
        if np.linalg.norm(grad) < tol:
            break

    return x, f(x), x_k, f_k, delta_k, rho_k_list, p_k_list, p_u_list, p_b_list, p_k_tipo, iter




"""
# Uso de la función
def f(x):
    return x[0]**2 + 2*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 4*x[1]])

def hess_f(x):
    return np.array([[2, 0], [0, 4]])

x0 = np.array([1.0, 1.0])
delta0 = 1.0
max_delta = 2.0
eta = 0.1

optimized_x, optimized_f = dog(f, grad_f, hess_f, x0, delta0, max_delta, eta)
print("Optimized x:", optimized_x)
print("Optimized f(x):", optimized_f)
"""

def rosenbrock(x): 
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([400*x[0]**3 - 400*x[0]*x[1]+2*x[0]-2, 200*(x[1] - x[0]**2)])

def rosenbrock_hess(x):
    return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]], [-400*x[0], 200]])

x, fx, x_k, f_k, delta_k, rho_k_list, p_k_list, p_u_list, p_b_list, p_k_tipo, iter = dogleg_debug_exacto(rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array([0.0, 0.0]), 1.0, 2.0, 0.1)
#x, fx, x_k, f_k, delta_k, rho_k_list, p_k_list, p_u_list, p_b_list, p_k_tipo, iter = dogleg_debug_approx(rosenbrock, np.array([0.0, 0.0]), 1.0, 2.0, 0.1, 10000, 1e-6)
print(x)
print(fx)
print(iter)